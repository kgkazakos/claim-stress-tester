import json
import sys

import pytest
from click.testing import CliRunner

from claim_stress_tester import llm
from claim_stress_tester.cli import main
from claim_stress_tester.models import AI_PARADIGM, EXPORT_SCHEMA

CLAIM = {
    "id": "c1",
    "statement": "Every admin abandons setup because of the import step.",
    "population": "admins",
    "evidence": [
        {"id": "e1", "summary": "Interviews", "method": "interview", "population": "admins",
         "n": 6, "collected": "2023-01"},
        {"id": "e2", "summary": "Tickets show the opposite", "method": "support tickets", "stance": "against"},
    ],
}


@pytest.fixture
def claim_file(tmp_path):
    p = tmp_path / "claim.json"
    p.write_text(json.dumps(CLAIM))
    return str(p)


def invoke(*args):
    return CliRunner().invoke(main, list(args))


def test_version_and_paradigm():
    assert invoke("--version").exit_code == 0
    r = invoke("paradigm")
    assert r.exit_code == 0 and AI_PARADIGM in r.output


def test_run_text_no_llm(claim_file):
    r = invoke("run", claim_file, "--no-llm")
    assert r.exit_code == 0
    assert "S01" in r.output and "S08" in r.output
    assert "Falsification paths" not in r.output


def test_run_json_is_versioned(claim_file):
    r = invoke("run", claim_file, "--no-llm", "--format", "json")
    payload = json.loads(r.output)
    assert payload["schema"] == EXPORT_SCHEMA
    assert payload["report"]["results"][0]["paths"] is None
    assert "_cpr_synthetic" not in payload


def test_no_llm_never_imports_litellm(claim_file, monkeypatch):
    monkeypatch.delitem(sys.modules, "litellm", raising=False)
    invoke("run", claim_file, "--no-llm")
    assert "litellm" not in sys.modules


def test_as_of_enables_stale_rule(claim_file):
    assert "S06" not in invoke("run", claim_file, "--no-llm").output
    assert "S06" in invoke("run", claim_file, "--no-llm", "--as-of", "2026-09-01").output
    assert invoke("run", claim_file, "--no-llm", "--as-of", "Sept").exit_code != 0


def test_probes_command(claim_file):
    r = invoke("probes", claim_file)
    assert r.exit_code == 0 and r.output.startswith("[c1]") and "[ ]" in r.output


def test_export_to_file(claim_file, tmp_path):
    out = tmp_path / "out.json"
    r = invoke("export", claim_file, "--no-llm", "-o", str(out))
    assert r.exit_code == 0
    assert json.loads(out.read_text())["schema"] == EXPORT_SCHEMA


def test_bad_input_is_a_clean_error(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"statement": ""}')
    r = invoke("run", str(p), "--no-llm")
    assert r.exit_code == 1 and "statement" in r.output


def test_verify_mode_end_to_end(claim_file, monkeypatch):
    reply = json.dumps({"paths": [
        {"attack": "Admins with nothing to import", "would_confirm": "Setup logs", "targets": ["e1", "S03"]},
        {"attack": "Seasonality", "would_confirm": "Month-on-month drop-off", "targets": []},
    ]})
    monkeypatch.setattr(llm, "_call_model", lambda prompt, model: reply)
    r = invoke("run", claim_file, "--verify", "--model", "test/model", "--format", "json")
    paths = json.loads(r.output)["report"]["results"][0]["paths"]
    assert [p["check"] for p in paths] == ["unanchored", "extends"]  # S03 did not fire
