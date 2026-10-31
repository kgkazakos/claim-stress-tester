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
    assert "Model off (--no-llm)" in r.output


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


REPLY = json.dumps({"paths": [
    {"attack": "Admins with nothing to import", "would_confirm": "Setup logs", "targets": ["e1", "S01"]},
    {"attack": "Seasonality", "would_confirm": "Month-on-month drop-off", "targets": []},
]})


def test_model_output_leads_the_text(claim_file, monkeypatch):
    monkeypatch.setattr(llm, "_call_model", lambda prompt, model: REPLY)
    r = invoke("run", claim_file, "--model", "test/model")
    assert r.exit_code == 0
    out = r.output
    assert "from test/model" in out.splitlines()[1]
    assert out.index("Falsification paths") < out.index("Verified against: stress profile")
    assert "1. [anchored] Admins with nothing to import" in out
    assert "checks: 1 anchored, 1 extends, 0 unanchored" in out


def test_verify_text_traces_each_target(claim_file, monkeypatch):
    monkeypatch.setattr(llm, "_call_model", lambda prompt, model: REPLY)
    out = invoke("run", claim_file, "--verify", "--model", "test/model").output
    assert "chain: e1 (supports, interview): Interviews" in out
    assert "chain: S01 Universal quantifier in the claim" in out
    assert "chain: no targets" in out


def test_model_unavailable_note_is_visible(claim_file, monkeypatch):
    def boom(prompt, model):
        raise RuntimeError("no key")

    monkeypatch.setattr(llm, "_call_model", boom)
    r = invoke("run", claim_file, "--model", "test/model")
    assert r.exit_code == 0
    assert "model unavailable (test/model)" in r.output.splitlines()[1]
    assert "S01" in r.output


def test_export_passes_model_flag(claim_file, monkeypatch):
    seen = {}

    def fake(prompt, model):
        seen["model"] = model
        return REPLY

    monkeypatch.setattr(llm, "_call_model", fake)
    payload = json.loads(invoke("export", claim_file, "--model", "flag/model").output)
    assert seen["model"] == "flag/model"
    assert payload["report"]["model"] == "flag/model"
    assert payload["report"]["results"][0]["paths"][0]["check"] == "anchored"
