import json

import pytest

from claim_stress_tester.engine import analyze, check_paths
from claim_stress_tester.io import load_input, render_probes, render_text, to_export
from claim_stress_tester.models import EXPORT_SCHEMA, FalsificationPath

CLAIM = {
    "id": "c1",
    "statement": "All admins abandon setup because of the import step.",
    "population": "admins",
    "evidence": [{"id": "e1", "summary": "5 interviews", "method": "interview", "population": "admins", "n": 5}],
}


def test_load_json_and_yaml(tmp_path):
    j = tmp_path / "c.json"
    j.write_text(json.dumps(CLAIM))
    y = tmp_path / "c.yaml"
    y.write_text("claims:\n  - statement: Admins prefer guided setup.\n")
    assert load_input(j)["id"] == "c1"
    assert load_input(y)["claims"][0]["statement"] == "Admins prefer guided setup."


def test_load_rejects_scalar(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('"just a string"')
    with pytest.raises(ValueError):
        load_input(p)


def test_export_is_versioned_and_has_no_synthetic_block():
    payload = to_export(analyze(CLAIM))
    assert payload["schema"] == EXPORT_SCHEMA
    assert "_cpr_synthetic" not in payload
    r = payload["report"]["results"][0]
    assert r["paths"] is None
    assert r["counts"]["high"] >= 1
    json.dumps(payload)  # serialisable


def test_text_render_shows_rules_and_probes():
    text = render_text(analyze(CLAIM))
    assert "[c1]" in text and "S01" in text and "probe:" in text
    assert "Falsification paths" not in text


def test_text_render_labels_paths_as_ai_and_engine_checked():
    report = analyze(CLAIM)
    r = report.results[0]
    r.paths = check_paths(r, [FalsificationPath("Admins with no data to import", "Setup logs", ("e1", "e7"))])
    report.model, report.verify = "test/model", True
    text = render_text(report)
    assert "AI-generated; the engine checked each one" in text
    assert "[unanchored]" in text and "chain: e7 (not found in input)" in text
    report.verify = False
    assert "not found in input: e7" in render_text(report)


def test_clean_claim_says_absence_is_not_support():
    claim = {"statement": "Admins report slower setup.", "population": "admins", "date": "2026-09",
             "would_be_wrong_if": "x", "evidence": [
                 {"id": "e1", "summary": "a", "method": "interview", "population": "admins"},
                 {"id": "e2", "summary": "b", "method": "survey", "population": "admins"}]}
    assert "not evidence the claim is true" in render_text(analyze(claim))


def test_probes_checklist():
    text = render_probes(analyze(CLAIM))
    assert text.startswith("[c1]")
    assert "[ ]" in text and "(S01)" in text
