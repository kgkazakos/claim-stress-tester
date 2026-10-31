import json

from claim_stress_tester import llm
from claim_stress_tester.engine import analyze

CLAIM = {
    "id": "c1",
    "statement": "Every admin abandons setup because of the import step.",
    "population": "admins",
    "evidence": [{"id": "e1", "summary": "Interviews", "method": "interview", "population": "admins", "n": 6}],
}

REPLY = json.dumps({"paths": [
    {"attack": "Admins with no legacy data never see import", "would_confirm": "Setup funnel by data source",
     "targets": ["e1", "S01"]},
    {"attack": "Drop-off peaks at billing, not import", "would_confirm": "Step-level abandonment", "targets": []},
    {"attack": "Interviewees were recruited from churned accounts", "would_confirm": "Recruitment log",
     "targets": ["e4"]},
]})


def test_default_model_used_when_none_given(monkeypatch):
    seen = {}

    def fake(prompt, model):
        seen["model"] = model
        return REPLY

    monkeypatch.setattr(llm, "_call_model", fake)
    llm.enrich(analyze(CLAIM))
    assert seen["model"] == llm.DEFAULT_MODEL


def test_model_failure_degrades_to_the_no_llm_report(monkeypatch):
    def boom(prompt, model):
        raise RuntimeError("backend down")

    monkeypatch.setattr(llm, "_call_model", boom)
    baseline = analyze(CLAIM)
    enriched = llm.enrich(analyze(CLAIM), model="any/model", verify=True, n_paths=4)
    assert enriched.to_dict() == baseline.to_dict()


def test_unparseable_reply_degrades(monkeypatch):
    monkeypatch.setattr(llm, "_call_model", lambda p, m: "I cannot help with that.")
    assert llm.enrich(analyze(CLAIM), model="x").to_dict() == analyze(CLAIM).to_dict()


def test_enrichment_never_changes_computed_values(monkeypatch):
    monkeypatch.setattr(llm, "_call_model", lambda p, m: REPLY)
    report = analyze(CLAIM)
    before = report.to_dict()
    enriched = llm.enrich(report, model="any/model", verify=True)
    assert report.to_dict() == before  # input untouched
    for a, b in zip(enriched.results, report.results):
        assert a.findings == b.findings and a.synthetic_share == b.synthetic_share
    assert enriched.model == "any/model" and enriched.verify is True


def test_verify_mode_lets_the_engine_grade(monkeypatch):
    monkeypatch.setattr(llm, "_call_model", lambda p, m: REPLY)
    paths = llm.enrich(analyze(CLAIM), model="x", verify=True, n_paths=3).results[0].paths
    assert [p.check for p in paths] == ["anchored", "extends", "unanchored"]
    assert paths[2].unknown_targets == ("e4",)


def test_default_mode_leaves_paths_unchecked(monkeypatch):
    monkeypatch.setattr(llm, "_call_model", lambda p, m: REPLY)
    paths = llm.enrich(analyze(CLAIM), model="x").results[0].paths
    assert len(paths) == llm.DEFAULT_PATHS and all(p.check is None for p in paths)


def test_reply_check_field_is_ignored(monkeypatch):
    sneaky = json.dumps({"paths": [{"attack": "a", "would_confirm": "w", "targets": ["e9"], "check": "anchored"}]})
    monkeypatch.setattr(llm, "_call_model", lambda p, m: sneaky)
    paths = llm.enrich(analyze(CLAIM), model="x", verify=True, n_paths=1).results[0].paths
    assert paths[0].check == "unanchored"


def test_parse_handles_code_fences_and_caps_count():
    text = "```json\n" + REPLY + "\n```"
    assert len(llm.parse_paths(text, 2)) == 2


def test_prompt_contains_ids_and_guards():
    prompt = llm.build_prompt(analyze(CLAIM).results[0], 4)
    assert "e1 [supports]" in prompt and "S01" in prompt
    assert "Never invent" in prompt and "personas" in prompt
    assert "Do not say whether the claim is true" in prompt
