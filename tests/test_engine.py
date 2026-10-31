from datetime import date

import pytest

from claim_stress_tester.engine import analyze, check_paths, parse_claims, syntheticity
from claim_stress_tester.models import FalsificationPath, Finding


def ev(id, method="interview", population="admins", n=8, stance="supports", **kw):
    return {"id": id, "summary": f"item {id}", "method": method, "population": population,
            "n": n, "stance": stance, **kw}


def sound(**overrides):
    """A claim that fires no rule. Each test breaks one thing."""
    claim = {
        "id": "c1",
        "statement": "Admins who skip the import step report slower first-week setup.",
        "population": "admins",
        "date": "2026-09-01",
        "would_be_wrong_if": "Admins who skip import set up as fast as those who do.",
        "limitations": "Self-selected sample.",
        "evidence": [ev("e1"), ev("e2", method="survey", n=40)],
    }
    claim.update(overrides)
    return claim


def rule_ids(doc, **kw):
    return [f.rule_id for f in analyze(doc, **kw).findings]


def test_sound_claim_fires_nothing():
    assert rule_ids(sound()) == []


@pytest.mark.parametrize("word", ["All", "always", "never", "Every", "no one"])
def test_s01_universal_quantifier(word):
    assert "S01" in rule_ids(sound(statement=f"{word} admins struggle with import."))


def test_s02_causal_without_experiment():
    doc = sound(statement="Skipping import causes slower setup.")
    f = next(f for f in analyze(doc).findings if f.rule_id == "S02")
    assert f.severity == "high" and f.evidence_ids == ("e1", "e2")


def test_s02_silent_with_experiment():
    doc = sound(statement="Skipping import causes slower setup.",
                evidence=[ev("e1"), ev("e2", method="A/B test", n=400)])
    assert "S02" not in rule_ids(doc)


def test_s03_scope_mismatch():
    doc = sound(population="all users")
    f = next(f for f in analyze(doc).findings if f.rule_id == "S03")
    assert "'admins'" in f.message


def test_s03_case_and_spacing_insensitive():
    assert "S03" not in rule_ids(sound(population="  Admins "))


def test_s04_too_few_items_is_high():
    f = next(f for f in analyze(sound(evidence=[ev("e1")])).findings if f.rule_id == "S04")
    assert f.severity == "high"


def test_s04_low_total_n_is_medium():
    doc = sound(evidence=[ev("e1", n=3), ev("e2", method="survey", n=4)])
    f = next(f for f in analyze(doc).findings if f.rule_id == "S04")
    assert f.severity == "medium"


def test_s04_low_n_needs_all_sizes_known():
    doc = sound(evidence=[ev("e1", n=3), ev("e2", method="survey", n=None)])
    assert "S04" not in rule_ids(doc)


def test_s05_single_method():
    assert "S05" in rule_ids(sound(evidence=[ev("e1"), ev("e2")]))


def test_s06_stale_evidence_against_claim_date():
    doc = sound(evidence=[ev("e1", collected="2023-01"), ev("e2", method="survey", n=40, collected="2026-06")])
    f = next(f for f in analyze(doc).findings if f.rule_id == "S06")
    assert f.evidence_ids == ("e1",)


def test_s06_uses_as_of_only_when_claim_has_no_date():
    doc = sound(date=None, evidence=[ev("e1", collected="2023-01"), ev("e2", method="survey", n=40)])
    assert "S06" not in rule_ids(doc)
    assert "S06" in rule_ids(doc, as_of=date(2026, 9, 1))


def test_s07_missing_falsification_condition():
    assert "S07" in rule_ids(sound(would_be_wrong_if=""))


def test_s08_unaddressed_counter_evidence():
    doc = sound(evidence=sound()["evidence"] + [ev("e3", stance="against")])
    assert "S08" in rule_ids(doc)


def test_s08_silent_when_limitations_reference_it():
    doc = sound(evidence=sound()["evidence"] + [ev("e3", stance="against")],
                limitations="e3 shows the opposite in enterprise tenants; claim scoped to SMB.")
    assert "S08" not in rule_ids(doc)


def test_s09_intensity_without_statistic():
    assert "S09" in rule_ids(sound(statement="Admins strongly prefer guided setup."))
    assert "S09" not in rule_ids(sound(statement="Most admins (31 of 40) prefer guided setup."))


def synthetic_block(grounding="none", ancestry=None):
    return {"tool": "synthetic-interviewer", "version": "0.1.0", "model": "m",
            "human_grounding": grounding, "ancestry": ancestry or []}


def test_syntheticity_human_and_pure_synthetic():
    assert syntheticity({"summary": "x"}) == 0.0
    assert syntheticity({"_cpr_synthetic": synthetic_block()}) == 1.0


def test_syntheticity_recursive_grounding():
    seeded_on_human = {"_cpr_synthetic": synthetic_block("seeded", ["interview-07"])}
    assert syntheticity(seeded_on_human) == 0.25
    nested = {"_cpr_synthetic": synthetic_block("partial", [seeded_on_human])}
    assert syntheticity(nested) == 0.625


def test_syntheticity_cycle_is_fully_synthetic():
    block = synthetic_block("seeded")
    block["ancestry"] = [block]
    assert syntheticity({"_cpr_synthetic": block}) == pytest.approx(1.0)


def test_s10_synthetic_share_above_threshold():
    doc = sound(evidence=[ev("e1", _cpr_synthetic=synthetic_block()), ev("e2", method="survey", n=40)])
    report = analyze(doc)
    assert report.results[0].synthetic_share == 0.5
    f = next(f for f in report.findings if f.rule_id == "S10")
    assert f.evidence_ids == ("e1",)


def test_s10_silent_at_or_below_threshold():
    seeded = {"_cpr_synthetic": synthetic_block("seeded", ["interview-07"])}  # 0.25
    doc = sound(evidence=[ev("e1", **seeded), ev("e2", method="survey", n=40)])
    assert "S10" not in rule_ids(doc)


def test_findings_sorted_by_severity_then_rule():
    doc = sound(statement="All admins strongly prefer it because it is faster.",
                would_be_wrong_if=None, evidence=[ev("e1")])
    sev = [f.severity for f in analyze(doc).findings]
    assert sev == sorted(sev, key=["high", "medium", "low"].index)


def test_deterministic():
    doc = sound(statement="Every admin abandons setup because of import.", evidence=[ev("e1")])
    assert analyze(doc).to_dict() == analyze(doc).to_dict()


def test_input_shapes():
    assert len(parse_claims(sound())) == 1
    assert len(parse_claims({"claims": [sound(), sound(id="c2")]})) == 2
    assert len(parse_claims([sound()])) == 1


def test_default_ids_and_validation():
    c = parse_claims({"statement": "x", "evidence": [{"summary": "a"}]})[0]
    assert c.id == "c1" and c.evidence[0].id == "e1"
    with pytest.raises(ValueError):
        parse_claims({"statement": ""})
    with pytest.raises(ValueError):
        parse_claims({"claims": [sound(), sound()]})
    with pytest.raises(ValueError):
        parse_claims(sound(evidence=[ev("e1", stance="maybe")]))


def test_invalid_severity_rejected():
    with pytest.raises(ValueError):
        Finding(rule_id="X", severity="critical", message="m", probe="p")


def test_every_finding_has_a_probe():
    doc = sound(statement="All admins strongly prefer it because it is faster.",
                population="users", would_be_wrong_if=None, evidence=[ev("e1")])
    assert all(f.probe.endswith("?") for f in analyze(doc).findings)


def test_check_paths_grades_anchors():
    result = analyze(sound(would_be_wrong_if=None)).results[0]  # S07 fires
    paths = [
        FalsificationPath("a", "w", targets=("e1", "S07")),
        FalsificationPath("b", "w"),
        FalsificationPath("c", "w", targets=("e1", "e9", "S02")),
    ]
    checked = check_paths(result, paths)
    assert [p.check for p in checked] == ["anchored", "extends", "unanchored"]
    assert checked[2].unknown_targets == ("e9", "S02")
    assert paths[0].check is None  # inputs untouched
