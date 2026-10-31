import json
from datetime import date
from pathlib import Path

import pytest

from claim_stress_tester.engine import RULES, analyze

EXAMPLES = sorted((Path(__file__).resolve().parent.parent / "examples").glob("*.json"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_loads(path):
    analyze(json.loads(path.read_text()))


def test_examples_exercise_every_rule():
    fired = {f.rule_id for p in EXAMPLES
             for f in analyze(json.loads(p.read_text()), as_of=date(2026, 10, 31)).findings}
    assert fired == set(RULES)


def test_sound_example_fires_nothing():
    path = next(p for p in EXAMPLES if p.name == "sound_claim.json")
    assert analyze(json.loads(path.read_text())).findings == []
