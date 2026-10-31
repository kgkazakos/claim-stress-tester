"""Data models and paradigm constants."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

TOOL_NAME = "claim-stress-tester"
EXPORT_SCHEMA = "cpr.cst.export/1"

AI_PARADIGM = "LLM (reasoning / test-time deliberation)"
AI_PARADIGM_RATIONALE = (
    "Falsifying a research claim is adversarial, open-ended reasoning, not a lookup: it "
    "means imagining the populations, contexts, confounds and rival explanations under "
    "which the claim would fail, and saying what evidence would show it. A reasoning model "
    "spending deliberate test-time compute is the right fit for that search. A deterministic "
    "stress profile runs first and always, so the model attacks a claim whose structural "
    "weak points are already named, and verification-chain mode has the engine check every "
    "falsification path against the evidence the researcher actually supplied."
)

SEVERITIES = ("high", "medium", "low")


@dataclass(frozen=True)
class Finding:
    rule_id: str
    severity: str
    message: str
    location: Optional[str] = None

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {self.severity!r}")


@dataclass
class Report:
    tool: str
    version: str
    findings: list[Finding] = field(default_factory=list)
    # Enrichment is additive and separate from computed values. None when --no-llm or on failure.
    enrichment: Optional[str] = None

    def counts(self) -> dict[str, int]:
        return {s: sum(1 for f in self.findings if f.severity == s) for s in SEVERITIES}

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["counts"] = self.counts()
        return d
