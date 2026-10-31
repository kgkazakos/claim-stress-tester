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
STANCES = ("supports", "against")
CHECKS = ("anchored", "extends", "unanchored")

# Ouroboros Boundary: warn when more than this share of a claim's evidence is synthetic.
SYNTHETIC_THRESHOLD = 0.30
# Same grounding factors as provenance-inspector, so shares agree across the tool chain.
GROUNDING_FACTORS: dict[str, float] = {"seeded": 0.75, "partial": 0.5, "none": 0.0}

STALE_MONTHS = 24
MIN_SUPPORTING_ITEMS = 2
MIN_TOTAL_N = 12


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    summary: str
    method: str = "unspecified"
    population: Optional[str] = None
    n: Optional[int] = None
    collected: Optional[str] = None  # ISO date, YYYY-MM or YYYY-MM-DD
    stance: str = "supports"
    syntheticity: float = 0.0  # computed from _cpr_synthetic ancestry, 0.0 for human sources

    def __post_init__(self) -> None:
        if self.stance not in STANCES:
            raise ValueError(f"stance must be one of {STANCES}, got {self.stance!r}")


@dataclass(frozen=True)
class Claim:
    id: str
    statement: str
    population: Optional[str] = None
    evidence: tuple[EvidenceItem, ...] = ()
    limitations: str = ""
    would_be_wrong_if: Optional[str] = None
    date: Optional[str] = None

    @property
    def supporting(self) -> tuple[EvidenceItem, ...]:
        return tuple(e for e in self.evidence if e.stance == "supports")

    @property
    def against(self) -> tuple[EvidenceItem, ...]:
        return tuple(e for e in self.evidence if e.stance == "against")


@dataclass(frozen=True)
class Finding:
    rule_id: str
    severity: str
    message: str
    probe: str  # a falsification question the researcher can go and answer
    location: Optional[str] = None
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {self.severity!r}")


@dataclass
class FalsificationPath:
    """AI-generated line of attack. `check` is set by the engine, never by the model."""

    attack: str
    would_confirm: str
    targets: tuple[str, ...] = ()
    check: Optional[str] = None
    unknown_targets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.check is not None and self.check not in CHECKS:
            raise ValueError(f"check must be one of {CHECKS}, got {self.check!r}")


@dataclass
class ClaimResult:
    claim: Claim
    findings: list[Finding] = field(default_factory=list)
    synthetic_share: float = 0.0
    # Additive AI output, separate from computed values. None when --no-llm or on failure.
    paths: Optional[list[FalsificationPath]] = None

    def counts(self) -> dict[str, int]:
        return {s: sum(1 for f in self.findings if f.severity == s) for s in SEVERITIES}


@dataclass
class Report:
    tool: str
    version: str
    results: list[ClaimResult] = field(default_factory=list)
    model: Optional[str] = None  # set only when the AI layer produced paths
    verify: bool = False
    # Visible note when the AI layer was asked for but produced nothing ("model unavailable").
    model_note: Optional[str] = None

    @property
    def findings(self) -> list[Finding]:
        return [f for r in self.results for f in r.findings]

    def counts(self) -> dict[str, int]:
        return {s: sum(1 for f in self.findings if f.severity == s) for s in SEVERITIES}

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for r_dict, r in zip(d["results"], self.results):
            r_dict["counts"] = r.counts()
        d["counts"] = self.counts()
        return d
