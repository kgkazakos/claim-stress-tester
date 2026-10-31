"""Deterministic core. Runs with no model (--no-llm).

Builds a stress profile for each claim: rule-based weak points of the claim against its
own evidence, each with a falsification probe. Same input, same findings, same order.
It never says whether a claim is true.
"""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import date
from typing import Any, Iterable, Optional

from .models import (
    GROUNDING_FACTORS,
    MIN_SUPPORTING_ITEMS,
    MIN_TOTAL_N,
    STALE_MONTHS,
    SYNTHETIC_THRESHOLD,
    TOOL_NAME,
    Claim,
    ClaimResult,
    EvidenceItem,
    FalsificationPath,
    Finding,
    Report,
)

SYNTHETIC_KEY = "_cpr_synthetic"

RULES: dict[str, str] = {
    "S01": "Universal quantifier in the claim",
    "S02": "Causal language without an experimental design",
    "S03": "Claim population wider than the evidence population",
    "S04": "Thin evidence",
    "S05": "Single method",
    "S06": "Stale evidence",
    "S07": "No falsification condition",
    "S08": "Unaddressed counter-evidence",
    "S09": "Intensity words without a reported statistic",
    "S10": "Synthetic share above the Ouroboros threshold",
}

_UNIVERSAL = re.compile(
    r"\b(all|always|never|every|everyone|everybody|nobody|no one|none|without exception)\b",
    re.IGNORECASE,
)
_CAUSAL = re.compile(
    r"\b(causes?|caused|causing|drives?|drove|driven|leads? to|led to|results? in|because|"
    r"due to|increases?|reduces?|decreases?|improves?|prevents?)\b",
    re.IGNORECASE,
)
_INTENSITY = re.compile(
    r"\b(significant(ly)?|dramatic(ally)?|clearly|most|majority|strongly|vast|"
    r"overwhelming(ly)?|substantial(ly)?)\b",
    re.IGNORECASE,
)
_STATISTIC = re.compile(
    r"(\d+(\.\d+)?\s*%|\bp\s*[<=>]|\bn\s*=\s*\d|\bCI\b|\b\d+\s+(of|out of)\s+\d+)",
    re.IGNORECASE,
)
_EXPERIMENTAL = {"experiment", "ab test", "a b test", "rct", "randomized", "randomised",
                 "randomized controlled trial", "field experiment"}

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def syntheticity(entry: Any, _path: frozenset[int] = frozenset()) -> float:
    """Share of an artifact that is synthetic, read recursively from `_cpr_synthetic`.

    Human source (no block) -> 0.0. Synthetic with no ancestry -> 1.0. Otherwise
    g * mean(parent syntheticity) + (1 - g), with g the human_grounding factor.
    A cycle in the ancestry counts as fully synthetic.
    """
    block = entry.get(SYNTHETIC_KEY) if isinstance(entry, dict) and SYNTHETIC_KEY in entry else entry
    if not isinstance(block, dict) or not any(k in block for k in ("tool", "ancestry", "human_grounding", "model")):
        return 0.0
    if id(entry) in _path:
        return 1.0
    ancestry = block.get("ancestry") or []
    if not isinstance(ancestry, list):
        ancestry = [ancestry]
    if not ancestry:
        return 1.0
    g = GROUNDING_FACTORS.get(str(block.get("human_grounding", "none")).strip().lower(), 0.0)
    path = _path | {id(entry)}
    parent_mean = sum(syntheticity(p, path) for p in ancestry) / len(ancestry)
    return round(max(0.0, min(1.0, g * parent_mean + (1.0 - g))), 6)


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def parse_evidence(raw: dict[str, Any], index: int) -> EvidenceItem:
    return EvidenceItem(
        id=str(raw.get("id") or f"e{index + 1}"),
        summary=str(raw.get("summary", "")),
        method=str(raw.get("method") or "unspecified"),
        population=raw.get("population"),
        n=_to_int(raw.get("n")),
        collected=raw.get("collected"),
        stance=str(raw.get("stance") or "supports"),
        syntheticity=syntheticity(raw),
    )


def parse_claim(raw: dict[str, Any], index: int = 0) -> Claim:
    if not str(raw.get("statement", "")).strip():
        raise ValueError(f"Claim {index + 1} has no 'statement'.")
    evidence = raw.get("evidence") or []
    if not isinstance(evidence, list):
        raise ValueError(f"Claim {index + 1}: 'evidence' must be a list.")
    return Claim(
        id=str(raw.get("id") or f"c{index + 1}"),
        statement=str(raw["statement"]).strip(),
        population=raw.get("population"),
        evidence=tuple(parse_evidence(e, i) for i, e in enumerate(evidence)),
        limitations=str(raw.get("limitations") or ""),
        would_be_wrong_if=(str(raw["would_be_wrong_if"]).strip() or None) if raw.get("would_be_wrong_if") else None,
        date=raw.get("date"),
    )


def parse_claims(doc: Any) -> list[Claim]:
    """Accept one claim object, {"claims": [...]}, or a bare list of claims."""
    if isinstance(doc, dict) and "claims" in doc:
        items = doc["claims"]
    elif isinstance(doc, list):
        items = doc
    else:
        items = [doc]
    if not isinstance(items, list) or not items:
        raise ValueError("No claims found in input.")
    claims = [parse_claim(c, i) for i, c in enumerate(items)]
    ids = [c.id for c in claims]
    if len(set(ids)) != len(ids):
        raise ValueError("Claim ids must be unique.")
    return claims


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
def _norm(text: Optional[str]) -> str:
    return re.sub(r"[\s_\-/]+", " ", (text or "").strip().lower())


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    parts = str(value).strip()[:10].split("-")
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
        day = int(parts[2]) if len(parts) > 2 else 1
        return date(year, month, day)
    except (ValueError, IndexError):
        return None


def _months_between(earlier: date, later: date) -> int:
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


def _who(claim: Claim) -> str:
    return claim.population or "the claimed population"


def _rules(claim: Claim, as_of: Optional[date]) -> Iterable[Finding]:
    loc = f"claim:{claim.id}"
    supporting = claim.supporting

    m = _UNIVERSAL.search(claim.statement)
    if m:
        yield Finding(
            "S01", "medium",
            f"The claim uses the universal '{m.group(0)}'. One counter-case is enough to break it.",
            f"Can you find a single member of {_who(claim)} for whom this does not hold?",
            loc,
        )

    m = _CAUSAL.search(claim.statement)
    if m and not any(_norm(e.method) in _EXPERIMENTAL for e in supporting):
        methods = sorted({_norm(e.method) for e in supporting}) or ["none"]
        yield Finding(
            "S02", "high",
            f"Causal language ('{m.group(0)}') with no experimental evidence "
            f"(methods: {', '.join(methods)}).",
            "What else changed at the same time, and would the outcome still differ if the "
            "proposed cause were removed or randomised?",
            loc, tuple(e.id for e in supporting),
        )

    if claim.population:
        pops = {_norm(e.population) for e in supporting if e.population}
        if pops and _norm(claim.population) not in pops:
            yield Finding(
                "S03", "high",
                f"The claim is about '{claim.population}' but the supporting evidence covers "
                f"{', '.join(repr(p) for p in sorted(pops))}.",
                f"Does the claim survive in the parts of '{claim.population}' that no evidence item sampled?",
                loc, tuple(e.id for e in supporting if e.population),
            )

    if len(supporting) < MIN_SUPPORTING_ITEMS:
        yield Finding(
            "S04", "high",
            f"{len(supporting)} supporting evidence item(s); at least {MIN_SUPPORTING_ITEMS} expected.",
            "If the one source you have were wrong or atypical, what would be left of the claim?",
            loc, tuple(e.id for e in supporting),
        )
    else:
        sizes = [e.n for e in supporting if e.n is not None]
        if sizes and len(sizes) == len(supporting) and sum(sizes) < MIN_TOTAL_N:
            yield Finding(
                "S04", "medium",
                f"Total n across supporting evidence is {sum(sizes)} (below {MIN_TOTAL_N}).",
                "Would the pattern hold with the next ten participants you have not spoken to?",
                loc, tuple(e.id for e in supporting),
            )

    methods = {_norm(e.method) for e in supporting}
    if len(supporting) >= MIN_SUPPORTING_ITEMS and len(methods) == 1:
        method = next(iter(methods))
        yield Finding(
            "S05", "medium",
            f"All {len(supporting)} supporting items use one method ('{method}').",
            f"What would a method other than '{method}' be expected to show if the claim were false?",
            loc, tuple(e.id for e in supporting),
        )

    ref = _parse_date(claim.date) or as_of
    if ref:
        stale = [e for e in supporting
                 if (d := _parse_date(e.collected)) and _months_between(d, ref) > STALE_MONTHS]
        if stale:
            yield Finding(
                "S06", "medium",
                f"{len(stale)} supporting item(s) collected more than {STALE_MONTHS} months "
                f"before {ref.isoformat()}.",
                "What has changed in the product, market or population since this evidence was collected?",
                loc, tuple(e.id for e in stale),
            )

    if not claim.would_be_wrong_if:
        yield Finding(
            "S07", "medium",
            "No falsification condition stated ('would_be_wrong_if').",
            "What observation would make you drop or narrow this claim?",
            loc,
        )

    unaddressed = [e for e in claim.against
                   if not re.search(rf"\b{re.escape(e.id)}\b", claim.limitations)]
    if unaddressed:
        yield Finding(
            "S08", "high",
            f"{len(unaddressed)} disconfirming item(s) not referenced in the limitations: "
            f"{', '.join(e.id for e in unaddressed)}.",
            "How does the claim account for this evidence, and does it need narrowing to do so?",
            loc, tuple(e.id for e in unaddressed),
        )

    m = _INTENSITY.search(claim.statement)
    if m and not _STATISTIC.search(claim.statement):
        yield Finding(
            "S09", "low",
            f"Intensity word '{m.group(0)}' with no statistic in the claim.",
            f"What number would justify '{m.group(0)}', and what number would not?",
            loc,
        )

    share = synthetic_share(claim)
    if share > SYNTHETIC_THRESHOLD:
        synth = [e for e in supporting if e.syntheticity > 0]
        yield Finding(
            "S10", "high",
            f"{share:.0%} of the supporting evidence is synthetic (threshold "
            f"{SYNTHETIC_THRESHOLD:.0%}).",
            "Does the claim still stand on the human-grounded evidence alone?",
            loc, tuple(e.id for e in synth),
        )


def synthetic_share(claim: Claim) -> float:
    """Mean syntheticity of the supporting evidence. 0.0 with no supporting evidence."""
    supporting = claim.supporting
    if not supporting:
        return 0.0
    return round(sum(e.syntheticity for e in supporting) / len(supporting), 6)


def stress_claim(claim: Claim, as_of: Optional[date] = None) -> ClaimResult:
    findings = sorted(
        _rules(claim, as_of),
        key=lambda f: (_SEVERITY_ORDER[f.severity], f.rule_id, f.location or ""),
    )
    return ClaimResult(claim=claim, findings=findings, synthetic_share=synthetic_share(claim))


def analyze(doc: Any, as_of: Optional[date] = None) -> Report:
    """Run the stress profile over every claim in an input document.

    `as_of` is the reference date for S06 when a claim carries no `date`. Without
    either, S06 does not fire, so output never depends on the wall clock.
    """
    from . import __version__

    results = [stress_claim(c, as_of) for c in parse_claims(doc)]
    return Report(tool=TOOL_NAME, version=__version__, results=results)


# --------------------------------------------------------------------------- #
# Verification chain: the engine, not a model, grades each falsification path
# --------------------------------------------------------------------------- #
def check_paths(result: ClaimResult, paths: list[FalsificationPath]) -> list[FalsificationPath]:
    """Grade each path's claimed anchors against the claim's evidence and fired rules.

    anchored: every target is an evidence id in the claim or a rule that fired.
    extends: no targets; a new line of attack, treat as a lead.
    unanchored: at least one target does not exist; treat with caution.
    """
    valid = {e.id for e in result.claim.evidence} | {f.rule_id for f in result.findings}
    out = []
    for p in paths:
        unknown = tuple(t for t in p.targets if t not in valid)
        check = "extends" if not p.targets else ("unanchored" if unknown else "anchored")
        out.append(replace(p, check=check, unknown_targets=unknown))
    return out
