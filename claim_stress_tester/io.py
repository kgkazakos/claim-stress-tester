"""Loaders, renderers and the versioned export."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .engine import RULES
from .models import AI_PARADIGM, CHECKS, EXPORT_SCHEMA, ClaimResult, Report


def load_input(path: str | Path) -> Any:
    """Load claims from JSON or YAML: one claim, {"claims": [...]}, or a list of claims."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, (dict, list)):
        raise ValueError("Expected a claim object, {\"claims\": [...]}, or a list of claims.")
    return data


def to_export(report: Report) -> dict[str, Any]:
    return {
        "schema": EXPORT_SCHEMA,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "paradigm": AI_PARADIGM,
        # No _cpr_synthetic block: falsification paths are labelled critique text, not
        # synthetic research data. Input ancestry is read and reported as synthetic_share.
        "report": report.to_dict(),
    }


def _claim_header(r: ClaimResult) -> list[str]:
    c = r.claim
    return [
        f"[{c.id}] {c.statement}",
        f"  population: {c.population or 'unspecified'} · evidence: {len(c.supporting)} supporting, "
        f"{len(c.against)} against · synthetic share {r.synthetic_share:.0%}",
    ]


def describe_target(r: ClaimResult, target: str) -> str:
    """Name a path target the way the engine resolved it: an evidence item or a fired rule."""
    for e in r.claim.evidence:
        if e.id == target:
            return f"{e.id} ({e.stance}, {e.method}): {e.summary}"
    if any(f.rule_id == target for f in r.findings):
        return f"{target} {RULES[target]}"
    return f"{target} (not found in input)"


def check_tally(r: ClaimResult) -> str:
    paths = r.paths or []
    return ", ".join(f"{sum(1 for p in paths if p.check == c)} {c}" for c in CHECKS)


def headline(report: Report) -> str:
    """One-line status of the model layer, shown above the results in every surface."""
    if report.model:
        n = sum(len(r.paths or []) for r in report.results)
        mode = "verification-chain mode" if report.verify else "default mode"
        return f"{n} falsification path(s) from {report.model} ({mode}), each checked by the engine"
    if report.model_note:
        return report.model_note
    return "Model off (--no-llm): deterministic stress profile only"


def _paths_block(report: Report, r: ClaimResult) -> list[str]:
    if not r.paths:
        if report.model:
            return ["  No falsification paths came back for this claim."]
        return []
    out = ["  Falsification paths (AI-generated; the engine checked each one):"]
    for i, p in enumerate(r.paths, 1):
        out.append(f"    {i}. [{p.check}] {p.attack}")
        out.append(f"       would confirm: {p.would_confirm}")
        if report.verify:
            for t in p.targets:
                out.append(f"       chain: {describe_target(r, t)}")
            if not p.targets:
                out.append("       chain: no targets; a new line of attack, treat it as a lead")
        else:
            if p.targets:
                out.append(f"       targets: {', '.join(p.targets)}")
            if p.unknown_targets:
                out.append(f"       not found in input: {', '.join(p.unknown_targets)}")
    out.append(f"    checks: {check_tally(r)}")
    return out


def _findings_block(report: Report, r: ClaimResult) -> list[str]:
    title = ("Verified against: stress profile (rules S01-S10, no model)" if report.model
             else "Stress profile (rules S01-S10, no model)")
    out = [f"  {title}"]
    if not r.findings:
        out.append("    No stress findings. That is not evidence the claim is true.")
    for f in r.findings:
        ids = f" ({', '.join(f.evidence_ids)})" if f.evidence_ids else ""
        out.append(f"    [{f.severity.upper():6}] {f.rule_id} {RULES[f.rule_id]}{ids}: {f.message}")
        out.append(f"             probe: {f.probe}")
    return out


def render_text(report: Report) -> str:
    """Model-first: the falsification paths lead, the stress profile that checked them follows."""
    c = report.counts()
    out = [
        f"{report.tool} {report.version}",
        headline(report),
        f"{len(report.results)} claim(s) · {len(report.findings)} findings "
        f"(high {c['high']}, medium {c['medium']}, low {c['low']})",
    ]
    for r in report.results:
        out += [""] + _claim_header(r)
        block = _paths_block(report, r)
        if block:
            out += [""] + block
        out += [""] + _findings_block(report, r)
    return "\n".join(out)


def render_probes(report: Report) -> str:
    """The falsification probes only, as a checklist the researcher can take into fieldwork."""
    out: list[str] = []
    for r in report.results:
        if out:
            out.append("")
        out.append(f"[{r.claim.id}] {r.claim.statement}")
        if not r.findings:
            out.append("  (no probes: no stress findings)")
        for f in r.findings:
            out.append(f"  [ ] {f.probe}  ({f.rule_id})")
    return "\n".join(out)
