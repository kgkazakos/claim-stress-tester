"""Loaders, renderers and the versioned export."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .engine import RULES
from .models import AI_PARADIGM, EXPORT_SCHEMA, ClaimResult, Report


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


def render_text(report: Report) -> str:
    c = report.counts()
    out = [
        f"{report.tool} {report.version}",
        f"{len(report.results)} claim(s) · {len(report.findings)} findings "
        f"(high {c['high']}, medium {c['medium']}, low {c['low']})",
    ]
    for r in report.results:
        out += [""] + _claim_header(r)
        if not r.findings:
            out.append("  No stress findings. That is not evidence the claim is true.")
        for f in r.findings:
            ids = f" ({', '.join(f.evidence_ids)})" if f.evidence_ids else ""
            out.append(f"  [{f.severity.upper():6}] {f.rule_id} {RULES[f.rule_id]}{ids}: {f.message}")
            out.append(f"           probe: {f.probe}")
        if r.paths:
            label = "checked by the engine" if report.verify else "unchecked"
            out += ["", f"  Falsification paths (AI-generated, additive, {label}; model {report.model}):"]
            for i, p in enumerate(r.paths, 1):
                chip = f"[{p.check}] " if p.check else ""
                out.append(f"    {i}. {chip}{p.attack}")
                out.append(f"       would confirm: {p.would_confirm}")
                if p.targets:
                    out.append(f"       targets: {', '.join(p.targets)}")
                if p.unknown_targets:
                    out.append(f"       not found in input: {', '.join(p.unknown_targets)}")
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
