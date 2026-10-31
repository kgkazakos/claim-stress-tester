"""Model layer for claim-stress-tester.

The model's falsification paths are the product. A reasoning model tries to falsify
each claim and returns its strongest lines of attack: the attack, what evidence would
confirm it, and which evidence ids or fired rules it targets. The engine, not a model,
then checks every path against the input and the S01-S10 findings. Any grade the model
returns is ignored.

If a call fails or the reply is unusable, a claim's paths stay None. If no claim gets
paths, the report equals the --no-llm report apart from a visible "model unavailable"
note. This module never reads back into or mutates any value the engine computed.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import re
from typing import Any, Optional

from .engine import RULES, check_paths
from .models import ClaimResult, FalsificationPath, Report

DEFAULT_MODEL = "anthropic/claude-opus-5-5"  # reasoning is the product: default to the strongest reasoner
MODEL_ENV = "CPR_LLM_MODEL"
DEFAULT_PATHS = 3  # without --verify


def resolve_model(model: Optional[str] = None) -> str:
    """Any litellm-routable id. Precedence: --model, then CPR_LLM_MODEL, then the default."""
    return model or os.environ.get(MODEL_ENV) or DEFAULT_MODEL


@contextlib.contextmanager
def _silenced():
    """litellm prints banners to stdout/stderr that corrupt JSON output. Swallow them."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        yield


def _call_model(prompt: str, model: str) -> str:
    import litellm  # lazy: --no-llm paths never import it

    with _silenced():
        resp = litellm.completion(model=model, messages=[{"role": "user", "content": prompt}])
    return resp["choices"][0]["message"]["content"]


def build_prompt(result: ClaimResult, n_paths: int) -> str:
    c = result.claim
    evidence = [
        f"- {e.id} [{e.stance}] {e.summary} (method: {e.method}; population: {e.population or 'unspecified'}; "
        f"n: {e.n if e.n is not None else 'unspecified'}; collected: {e.collected or 'unspecified'})"
        for e in c.evidence
    ] or ["- none supplied"]
    findings = [f"- {f.rule_id} {RULES[f.rule_id]}: {f.message}" for f in result.findings] or ["- none fired"]
    return (
        "You are helping a human researcher stress-test one of their own research claims. "
        "Your job is to try to falsify it. Find the strongest reasons it could be wrong: "
        "populations or contexts where it would fail, confounds, rival explanations, "
        "measurement problems, and tensions with the evidence below.\n\n"
        "Rules:\n"
        "- Write as a single analyst. Do not role-play personas, panels or opposing voices.\n"
        "- Never invent studies, citations, statistics, quotes or data. Each path is a "
        "hypothesis to check, and 'would_confirm' says what evidence the researcher would "
        "need to collect or look up to confirm it.\n"
        "- 'targets' lists only evidence ids and rule ids from the lists below that the "
        "attack bears on. Use an empty list if it bears on none of them.\n"
        "- Do not say whether the claim is true. The researcher decides.\n\n"
        f"Claim {c.id}: {c.statement}\n"
        f"Claimed population: {c.population or 'unspecified'}\n"
        f"Stated falsification condition: {c.would_be_wrong_if or 'none stated'}\n"
        f"Stated limitations: {c.limitations or 'none stated'}\n\n"
        "Evidence:\n" + "\n".join(evidence) + "\n\n"
        "Rule-based stress findings that fired:\n" + "\n".join(findings) + "\n\n"
        f"Return the {n_paths} strongest falsification paths, strongest first, as JSON only:\n"
        '{"paths": [{"attack": "...", "would_confirm": "...", "targets": ["e1", "S02"]}]}'
    )


def parse_paths(text: str, n_paths: int) -> Optional[list[FalsificationPath]]:
    """Parse the model's JSON reply. Returns None if nothing usable came back."""
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    data: Any = json.loads(m.group(0))
    out = []
    for item in (data.get("paths") or [])[:n_paths]:
        if not isinstance(item, dict):
            continue
        attack = str(item.get("attack") or "").strip()
        confirm = str(item.get("would_confirm") or "").strip()
        if not attack:
            continue
        targets = item.get("targets") or []
        if not isinstance(targets, list):
            targets = [targets]
        out.append(FalsificationPath(
            attack=attack,
            would_confirm=confirm,
            targets=tuple(dict.fromkeys(str(t).strip() for t in targets if str(t).strip())),
        ))
    return out or None


def falsify(result: ClaimResult, model: str, n_paths: int) -> Optional[list[FalsificationPath]]:
    try:
        paths = parse_paths(_call_model(build_prompt(result, n_paths), model), n_paths)
    except Exception:  # graceful degradation: the rule-based result stands
        return None
    # The engine checks every path, in every mode. The model never grades itself.
    return check_paths(result, paths) if paths else None


def enrich(report: Report, model: Optional[str] = None, verify: bool = False,
           n_paths: Optional[int] = None) -> Report:
    """Return a copy of the report with engine-checked falsification paths where the model answered.

    Findings, probes and synthetic shares are copied untouched. If every call fails, the
    returned report equals the --no-llm report apart from `model_note`.
    """
    model = resolve_model(model)
    n = n_paths if (verify and n_paths) else DEFAULT_PATHS
    out = copy.deepcopy(report)
    for r in out.results:
        r.paths = falsify(r, model, n)
    if any(r.paths for r in out.results):
        out.model, out.verify = model, verify
    else:
        out.model_note = f"model unavailable ({model}); showing the --no-llm report"
    return out
