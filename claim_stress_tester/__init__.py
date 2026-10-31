"""claim-stress-tester: CPR Phase II, Frame A (Test-Time Deliberation), tool 2.

Tries to falsify a research claim. The deterministic core (--no-llm) builds a stress
profile of the claim against its evidence, with a falsification probe per weak point.
The AI layer adds the strongest counter-arguments as falsification paths, which the
engine checks against the supplied evidence in verification-chain mode.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .engine import analyze
from .models import AI_PARADIGM, AI_PARADIGM_RATIONALE, EXPORT_SCHEMA, Finding, Report

__all__ = [
    "__version__",
    "AI_PARADIGM",
    "AI_PARADIGM_RATIONALE",
    "EXPORT_SCHEMA",
    "Finding",
    "Report",
    "analyze",
]
