"""Streamlit UI. Run: streamlit run claim_stress_tester/app.py"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import streamlit as st

# Streamlit runs this file as a script, so relative imports break. Use the repo root + absolute import.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from claim_stress_tester import __version__  # noqa: E402
from claim_stress_tester.engine import RULES, analyze  # noqa: E402
from claim_stress_tester.models import AI_PARADIGM, AI_PARADIGM_RATIONALE, SYNTHETIC_THRESHOLD, TOOL_NAME  # noqa: E402

ACCENT = "#4f3d8f"  # keep in sync with .streamlit/config.toml
CHIP = {"high": "#9b2c2c", "medium": "#b7791f", "low": "#4a5568"}
CHECK_CHIP = {"anchored": "#2f6b4f", "extends": ACCENT, "unanchored": "#7a4b2a"}
EXAMPLES = sorted((_ROOT / "examples").glob("*.json"))

st.set_page_config(page_title=TOOL_NAME, layout="wide")
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=DM+Mono&display=swap');
html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}
code, .mono {{ font-family: 'DM Mono', monospace; }}
.stApp {{ background: #fbfaf8; }}
section[data-testid="stSidebar"] {{ background: #16161a; }}
section[data-testid="stSidebar"] * {{ color: #ece9e2 !important; }}
div[data-testid="column"]:first-child > div {{ background: #f2ede1; border-radius: 10px; padding: 8px 12px; }}
.card {{ background: #fff; border: 1px solid #e8e2d6; border-radius: 10px; padding: 12px 16px; margin-bottom: 8px; }}
.claim {{ border-left: 4px solid {ACCENT}; }}
.chip {{ display: inline-block; padding: 1px 8px; border-radius: 999px; color: #fff; font-size: 12px; margin-right: 8px; }}
.probe {{ color: #5a554c; font-style: italic; margin-top: 4px; }}
.muted {{ color: #7a746a; font-size: 13px; }}
.accent {{ color: {ACCENT}; }}
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(f"### {TOOL_NAME}")
    st.caption(f"v{__version__} · {AI_PARADIGM}")
    with st.expander("Why this paradigm"):
        st.write(AI_PARADIGM_RATIONALE)
    with st.expander("Stress rules"):
        for rid, name in RULES.items():
            st.markdown(f"`{rid}` {name}")
    st.caption("The tool tries to break a claim. It never says whether the claim is true.")

controls, main = st.columns([1, 2.4])
with controls:
    st.markdown("#### Input")
    names = [p.name for p in EXAMPLES]
    choice = st.selectbox("Example", names) if names else None
    default = (_ROOT / "examples" / choice).read_text(encoding="utf-8") if choice else '{"statement": ""}'
    raw = st.text_area("Claims (JSON)", value=default, height=320, key=f"raw-{choice}")
    as_of = st.date_input("Reference date (if a claim has no date)", value=None)
    st.markdown("#### AI layer")
    use_ai = st.toggle("Falsification paths", value=False)
    verify = st.toggle("Verification-chain mode", value=True, disabled=not use_ai)
    n_paths = st.slider("Paths", 1, 8, 4, disabled=not (use_ai and verify))
    model = st.text_input("Model (litellm id)", value="", placeholder="CPR_LLM_MODEL or gpt-4o-mini",
                          disabled=not use_ai)


def chip(label: str, color: str) -> str:
    return f"<span class='chip' style='background:{color}'>{html.escape(label)}</span>"


with main:
    st.markdown("<h3 class='accent'>Stress profile</h3>", unsafe_allow_html=True)
    try:
        report = analyze(json.loads(raw), as_of=as_of)
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON: {exc}")
        st.stop()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if use_ai:
        from claim_stress_tester.llm import enrich

        with st.spinner("Trying to falsify each claim..."):
            report = enrich(report, model=model or None, verify=verify, n_paths=n_paths)
        if not report.model:
            st.warning("No falsification paths came back (no backend, or the call failed). "
                       "The rule-based stress profile below is unchanged.")

    c = report.counts()
    st.caption(f"{len(report.results)} claim(s) · {len(report.findings)} findings "
               f"(high {c['high']}, medium {c['medium']}, low {c['low']})")

    for r in report.results:
        cl = r.claim
        share_color = CHIP["high"] if r.synthetic_share > SYNTHETIC_THRESHOLD else CHIP["low"]
        st.markdown(
            f"<div class='card claim'><span class='mono'>{html.escape(cl.id)}</span> "
            f"{chip(f'synthetic {r.synthetic_share:.0%}', share_color)}<br/>"
            f"<b>{html.escape(cl.statement)}</b><br/>"
            f"<span class='muted'>population: {html.escape(cl.population or 'unspecified')} · "
            f"{len(cl.supporting)} supporting, {len(cl.against)} against · "
            f"falsification condition: {html.escape(cl.would_be_wrong_if or 'none stated')}</span></div>",
            unsafe_allow_html=True,
        )
        if not r.findings:
            st.info("No stress findings. That is not evidence the claim is true.")
        for f in r.findings:
            ids = f" <span class='mono muted'>{html.escape(', '.join(f.evidence_ids))}</span>" if f.evidence_ids else ""
            st.markdown(
                f"<div class='card'>{chip(f.severity, CHIP[f.severity])}"
                f"<span class='mono'>{f.rule_id}</span> {html.escape(RULES[f.rule_id])}{ids}<br/>"
                f"{html.escape(f.message)}<div class='probe'>Probe: {html.escape(f.probe)}</div></div>",
                unsafe_allow_html=True,
            )
        if r.paths:
            mode = "checked by the engine" if report.verify else "unchecked"
            st.markdown(f"**Falsification paths** <span class='muted'>AI-generated, additive, {mode} · "
                        f"{html.escape(report.model or '')}</span>", unsafe_allow_html=True)
            for i, p in enumerate(r.paths, 1):
                check = chip(p.check, CHECK_CHIP[p.check]) if p.check else ""
                targets = (f"<br/><span class='muted'>targets: <span class='mono'>"
                           f"{html.escape(', '.join(p.targets))}</span></span>") if p.targets else ""
                unknown = (f"<br/><span class='muted'>not found in input: <span class='mono'>"
                           f"{html.escape(', '.join(p.unknown_targets))}</span></span>") if p.unknown_targets else ""
                st.markdown(
                    f"<div class='card'><span class='mono'>{i}</span> {check}{html.escape(p.attack)}"
                    f"<div class='probe'>Would confirm: {html.escape(p.would_confirm)}</div>{targets}{unknown}</div>",
                    unsafe_allow_html=True,
                )
        st.markdown("&nbsp;", unsafe_allow_html=True)
