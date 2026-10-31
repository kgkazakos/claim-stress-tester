# claim-stress-tester

Test-time deliberation that tries to break a research claim before anyone else does.

CPR Phase II, Frame A (Test-Time Deliberation), tool 2. It reads a finished research claim and the evidence behind it, and surfaces the weak points and the strongest lines of attack for a researcher to weigh. It never says whether a claim is true, and it never rewrites one. The human is the epistemic ground.

## What it does

claim-stress-tester reads one or more claims, each with its supporting and disconfirming evidence, and produces two things:

1. A stress profile from a deterministic engine. Ten rules check the claim against its own evidence (overreach, causal language, scope, thinness, staleness, synthetic share and more). Each finding carries a probe: a falsification question you can take back into the data or the field. This runs with no model.
2. Falsification paths from a reasoning model. Each path is a line of attack on the claim, what evidence would confirm it, and which evidence items or fired rules it targets. Paths are hypotheses to check, not evidence.

Start with the probes. They are the cheapest way to find out whether a claim survives, and they do not depend on a model. An empty stress profile is not support for a claim: it only means the claim cleared these structural checks.

## Verification-chain mode

A model asked to falsify a claim can invent problems as confidently as it invents support. Verification-chain mode (`--verify`) asks for several paths, then the engine, not the model, checks each path's targets against the input:

- `anchored`: every target is an evidence id in the claim or a stress rule that fired.
- `extends`: no targets. A new line of attack the engine has nothing to check against. Treat it as a lead.
- `unanchored`: at least one target does not exist in the input. Treat with caution; the ids it invented are listed.

The model does not grade its own paths, and nothing it returns can change a finding. Without `--verify` the tool asks for the three strongest paths and leaves them unchecked.

## AI paradigm

LLM (reasoning / test-time deliberation).

Falsifying a research claim is adversarial, open-ended reasoning, not a lookup: it means imagining the populations, contexts, confounds and rival explanations under which the claim would fail, and saying what evidence would show it. A reasoning model spending deliberate test-time compute is the right fit for that search. A deterministic stress profile runs first and always, so the model attacks a claim whose structural weak points are already named, and verification-chain mode has the engine check every falsification path against the evidence the researcher actually supplied.

Run `cst paradigm` to print this at the tool level.

## No synthetic data, no Provenance Hash

claim-stress-tester produces no synthetic data. It reads claims and evidence and emits findings, probes and critique text. There is no `_cpr_synthetic` block in its export, by design: falsification paths are labelled as AI-generated, kept in their own field with the model id, and are not data about people.

It does consume evidence that may be synthetic. Each evidence item's `_cpr_synthetic` ancestry is read recursively, using the same grounding factors as `provenance-inspector`, and the claim's synthetic share is reported. Above 30% the S10 rule fires: the claim leans on synthetic evidence past the Ouroboros threshold.

## Install

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev,ui]" --no-build-isolation
```

The reasoning layer uses `litellm`, a core dependency. Set a backend with `CPR_LLM_MODEL` (default `gpt-4o-mini`; a reasoning model is recommended) and the usual provider key. With no backend, or if a call fails, the tool degrades gracefully to the deterministic stress profile.

## CLI

```bash
cst run examples/onboarding_claim.json                      # stress profile + falsification paths
cst run examples/onboarding_claim.json --no-llm             # deterministic core only
cst run examples/onboarding_claim.json --verify --paths 5   # verification-chain mode
cst run examples/claims_batch.json --as-of 2026-10-31       # reference date for undated claims
cst probes examples/pricing_claim.json                      # probes only, as a fieldwork checklist
cst export examples/onboarding_claim.json -o out.json       # versioned JSON (cpr.cst.export/1)
cst paradigm                                                # paradigm + rationale
```

Every command that can call a model accepts `--no-llm`, `--verify`, `--paths` and `--model`. `probes` never calls a model.

## Streamlit

```bash
streamlit run claim_stress_tester/app.py
```

Frame A visual language: deep violet accent `#4f3d8f`, DM Sans and DM Mono, near-black sidebar, warm-beige controls, cream main, card rows with severity chips. Each claim renders as a card with its synthetic share, followed by its stress findings with probes, then its falsification paths with verification chips, labelled as additive AI output.

## The stress rules

The engine is a set of textual and structural checks over a claim and its evidence, honest about being one. It is fast, reproducible and cheap, and it is not an oracle: it cannot tell whether a claim is right, only where it is exposed. Output depends only on the input. Stale evidence is measured against the claim's `date`, or `--as-of` when a claim has none, never the wall clock.

| Rule | Fires when | Severity |
|------|-----------|----------|
| S01 | The claim uses a universal ("all", "always", "never", "every", "no one") | medium |
| S02 | Causal language ("causes", "drives", "because", "reduces") with no experimental evidence | high |
| S03 | The claim's population is not among the supporting evidence populations | high |
| S04 | Fewer than 2 supporting items (high), or total n below 12 (medium) | high / medium |
| S05 | Two or more supporting items, all from one method | medium |
| S06 | Supporting evidence collected more than 24 months before the reference date | medium |
| S07 | No falsification condition (`would_be_wrong_if`) | medium |
| S08 | Disconfirming evidence whose id the limitations never mention | high |
| S09 | Intensity words ("significantly", "most", "clearly") with no statistic in the claim | low |
| S10 | Synthetic share of supporting evidence above 30% | high |

## Input format

A JSON or YAML file holding one claim, `{"claims": [...]}`, or a list of claims. `statement` is required; everything else is optional, and missing ids default to `c1`, `e1` and so on. Evidence `stance` is `supports` (the default) or `against`.

```json
{
  "id": "pricing-seat-cap",
  "statement": "Small teams churn at renewal when they hit the five-seat cap.",
  "population": "small teams",
  "date": "2026-10-01",
  "would_be_wrong_if": "Churned teams rarely reached five seats.",
  "limitations": "Exit survey response rate was low. e3 points the other way.",
  "evidence": [
    {"id": "e1", "summary": "Exit survey theme: too expensive to add people.",
     "method": "survey", "population": "small teams", "n": 61, "collected": "2023-11"},
    {"id": "e3", "summary": "Most churned teams had three seats or fewer.",
     "method": "billing analytics", "population": "small teams", "n": 412,
     "collected": "2026-08", "stance": "against"}
  ]
}
```

An evidence item may carry a `_cpr_synthetic` block from an upstream CPR tool; its ancestry feeds S10.

See `examples/` for:

- `onboarding_claim.json`: an overreaching causal claim about a wider population than the evidence covers (S01, S02, S03, S07, S09).
- `pricing_claim.json`: a single-method claim with stale evidence and disconfirming billing data left out of the limitations (S05, S06, S07, S08).
- `synthetic_grounded.json`: a claim resting mostly on synthetic interviews and a persona panel (S04, S10).
- `sound_claim.json`: a scoped, quantified claim with a falsification condition, which fires nothing.
- `claims_batch.json`: three claims in one file, including an undated one for `--as-of` (S01, S02, S04, S06, S07).

## Build

- Rule-based deterministic core, additive reasoning layer, graceful degradation.
- `paradigm` command and `--no-llm` flag, per the Phase II conventions.
- pytest suite, CI on Python 3.10, 3.11, 3.12.
- Versioned export `cpr.cst.export/1`.

## IP

Clean. A single-model critique tool: no synthetic data produced, no agents, no orchestration, no automatic linkage to external systems. The engine checks the model's output; no model judges another. Not on any constraint list.

## License

MIT.
