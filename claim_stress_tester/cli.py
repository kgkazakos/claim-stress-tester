"""Click CLI. The `paradigm` command is mandatory on every CPR Phase II tool."""

from __future__ import annotations

import json
import sys
from datetime import date
from typing import Optional

import click

from . import __version__
from .engine import analyze
from .io import load_input, render_probes, render_text, to_export
from .models import AI_PARADIGM, AI_PARADIGM_RATIONALE, TOOL_NAME


def _as_of(value: Optional[str]) -> Optional[date]:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise click.BadParameter("use YYYY-MM-DD", param_hint="--as-of") from exc


def _load(path: str, as_of: Optional[str]):
    try:
        return analyze(load_input(path), as_of=_as_of(as_of))
    except (ValueError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc


def _run(path: str, no_llm: bool, model: Optional[str], verify: bool, paths: int, as_of: Optional[str]):
    report = _load(path, as_of)
    if not no_llm:
        from .llm import enrich

        report = enrich(report, model=model, verify=verify, n_paths=paths)
    return report


def ai_options(f):
    f = click.option("--as-of", default=None, help="Reference date for stale evidence when a claim has no date.")(f)
    f = click.option("--paths", default=4, show_default=True, type=click.IntRange(1, 8),
                     help="Falsification paths to ask the model for in --verify mode.")(f)
    f = click.option("--verify", is_flag=True, help="Verification-chain mode: more paths, each traced to the evidence and rules it targets.")(f)
    f = click.option("--model", default=None, help="litellm model id. Precedence: --model, CPR_LLM_MODEL, anthropic/claude-opus-5-5.")(f)
    f = click.option("--no-llm", is_flag=True, help="Deterministic core only; never calls a model.")(f)
    return f


@click.group()
@click.version_option(__version__, prog_name=TOOL_NAME)
def main() -> None:
    """Try to falsify a research claim against its evidence."""


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@ai_options
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
def run(path, no_llm, model, verify, paths, as_of, fmt) -> None:
    """Falsification paths for each claim, checked against the stress profile."""
    report = _run(path, no_llm, model, verify, paths, as_of)
    if fmt == "json":
        click.echo(json.dumps(to_export(report), indent=2))
    else:
        click.echo(render_text(report))


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("--as-of", default=None, help="Reference date for stale evidence when a claim has no date.")
def probes(path, as_of) -> None:
    """Print only the falsification probes, as a fieldwork checklist. Never calls a model."""
    click.echo(render_probes(_load(path, as_of)))


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", type=click.Path(dir_okay=False), default=None)
@ai_options
def export(path, output, no_llm, model, verify, paths, as_of) -> None:
    """Write the versioned JSON export."""
    payload = json.dumps(to_export(_run(path, no_llm, model, verify, paths, as_of)), indent=2)
    if output:
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")
        click.echo(f"Wrote {output}", err=True)
    else:
        sys.stdout.write(payload + "\n")


@main.command()
def paradigm() -> None:
    """Print the AI paradigm and why it fits this research task."""
    click.echo(f"AI paradigm: {AI_PARADIGM}\n")
    click.echo(AI_PARADIGM_RATIONALE)


if __name__ == "__main__":
    main()
