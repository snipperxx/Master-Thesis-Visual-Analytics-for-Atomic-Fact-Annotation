"""
Batch driver for Phase-1 extraction: run M models × D docs once.

Why this exists alongside `src.extractor`:
  `src.extractor` is the single-(model, doc) workhorse. This script is
  the orchestrator that:
    * loops models on the outside and docs on the inside (minimises
      VRAM swaps on 6 GB hardware — switching models is the expensive
      operation, not switching docs);
    * is **resumable** via a manifest at `data/facts/manifest.json`
      keyed by "<model>|<doc_id>". Re-running picks up where it left
      off; only failed/missing pairs get retried.
    * writes results in a shape the future Cytoscape / Flask UI can
      read directly (status, fact_count, duration, relative paths).

Subcommands:

    run     — execute pending (model, doc) pairs
    status  — print the manifest as a coverage table
    list    — list pending and completed pairs (machine-friendly)

Usage:

    # Default: 3 models × the selection from select_short_docs.py
    python -m scripts.run_extraction run

    # Smoke test first:
    python -m scripts.run_extraction run \\
        --models qwen3.5:4b --max-docs 2

    # Just two of the three models:
    python -m scripts.run_extraction run \\
        --models qwen3.5:4b,gemma3:4b

    # Force re-run a model that finished:
    python -m scripts.run_extraction run --models gemma3:4b --force
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.extractor import (  # noqa: E402
    DEFAULT_BASE_URL,
    extract,
    load_parsed_doc,
    unload,
    write_facts,
    _unload_others,
)
from src.segment_population import failed_section_records  # noqa: E402

app = typer.Typer(add_completion=False)
console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_DOC_LIST = ROOT / "data" / "doc_lists" / "phase1_short.json"
DEFAULT_MANIFEST = ROOT / "data" / "facts" / "manifest.json"
DEFAULT_FACTS_ROOT = ROOT / "data" / "facts"
DEFAULT_MODELS = ("qwen3.5:4b", "gemma3:4b", "nemotron-3-nano:4b")

# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _run_key(model: str, doc_id: str) -> str:
    return f"{model}|{doc_id}"


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {
            "schema_version": 1,
            "guideline_version": "v1",
            "runs": {},  # keyed by "<model>|<doc_id>"
        }
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def load_doc_list(doc_list_path: Path) -> list[dict]:
    """Read the selection manifest produced by select_short_docs.py."""
    if not doc_list_path.exists():
        console.print(
            f"[red]Doc list not found:[/red] {doc_list_path}\n"
            "Run [bold]python -m scripts.select_short_docs[/bold] first."
        )
        raise typer.Exit(code=1)
    payload = json.loads(doc_list_path.read_text(encoding="utf-8"))
    return payload.get("docs", [])


# ---------------------------------------------------------------------------
# Core: run one (model, doc) pair and record the outcome
# ---------------------------------------------------------------------------


def _run_one(model: str,
             doc_entry: dict,
             *,
             out_root: Path,
             base_url: str,
             guideline_version: str,
             retries: int) -> dict:
    """Run a single (model, doc). Returns a manifest-shaped run record."""
    doc_path = ROOT / doc_entry["parsed_path"]
    doc_id = doc_entry["doc_id"]
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()

    record = {
        "model": model,
        "doc_id": doc_id,
        "doc_path": doc_entry["parsed_path"],
        "output_path": None,
        "status": "failed",
        "fact_count": 0,
        "duration_s": 0.0,
        "guideline_version": guideline_version,
        "started_at": started_at,
        "finished_at": None,
        "error": None,
    }

    incomplete: list[str] = []
    try:
        parsed_doc = load_parsed_doc(doc_path)
        facts = extract(
            model_name=model,
            parsed_doc=parsed_doc,
            guideline_version=guideline_version,
            base_url=base_url,
            retries=retries,
            # We manage solo-residency at the model-loop level (see run()),
            # so don't re-evict on every doc.
            ensure_solo=False,
            unload_after=False,
            incomplete_out=incomplete,
        )
        # WHICH sections returned nothing, not only how many: RQ1's segment
        # population (method.tex sec:missing-segments) excludes exactly
        # these, and a section with zero facts is otherwise a legitimate
        # empty answer. Recorded in the facts file and the manifest record
        # alike (src/segment_population.py reads both and refuses if they
        # disagree). Additive: no fact record changes.
        failed_sections = failed_section_records(incomplete, strict=False)
        out_path = write_facts(
            facts,
            model_name=model,
            doc_id=doc_id,
            guideline_version=guideline_version,
            out_root=out_root,
            failed_sections=failed_sections,
        )
        # A document that lost sections to the proxy under-reports its fact
        # count, and fact count is the thesis's measurement — so it is NOT a
        # clean result even though some sections succeeded. Record it as
        # retryable rather than `ok`; 9 qwen documents were silently
        # incomplete this way before this guard existed (BUG-2026-09-04).
        record.update({
            "status": "failed" if incomplete else "ok",
            "incomplete_sections": len(incomplete),
            "failed_sections": failed_sections,
            "fact_count": len(facts),
            # .resolve() before relative_to: out_path is relative whenever
            # --out is passed as a relative CLI string (e.g. "data/facts_x"),
            # and Path.relative_to() refuses to compare a relative path
            # against ROOT (absolute) even when they denote the same
            # location — it errored on every write on Windows until this,
            # so `status: ok` never got recorded and every run "failed".
            "output_path": out_path.resolve().relative_to(ROOT).as_posix(),
        })
        if incomplete:
            record["error"] = (
                f"INCOMPLETE: {len(incomplete)} section(s) skipped on persistent "
                f"network failures ({', '.join(incomplete)}); {len(facts)} facts "
                f"were written but the count under-reports — re-run this pair"
            )
            console.print(
                f"[yellow]INCOMPLETE[/yellow] {model} × {doc_id}: "
                f"{len(incomplete)} section(s) skipped, marked for re-run"
            )
    except KeyboardInterrupt:
        record["error"] = "interrupted"
        raise
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
        console.print(f"[red]FAIL[/red] {model} × {doc_id}: {record['error']}")
    finally:
        record["duration_s"] = round(time.monotonic() - t0, 2)
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
    return record


# ---------------------------------------------------------------------------
# `run` subcommand
# ---------------------------------------------------------------------------


@app.command()
def run(
    models: str = typer.Option(
        ",".join(DEFAULT_MODELS), "--models",
        help="Comma-separated Ollama model tags."),
    doc_list: Path = typer.Option(DEFAULT_DOC_LIST, "--doc-list"),
    out_root: Path = typer.Option(DEFAULT_FACTS_ROOT, "--out"),
    manifest_path: Path = typer.Option(DEFAULT_MANIFEST, "--manifest"),
    guideline_version: str = typer.Option("v1", "--guideline"),
    retries: int = typer.Option(2, "--retries"),
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url"),
    max_docs: Optional[int] = typer.Option(
        None, "--max-docs",
        help="Cap docs per model (smoke-test cap)."),
    force: bool = typer.Option(
        False, "--force",
        help="Re-run pairs that have status=ok in the manifest."),
):
    """Execute pending (model, doc) pairs. Resumable."""
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    docs = load_doc_list(doc_list)
    if max_docs is not None:
        docs = docs[:max_docs]

    manifest = load_manifest(manifest_path)
    manifest["guideline_version"] = guideline_version

    total_pairs = len(model_list) * len(docs)
    skipped = sum(
        1 for m in model_list for d in docs
        if not force and manifest["runs"].get(_run_key(m, d["doc_id"]), {}).get("status") == "ok"
    )
    pending = total_pairs - skipped
    console.print(
        f"[bold]Plan:[/bold] {len(model_list)} model(s) × {len(docs)} doc(s) "
        f"= {total_pairs} pairs; {pending} pending, {skipped} skipped (already ok)."
    )
    if pending == 0:
        console.print("[green]Nothing to do.[/green]")
        return

    try:
        for model in model_list:
            # Model-outer loop: evict everything else once, run all docs.
            console.print(f"\n[bold cyan]>>> Loading model:[/bold cyan] {model}")
            try:
                _unload_others(keep=model, base_url=base_url)
            except Exception as e:
                console.print(f"[yellow]Warning while evicting others: {e}[/yellow]")

            for doc_entry in docs:
                doc_id = doc_entry["doc_id"]
                key = _run_key(model, doc_id)
                prior = manifest["runs"].get(key)
                if prior and prior.get("status") == "ok" and not force:
                    console.print(f"  [dim]skip[/dim] {doc_id} (already ok, {prior['fact_count']} facts)")
                    continue

                console.print(f"  [bold]run[/bold]  {doc_id}")
                record = _run_one(
                    model, doc_entry,
                    out_root=out_root,
                    base_url=base_url,
                    guideline_version=guideline_version,
                    retries=retries,
                )
                manifest["runs"][key] = record
                save_manifest(manifest_path, manifest)
                tag = "[green]ok[/green]" if record["status"] == "ok" else "[red]fail[/red]"
                console.print(
                    f"       {tag} — {record['fact_count']} facts in {record['duration_s']}s"
                )

            # Done with this model — free VRAM for the next one.
            if model != model_list[-1]:
                console.print(f"[dim]Unloading {model} to free VRAM for next model.[/dim]")
                try:
                    unload(model, base_url=base_url)
                except Exception as e:
                    console.print(f"[yellow]Warning while unloading: {e}[/yellow]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Manifest saved.[/yellow]")
        save_manifest(manifest_path, manifest)
        raise typer.Exit(code=130)

    save_manifest(manifest_path, manifest)
    _print_status(manifest, model_list, docs)


# ---------------------------------------------------------------------------
# `status` / `list` subcommands
# ---------------------------------------------------------------------------


def _print_status(manifest: dict, models: list[str], docs: list[dict]) -> None:
    summary = Table(title="Extraction matrix", show_lines=False)
    summary.add_column("doc_id", overflow="fold")
    for m in models:
        summary.add_column(m, overflow="fold")

    ok_per_model = {m: 0 for m in models}
    fact_per_model = {m: 0 for m in models}
    for d in docs:
        row = [d["doc_id"]]
        for m in models:
            r = manifest["runs"].get(_run_key(m, d["doc_id"]))
            if r is None:
                row.append("[dim]–[/dim]")
            elif r["status"] == "ok":
                row.append(f"[green]{r['fact_count']}[/green]")
                ok_per_model[m] += 1
                fact_per_model[m] += r["fact_count"]
            else:
                err = (r.get("error") or "?")[:40]
                row.append(f"[red]× {err}[/red]")
        summary.add_row(*row)

    console.print(summary)
    totals = Table(title="Totals", show_lines=False)
    for col in ("Model", "Docs ok", "Total facts"):
        totals.add_column(col)
    for m in models:
        totals.add_row(m, f"{ok_per_model[m]}/{len(docs)}", str(fact_per_model[m]))
    console.print(totals)


@app.command()
def status(
    doc_list: Path = typer.Option(DEFAULT_DOC_LIST),
    manifest_path: Path = typer.Option(DEFAULT_MANIFEST),
    models: str = typer.Option(",".join(DEFAULT_MODELS)),
):
    """Print the manifest as a coverage table."""
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    docs = load_doc_list(doc_list)
    manifest = load_manifest(manifest_path)
    _print_status(manifest, model_list, docs)


@app.command("list")
def list_runs(
    manifest_path: Path = typer.Option(DEFAULT_MANIFEST),
    status_filter: Optional[str] = typer.Option(
        None, "--status",
        help="Filter to one of {ok, failed}."),
):
    """Machine-friendly dump of run records (one JSON per line)."""
    manifest = load_manifest(manifest_path)
    for key, record in manifest["runs"].items():
        if status_filter and record.get("status") != status_filter:
            continue
        typer.echo(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    app()
