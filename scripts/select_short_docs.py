"""
Pick a fixed-size set of short, well-parsed EUR-Lex docs for Phase 1.

Pipeline:

    HF lex_glue (split=train)
      └── load_short_docs(min_words, max_words)
            └── for each candidate:
                  parse_plain_text(...)  → ParsedDocument
                    └── quality_filter:
                          * non-empty title
                          * >= 1 article
                          * >= 2 recitals OR >= 2 citations
                          * enacting_text length > 200 chars
                  if passes:
                    write data/parsed/<doc_id>.json
                    record into manifest
                  stop when manifest reaches --n entries

Output files:

    data/parsed/<doc_id>.json                  (one per selected doc)
    data/doc_lists/phase1_short.json           (the selection manifest)

The selection manifest is the single source of truth for the batch
extraction script (`scripts/run_extraction.py`) and the future
Cytoscape / Flask UI. Its shape is intentionally UI-friendly:

    {
      "schema_version": 1,
      "created_at": "2026-05-22T...",
      "selection_criteria": { ... },
      "docs": [
        {
          "doc_id": "train-000000",
          "title": "...",
          "n_words": 1234,
          "char_length": 4567,
          "recital_count": 6,
          "article_count": 3,
          "citation_count": 2,
          "parsed_path": "data/parsed/train-000000.json"
        },
        ...
      ]
    }

Usage:

    python -m scripts.select_short_docs                    # default: n=50
    python -m scripts.select_short_docs --n 20             # smaller for smoke
    python -m scripts.select_short_docs --resume           # add to existing
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eurlex_dataset import EurlexDoc, load_short_docs  # noqa: E402
from src.eurlex_parse import parse_plain_text  # noqa: E402
from src.schema import ParsedDocument  # noqa: E402

app = typer.Typer(add_completion=False)
console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------


def passes_quality_filter(doc: ParsedDocument) -> tuple[bool, str]:
    """Return (passes, reason). `reason` is empty on pass, human-readable on fail."""
    if not doc.title or len(doc.title.strip()) < 5:
        return False, "title missing or too short"
    if len(doc.articles) < 1:
        return False, "no articles parsed"
    if len(doc.recitals) < 2 and len(doc.citations) < 2:
        return False, f"thin preamble (recitals={len(doc.recitals)}, citations={len(doc.citations)})"
    if len(doc.enacting_text) < 200:
        return False, f"enacting_text too short ({len(doc.enacting_text)} chars)"
    return True, ""


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def load_existing_manifest(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": 1, "docs": []}
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def doc_entry(doc: ParsedDocument, n_words: int, parsed_path: Path) -> dict:
    return {
        "doc_id": doc.celex,
        "title": doc.title,
        "n_words": n_words,
        "char_length": len(doc.preamble_text) + len(doc.enacting_text),
        "recital_count": len(doc.recitals),
        "article_count": len(doc.articles),
        "citation_count": len(doc.citations),
        # UI-friendly relative path (forward slashes).
        "parsed_path": parsed_path.relative_to(ROOT).as_posix(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@app.command()
def main(
    n: int = typer.Option(50, "--n", help="Target number of docs to select."),
    min_words: int = typer.Option(500, help="HF filter: lower word-count bound."),
    max_words: int = typer.Option(2500, help="HF filter: upper word-count bound."),
    candidate_pool: int = typer.Option(
        300, "--pool",
        help="Pull this many HF candidates and run the parse-quality filter on each.",
    ),
    parsed_dir: Path = typer.Option(ROOT / "data" / "parsed"),
    hf_cache: Path = typer.Option(ROOT / "data" / "hf_cache"),
    manifest_path: Path = typer.Option(ROOT / "data" / "doc_lists" / "phase1_short.json"),
    resume: bool = typer.Option(
        False, "--resume",
        help="Append to an existing manifest instead of starting fresh.",
    ),
):
    """Select N short, well-parsed EUR-Lex docs for Phase 1."""
    parsed_dir.mkdir(parents=True, exist_ok=True)
    hf_cache.mkdir(parents=True, exist_ok=True)

    manifest = load_existing_manifest(manifest_path) if resume else {
        "schema_version": 1,
        "docs": [],
    }
    already_have = {d["doc_id"] for d in manifest["docs"]}
    target_total = n
    need = max(0, target_total - len(already_have))
    if need == 0:
        console.print(f"[green]Manifest already has {len(already_have)} docs (>= {target_total}). Nothing to do.[/green]")
        raise typer.Exit()

    console.print(
        f"[bold]Selecting[/bold] {need} new docs "
        f"(target total {target_total}, already have {len(already_have)}). "
        f"Word range {min_words}–{max_words}. Candidate pool {candidate_pool}."
    )

    candidates: list[EurlexDoc] = load_short_docs(
        n=candidate_pool,
        min_words=min_words,
        max_words=max_words,
        cache_dir=hf_cache,
    )
    console.print(f"Pulled {len(candidates)} HF candidates. Parsing & filtering...")

    summary = Table(title="Selected docs", show_lines=False)
    for col in ("DocID", "Words", "Cit", "Rec", "Art", "Title"):
        summary.add_column(col, overflow="fold")

    rejected: list[tuple[str, str]] = []
    new_entries: list[dict] = []

    for eurlex_doc in candidates:
        if eurlex_doc.doc_id in already_have:
            continue
        try:
            parsed: ParsedDocument = parse_plain_text(eurlex_doc.text, doc_id=eurlex_doc.doc_id)
        except Exception as e:
            rejected.append((eurlex_doc.doc_id, f"parser raised: {e!r}"))
            continue

        ok, reason = passes_quality_filter(parsed)
        if not ok:
            rejected.append((eurlex_doc.doc_id, reason))
            continue

        out_path = parsed_dir / f"{parsed.celex}.json"
        out_path.write_text(
            json.dumps(
                {
                    "document": parsed.model_dump(mode="json"),
                    "linearized_table_rows": [],
                },
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        entry = doc_entry(parsed, eurlex_doc.n_words, out_path)
        new_entries.append(entry)
        summary.add_row(
            entry["doc_id"],
            str(entry["n_words"]),
            str(entry["citation_count"]),
            str(entry["recital_count"]),
            str(entry["article_count"]),
            (entry["title"] or "")[:60],
        )

        if len(new_entries) >= need:
            break

    manifest["docs"].extend(new_entries)
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    manifest["selection_criteria"] = {
        "min_words": min_words,
        "max_words": max_words,
        "candidate_pool": candidate_pool,
        "filter": "title>=5chars AND articles>=1 AND (recitals>=2 OR citations>=2) AND enacting>=200chars",
    }
    write_manifest(manifest_path, manifest)

    console.print(summary)
    console.print(
        f"[green]Selected[/green] {len(new_entries)} new docs "
        f"({len(manifest['docs'])} total in manifest).")
    if rejected:
        console.print(
            f"[yellow]Rejected[/yellow] {len(rejected)} candidates. "
            f"First 5 reasons:"
        )
        for did, reason in rejected[:5]:
            console.print(f"  {did}: {reason}")
    console.print(f"Manifest: [cyan]{manifest_path}[/cyan]")
    console.print(f"Parsed JSONs: [cyan]{parsed_dir}[/cyan]")


if __name__ == "__main__":
    app()
