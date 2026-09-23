"""
End-to-end dry run.

Two source modes:

    --source hf      (default)
        Load documents from HuggingFace `coastalcph/lex_glue` eurlex
        subset. Plain text only, no annex tables. This is the working
        path on day-1 since eur-lex.europa.eu is WAF-blocked.

    --source html
        Use cached HTML in data/raw_html/ (one file per CELEX in
        data/celex_dry_run.txt). Will fail until the WAF/CELLAR issue
        is resolved — kept for future use when annex tables matter.

The script does NOT call any LLM. It produces the input artefacts that
the Phase-1 extraction prompt will consume.

Usage:
    python -m scripts.run_dry_run                    # HF mode, 5 docs
    python -m scripts.run_dry_run --n 10             # HF mode, 10 docs
    python -m scripts.run_dry_run --source html      # HTML mode
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eurlex_dataset import load_short_docs  # noqa: E402
from src.eurlex_fetch import fetch_celex  # noqa: E402
from src.eurlex_parse import parse_document, parse_plain_text  # noqa: E402
from src.table_linearize import linearize_table  # noqa: E402

app = typer.Typer(add_completion=False)
console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _summary_row(doc, linearized_count: int) -> tuple:
    return (
        doc.celex,
        (doc.title or "")[:60],
        str(len(doc.citations)),
        str(len(doc.recitals)),
        str(len(doc.articles)),
        str(len(doc.annexes)),
        str(sum(len(a.tables) for a in doc.annexes)),
        str(linearized_count),
    )


def _write_output(doc, linearized: list[dict], parsed_dir: Path) -> Path:
    out = {
        "document": doc.model_dump(mode="json"),
        "linearized_table_rows": linearized,
    }
    out_path = parsed_dir / f"{doc.celex}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def _process_hf_doc(eurlex_doc, parsed_dir: Path) -> tuple:
    """HF plain-text path."""
    doc = parse_plain_text(eurlex_doc.text, doc_id=eurlex_doc.doc_id)
    # No tables in plain text; linearized is always empty.
    linearized: list[dict] = []
    _write_output(doc, linearized, parsed_dir)
    return _summary_row(doc, 0)


def _process_html_celex(celex: str, raw_dir: Path, parsed_dir: Path) -> tuple:
    """HTML path — currently blocked by WAF, kept for future use."""
    html = fetch_celex(celex, cache_dir=raw_dir)
    doc = parse_document(html, celex=celex)
    linearized: list[dict] = []
    for annex in doc.annexes:
        for tbl in annex.tables:
            for row in linearize_table(tbl):
                linearized.append({
                    "sentence": row.sentence,
                    "locator": row.locator.model_dump(mode="json"),
                })
    _write_output(doc, linearized, parsed_dir)
    return _summary_row(doc, len(linearized))


@app.command()
def main(
    source: str = typer.Option("hf", help="Document source: 'hf' or 'html'"),
    n: int = typer.Option(5, help="(hf mode) number of short documents to load"),
    min_words: int = typer.Option(500, help="(hf mode) minimum doc length in words"),
    max_words: int = typer.Option(2500, help="(hf mode) maximum doc length in words"),
    seed_offset: int = typer.Option(0, help="(hf mode) skip the first K matching docs"),
    celex_file: Path = typer.Option(
        Path("data/celex_dry_run.txt"),
        help="(html mode) plain-text file of CELEX numbers, one per line",
    ),
    raw_dir: Path = typer.Option(Path("data/raw_html")),
    parsed_dir: Path = typer.Option(Path("data/parsed")),
    hf_cache: Path = typer.Option(Path("data/hf_cache"), help="(hf mode) HuggingFace cache directory"),
):
    parsed_dir.mkdir(parents=True, exist_ok=True)

    summary = Table(title="Dry-run summary", show_lines=False)
    for col in ("DocID", "Title", "Cit", "Rec", "Art", "Annex", "Tbl", "Rows"):
        summary.add_column(col, overflow="fold")

    if source == "hf":
        hf_cache.mkdir(parents=True, exist_ok=True)
        console.print(f"[bold]HF mode[/bold] — loading {n} short docs ({min_words}–{max_words} words)")
        docs = load_short_docs(
            n=n, min_words=min_words, max_words=max_words,
            cache_dir=hf_cache, seed_offset=seed_offset,
        )
        for d in docs:
            try:
                row = _process_hf_doc(d, parsed_dir)
                summary.add_row(*row)
            except Exception as e:
                console.print(f"[red]FAIL[/red] {d.doc_id}: {e}")

    elif source == "html":
        raw_dir.mkdir(parents=True, exist_ok=True)
        celex_list = [
            line.strip()
            for line in celex_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        console.print(f"[bold]HTML mode[/bold] — processing {len(celex_list)} CELEX(s)")
        for celex in celex_list:
            try:
                row = _process_html_celex(celex, raw_dir, parsed_dir)
                summary.add_row(*row)
            except Exception as e:
                console.print(f"[red]FAIL[/red] {celex}: {e}")

    else:
        raise typer.BadParameter(f"Unknown --source: {source!r}. Use 'hf' or 'html'.")

    console.print(summary)
    console.print(f"[green]Done.[/green] Parsed JSON in {parsed_dir}")


if __name__ == "__main__":
    app()
