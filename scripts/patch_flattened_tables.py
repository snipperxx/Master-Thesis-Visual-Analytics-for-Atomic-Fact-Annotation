"""Rewrite flattened OJ tables in place in data/parsed/*.json.

Why a targeted migration instead of re-running the parser
---------------------------------------------------------
`parse_plain_text` is the right place for the table fix, and it has it. But
re-parsing the frozen corpus from the HuggingFace source would ALSO pick up
the "Whereas" recital fallback that landed in the parser on 2026-06-16 and was
never applied to the data. Measured 2026-08-06, that would move the frozen
corpus by:

    eval  90 -> 116 segments (+26)   |   dev  305 -> 433 segments (+128)
    21 of 50 documents change size_bucket — the split's stratification key

That is a re-freeze, not a bug fix, and it must be an explicit decision (it
orphans the gold sample and voids the stratification). So this script changes
*only* what the table fix requires: `enacting_text` for the documents that
carry a `1.2.3.4 // ...` run, plus the article offsets that follow from it.

It refuses to write if the segment inventory would change.

Usage
-----
    python -m scripts.patch_flattened_tables            # dry run, prints a report
    python -m scripts.patch_flattened_tables --write    # apply
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eurlex_parse import slice_articles
from src.gold_store import remap_sample_offsets
from src.oj_table import normalize_flattened_tables
from src.schema import FlattenedTableRecord

PARSED_DIR = Path("data/parsed")
SAMPLE_PATH = Path("data/gold/segment_sample.json")


def patch_document(doc: dict) -> tuple[dict, list[FlattenedTableRecord], list[str]]:
    """Return (patched doc, table records, problems). `doc` is not mutated."""
    out = dict(doc)
    problems: list[str] = []

    before = out.get("enacting_text") or ""
    after, tables = normalize_flattened_tables(before)
    if not tables:
        return out, [], problems

    old_articles = out.get("articles") or []
    new_articles = slice_articles(after)

    # The frozen segment inventory is keyed on article numbers. A table lives
    # inside an article body and cannot introduce or remove an "Article N"
    # heading, so any change here means the rewrite corrupted the text.
    old_nums = [a["number"] for a in old_articles]
    new_nums = [a.number for a in new_articles]
    if old_nums != new_nums:
        problems.append(f"article numbers changed {old_nums} -> {new_nums}")

    out["enacting_text"] = after
    out["articles"] = [a.model_dump(mode="json") for a in new_articles]

    records = [
        FlattenedTableRecord(
            field="enacting_text",
            marker=ft.marker,
            n_cols=ft.n_cols,
            n_cells=len(ft.cells),
            n_rows=len(ft.rows),
            headers=list(ft.headers),
            ragged=ft.ragged,
            merged_fill=ft.merged_fill,
            raw=ft.raw,
        )
        for ft in tables
    ]
    out["flattened_tables"] = [r.model_dump(mode="json") for r in records]
    return out, records, problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="apply the patch")
    ap.add_argument("--parsed-dir", type=Path, default=PARSED_DIR)
    args = ap.parse_args()

    paths = sorted(args.parsed_dir.glob("*.json"))
    touched, all_problems = [], []

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        doc = payload.get("document") or {}
        patched, records, problems = patch_document(doc)
        if not records:
            continue

        # Recitals and the preamble must come through untouched: this migration
        # deliberately does NOT adopt the newer recital parsing (see module docstring).
        for field in ("preamble_text", "recitals", "citations", "annexes"):
            if json.dumps(doc.get(field), ensure_ascii=False) != \
               json.dumps(patched.get(field), ensure_ascii=False):
                problems.append(f"{field} changed; migration must leave it alone")

        touched.append((path, patched, records, problems))
        all_problems.extend(f"{path.name}: {p}" for p in problems)

        print(f"\n{path.name}")
        print(f"  enacting_text {len(doc.get('enacting_text') or '')} -> "
              f"{len(patched['enacting_text'])} chars")
        print(f"  articles: {[a['number'] for a in patched['articles']]} "
              f"(unchanged: {[a['number'] for a in doc.get('articles') or []] == [a['number'] for a in patched['articles']]})")
        for r in records:
            print(f"  table marker={r.marker} cols={r.n_cols} rows={r.n_rows} "
                  f"ragged={r.ragged} merged_fill={r.merged_fill} "
                  f"headers={r.headers or '[] (positional)'}")
        if problems:
            for p in problems:
                print(f"  PROBLEM: {p}")

    print(f"\n{len(touched)} document(s) carry a flattened table.")

    if all_problems:
        print("\nREFUSING TO WRITE — the migration would change frozen structure:")
        for p in all_problems:
            print(f"  - {p}")
        return 1

    if not args.write:
        print("Dry run. Re-run with --write to apply.")
        return 0

    for path, patched, _records, _problems in touched:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["document"] = patched
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        print(f"wrote {path}")

    # Now that the parsed text is on disk, slide the gold sample's offsets onto it.
    doc_ids = {p.stem for p, _, _, _ in touched}
    if SAMPLE_PATH.exists():
        sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
        n_before = len(sample.get("segments", []))
        notes = remap_sample_offsets(sample, args.parsed_dir, doc_ids)
        assert len(sample.get("segments", [])) == n_before, "segment count must not change"
        SAMPLE_PATH.write_text(json.dumps(sample, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"\nremapped {len(notes)} segment offset(s) in {SAMPLE_PATH}:")
        for n in notes:
            print(f"  {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
