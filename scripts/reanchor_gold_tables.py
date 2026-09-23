"""Re-anchor the gold facts whose quotes pointed into the old `//` table runs.

Run after `scripts.patch_flattened_tables --write`.

Three of the 43 gold facts in the table-bearing segments had a `source_quote`
that spanned an old `//` cell boundary, so they stopped being verbatim once the
parser rebuilt the tables. The other 40 survived unchanged, because the
reconstructed text still contains their cell content verbatim.

One of the three (`train-000121::enacting.article_1::f04`) turns out to have
been **mis-anchored from the start**: its triple asserts that order No 40.0260
covers "Women's or girls dresses ...", but its quote was
`40.0260 // 26 (1 000 pieces)` — the order-number and category cells, which do
not contain the description at all. The `//` noise hid that. It is re-anchored
to the description text, matching how the parallel fact f07 is anchored, and
the defect is recorded rather than quietly corrected.

Every replacement is checked against the live segment text before anything is
written; the script refuses to write if any new quote is not verbatim.

Usage:
    python -m scripts.reanchor_gold_tables            # dry run
    python -m scripts.reanchor_gold_tables --write
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.gold_store import is_verbatim, sample_index, segment_text

GOLD = Path("data/gold")
AUTHORITY_TYPO = (
    "'U | Ministerio de Ganadería, Agricultura y Pesca, Dirección Granos "
    "(DIGRA), Unidad Ejucurora de Semillas, Montevideo"
)
AUTHORITY_OK = AUTHORITY_TYPO.replace("Ejucurora", "Ejecutora")

# gold_id -> (new source_quote, reviewer_note)
REANCHOR: dict[str, tuple[str, str]] = {
    "train-000019::enacting.article_1::f06": (
        AUTHORITY_TYPO,
        "Re-anchored 2026-08-06: the quote spanned the old '//' cell boundary. "
        "The reconstructed row separates cells with ' | ', so the same two "
        "cells are quoted in their new form. Slots unchanged. The source typo "
        "'Ejucurora' is preserved (decision log D-07).",
    ),
    "train-000019::enacting.article_2::f05": (
        AUTHORITY_OK,
        "Re-anchored 2026-08-06: as f06 in Article 1, but this Article spells "
        "the authority 'Ejecutora' (decision log D-07). Slots unchanged.",
    ),
    "train-000121::enacting.article_1::f04": (
        "Women's or girls dresses, of wool, of cotton or man-made fibres",
        "Re-anchored 2026-08-06. The previous quote, '40.0260 // 26 (1 000 "
        "pieces)', did not support this fact: the triple asserts the "
        "order-to-description mapping but the quote covered the order-number "
        "and category cells only. This was a mis-anchoring present in the "
        "original pre-annotation, not a consequence of the table rebuild; the "
        "'//' noise concealed it. Now anchored to the description cell, "
        "matching sibling fact f07. Slots unchanged.",
    ),
}

NOTES = {
    "train-000019": (
        "Articles 1 and 2 embed a table that the OJ plain-text export flattened "
        "into '//' separators. As of 2026-08-06 the parser reconstructs the grid "
        "(src/oj_table.py): the marker '1.2.3.4' / '1.2.3.4.5.6' declares the "
        "column count, cells are separated by ' | ', and the vertically merged "
        "country/authority cells are carried down into every row they span so "
        "each row is independently interpretable. The source table numbers its "
        "columns rather than labelling them, so no column names are attached. "
        "See decision log D-01."
    ),
    "train-000121": (
        "Article 1 embeds a table that the OJ plain-text export flattened into "
        "'//' separators. As of 2026-08-06 the parser reconstructs the grid "
        "(src/oj_table.py) and reattaches the real column headers (Order No, "
        "Category (unit), CN code, Description). The CN-code block remains one "
        "fact per order rather than one per code. See decision log D-01."
    ),
}

D01_UPDATE = {
    "problem": (
        "The OJ plain-text export flattens table rows into a run of '//' "
        "separators. Until 2026-08-06 cell boundaries and column geometry were "
        "not recoverable from the segment text, so the pre-annotation treated "
        "the run as opaque."
    ),
    "choice": (
        "Originally: annotate the prose frame and only unambiguously "
        "identifiable cell content, without reconstructing the column "
        "geometry. Superseded 2026-08-06 — the geometry IS recoverable: the "
        "dotted marker declares the column count and the empty cells encode "
        "vertical merges, so src/oj_table.py rebuilds the grid at parse time "
        "and renders one sentence per row. A grid is only emitted when the "
        "cell count divides evenly by the column count; otherwise the table is "
        "marked ragged and the rows are NOT invented. Merged grouping cells "
        "are carried down so every row names its own subject. For "
        "train-000121 the CN-code block is still one fact per order rather "
        "than one per code."
    ),
    "reason": (
        "Splitting to a granularity the source does not support would put "
        "invented structure into the gold, and every model would then be "
        "scored against a reading no annotator could reproduce. That argument "
        "is what still forbids fabricating rows for a ragged table; it no "
        "longer applies to the well-formed case, where the geometry is "
        "explicit in the encoding."
    ),
    "feeds_diagnosis": (
        "Table handling is a guideline gap: v1 says nothing about what to do "
        "with a linearised table, and the models are likely to diverge here "
        "for reasons unrelated to the legal content. Note the split of causes "
        "— the unreadable '//' run was a PARSER defect, not a guideline "
        "defect. Do not attribute the resulting disagreement to the guideline."
    ),
    "superseded_on": "2026-08-06",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    idx = sample_index()
    problems: list[str] = []
    planned: dict[str, list[tuple[str, str, str, str]]] = {}

    for gold_id, (new_quote, note) in REANCHOR.items():
        doc_id, section_path, fact_id = gold_id.split("::")
        meta = idx[f"{doc_id}::{section_path}"]
        text = segment_text(doc_id, section_path, meta["char_start"], meta["char_end"])
        if not is_verbatim(new_quote, text):
            problems.append(f"{gold_id}: replacement quote is NOT verbatim -> {new_quote!r}")
            continue
        planned.setdefault(doc_id, []).append((section_path, fact_id, new_quote, note))
        print(f"OK  {gold_id}\n    -> {new_quote!r}")

    if problems:
        print("\nREFUSING TO WRITE:")
        for p in problems:
            print(f"  - {p}")
        return 1

    if not args.write:
        print(f"\n{sum(len(v) for v in planned.values())} fact(s) ready. "
              f"Re-run with --write to apply.")
        return 0

    for doc_id, items in planned.items():
        path = GOLD / "facts" / f"{doc_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        for section_path, fact_id, new_quote, note in items:
            seg = next(s for s in payload["segments"]
                       if s["segment_id"] == f"{doc_id}::{section_path}")
            fact = next(f for f in seg["facts"]
                        if f["gold_id"] == f"{doc_id}::{section_path}::{fact_id}")
            fact["source_quote"] = new_quote
            fact["reviewer_note"] = note
        if doc_id in NOTES:
            payload["note"] = NOTES[doc_id]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {path} ({len(items)} fact(s))")

    log_path = GOLD / "decision_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8"))
    for entry in log["entries"]:
        if entry["id"] == "D-01":
            entry.update(D01_UPDATE)
            entry["title"] = "Tables flattened into '//' separators (parser fix 2026-08-06)"
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {log_path} (D-01 superseded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
