"""Re-parse the EUR-Lex corpus so data/parsed matches the current parser.

Why this is needed
------------------
The "Whereas" recital fallback for pre-1999 acts landed in `src/eurlex_parse.py`
on 2026-06-16, but only `train-000014` was re-parsed by hand. The 2026-07-31
freeze was then computed from the still-stale files, so 24 of 50 documents carry
zero parsed recitals and their entire preamble has never been segmented, never
been sent to a model, and was never eligible for gold sampling:

    eval  90 -> 116 segments (+26, +29%)   |   dev  305 -> 433 (+128, +42%)

What this script does and does not do
------------------------------------
It re-parses from the HuggingFace source, which adopts BOTH the recital fallback
and the flattened-table reconstruction (`src/oj_table.py`), and then remaps the
gold sample's stored offsets onto the new text.

It does NOT redraw the dev/eval split. The 38/12 document assignment is a valid
random partition and is kept, so the existing 277 gold facts stay valid. The
consequence to document: `size_bucket` was computed from the stale segment
counts, so the split is correctly *randomised* but not optimally *stratified* —
21 of 50 documents would land in a different bucket today. That is a limitation
to report, not a validity threat, because random assignment is preserved.
Redrawing the split would discard the gold and put ~13 h of adjudication back on
the critical path.

Safety
------
Refuses to write if any sampled gold segment would lose its `section_path`, or
if any existing gold quote would stop being verbatim, unless
`--allow-gold-breakage` is passed. Run without `--write` first.

Usage
-----
    python -m scripts.reparse_corpus                 # dry run + full diff report
    python -m scripts.reparse_corpus --write
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.eurlex_parse import parse_plain_text
from src.gold_store import GOLD_FACTS, is_verbatim, remap_sample_offsets

PARSED = Path("data/parsed")
SAMPLE_PATH = Path("data/gold/segment_sample.json")
DATASET, CONFIG, SPLIT = "coastalcph/lex_glue", "eurlex", "train"

SIZE_BUCKETS = (("S", 3), ("M", 8))          # S <=3, M 4-8, L >=9


def bucket(n_segments: int) -> str:
    for name, hi in SIZE_BUCKETS:
        if n_segments <= hi:
            return name
    return "L"


def _fields_changed(old: dict, new: dict) -> list[str]:
    out = []
    for f in ("title", "preamble_text", "enacting_text", "concluding_formulas"):
        if (old.get(f) or "") != (new.get(f) or ""):
            out.append(f)
    for f in ("citations", "recitals", "articles", "annexes"):
        n_old, n_new = len(old.get(f) or []), len(new.get(f) or [])
        if n_old != n_new:
            out.append(f"{f}({n_old}->{n_new})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--allow-gold-breakage", action="store_true")
    ap.add_argument("--parsed-dir", type=Path, default=PARSED)
    args = ap.parse_args()

    from datasets import load_dataset

    paths = sorted(args.parsed_dir.glob("train-*.json"))
    print(f"loading {DATASET}/{CONFIG}:{SPLIT} (cached) for {len(paths)} document(s)...")
    ds = load_dataset(DATASET, CONFIG, split=SPLIT,
                      cache_dir=str(Path("data/hf_cache")))

    reparsed: dict[str, dict] = {}
    changed: list[tuple[str, list[str], str, str]] = []

    for path in paths:
        doc_id = path.stem
        payload = json.loads(path.read_text(encoding="utf-8"))
        old = payload["document"]
        new = parse_plain_text(ds[int(doc_id.split("-")[1])]["text"],
                               doc_id=doc_id).model_dump(mode="json")
        reparsed[doc_id] = new

        diffs = _fields_changed(old, new)
        if diffs:
            n_old = len(old.get("recitals") or []) + len(old.get("articles") or [])
            n_new = len(new.get("recitals") or []) + len(new.get("articles") or [])
            changed.append((doc_id, diffs, f"{n_old}->{n_new}",
                            f"{bucket(n_old)}->{bucket(n_new)}"))

    print(f"\n=== {len(changed)} of {len(paths)} document(s) change ===")
    print(f"{'doc':<14} {'segments':>10} {'bucket':>10}  fields")
    for doc_id, diffs, segs, buck in changed:
        mark = "  <-- bucket moves" if buck.split("->")[0] != buck.split("->")[1] else ""
        print(f"{doc_id:<14} {segs:>10} {buck:>10}  {', '.join(diffs)}{mark}")

    # ---- what this does to the gold sample -------------------------------
    problems: list[str] = []
    if SAMPLE_PATH.exists():
        sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
        for seg in sample["segments"]:
            doc_id, path_ = seg["doc_id"], seg["section_path"]
            new = reparsed.get(doc_id)
            if new is None:
                continue
            if path_.startswith("preamble.recitals["):
                i = int(path_[len("preamble.recitals["):-1])
                src = (new.get("recitals") or [])[i:i + 1]
                container = new.get("preamble_text") or ""
            else:
                num = path_.split("article_", 1)[1]
                src = [a for a in new.get("articles") or [] if a["number"] == num]
                container = new.get("enacting_text") or ""
            if not src:
                problems.append(f"{seg['segment_id']}: section_path disappears")
                continue
            text = container[src[0]["char_start"]:src[0]["char_end"]]

            gold_path = GOLD_FACTS / f"{doc_id}.json"
            if not gold_path.exists():
                continue
            gold = json.loads(gold_path.read_text(encoding="utf-8"))
            gseg = next((s for s in gold["segments"]
                         if s["segment_id"] == seg["segment_id"]), None)
            for fact in (gseg or {}).get("facts", []):
                if not is_verbatim(fact["source_quote"], text):
                    problems.append(f"{fact['gold_id']}: quote stops being verbatim "
                                    f"-> {fact['source_quote'][:60]!r}")

    print(f"\n=== gold impact: {len(problems)} problem(s) ===")
    for p in problems:
        print(f"  {p}")
    if not problems:
        print("  none — every sampled segment survives and every quote stays verbatim")

    if problems and not args.allow_gold_breakage:
        print("\nREFUSING TO WRITE. Re-run with --allow-gold-breakage to proceed "
              "anyway (then re-anchor the listed facts).")
        return 1

    if not args.write:
        print("\nDry run. Re-run with --write to apply.")
        return 0

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["document"] = reparsed[path.stem]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nwrote {len(paths)} parsed document(s)")

    if SAMPLE_PATH.exists():
        sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
        n_before = len(sample["segments"])
        notes = remap_sample_offsets(sample, args.parsed_dir)
        assert len(sample["segments"]) == n_before, "remap must not change the sample"
        SAMPLE_PATH.write_text(json.dumps(sample, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"remapped {len(notes)} segment offset(s) in {SAMPLE_PATH}")
        for n in notes:
            print(f"  {n}")

    print("\nNext: python -m src.gold_store   (must print OK)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
