"""Apply your adjudication of the SECOND annotation, without touching the raw pass.

    python scripts/make_gold_review_page.py --facts-root data/gold/annotator_b
    # adjudicate in data/gold/review_annotator_b.html, download the decisions
    python -m scripts.apply_second_annotation_review data/gold/review_decisions.json

Reads `data/gold/annotator_b/` (the model's blind annotation), applies your
accept / edit / reject decisions, and writes the result to
`data/gold/annotator_b_reviewed/`. **`annotator_b/` itself is never modified.**

## Why the raw pass stays frozen

Inter-annotator agreement is computed **before** any reconciliation. That is the
whole ordering the number depends on:

  1. gold: machine pre-annotation, adjudicated by you
  2. second pass: annotated blind by a model, from the segment text alone
  3. **IAA = raw gold vs raw second pass** — the reliability estimate
  4. reconciliation: you adjudicate the second pass, producing this file

Run step 4 before step 3 and there is no reliability estimate left: the same
person's judgement would be inside both annotations, so agreement measures how
consistently you harmonise, not how reproducibly the protocol can be applied.
Keeping `annotator_b/` frozen is what keeps step 3 available afterwards.

Adjudicating the second pass does make it *human-adjudicated* — a real upgrade to
its provenance, and the thing that lets a reconciled final gold be built from
both passes. What it cannot do is turn it into an independent second annotation
after the fact. Every fact here therefore carries both stamps: `annotated_by`
(the model, blind) and `review` (you), and the file records
`iaa_must_use: data/gold/annotator_b`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GOLD = ROOT / "data" / "gold"
RAW = GOLD / "annotator_b"
DEST = GOLD / "annotator_b_reviewed"

FACT_LEVEL = {"natural_language", "source_quote", "condition", "temporal_context",
              "logic_group", "logic_op", "boilerplate", "reviewer_note"}
TRIPLE_LEVEL = {"subject", "predicate", "object"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("decisions", type=Path, help="a decisions JSON from the review page")
    ap.add_argument("--reviewer", default="annotator")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not RAW.exists():
        raise SystemExit(f"no {RAW} — import the second annotation first "
                         f"(scripts/import_second_annotation.py)")

    payload = json.loads(args.decisions.read_text(encoding="utf-8"))
    root = payload.get("facts_root")
    if root and root != "annotator_b":
        raise SystemExit(
            f"refusing: {args.decisions} was swept from data/gold/{root}, not "
            f"data/gold/annotator_b.\n  For the gold itself use "
            f"scripts/apply_gold_review.py.")
    dec = payload.get("decisions") or {}
    if not dec:
        raise SystemExit(f"no `decisions` in {args.decisions}")

    on = str(date.today())
    stats: Counter = Counter()
    unknown: set[str] = set()
    out: dict[Path, dict] = {}
    seen: set[str] = set()

    for p in sorted(RAW.glob("*.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        for seg in doc["segments"]:
            for f in seg["facts"]:
                d = dec.get(f["gold_id"])
                if not d:
                    stats["untouched"] += 1
                    continue
                seen.add(f["gold_id"])
                f["status"] = d["status"]
                f["reviewer_note"] = d.get("note", "")
                f["review"] = {"by": args.reviewer, "mode": "human", "on": on}
                stats[d["status"]] += 1
                for k, v in (d.get("edits") or {}).items():
                    if v is None or v == "":
                        continue
                    if k in FACT_LEVEL:
                        f[k] = v
                    elif k in TRIPLE_LEVEL:
                        f["triples"][0][k] = v
                    elif k == "triples":
                        f["triples"] = v
                    else:
                        unknown.add(k)
            for a in payload.get("added", []):
                if a["segment_id"] == seg["segment_id"] and not any(
                        x["gold_id"] == a["fact"]["gold_id"] for x in seg["facts"]):
                    fact = dict(a["fact"])
                    fact.setdefault("status", "edited")
                    fact["provenance_override"] = f"written by reviewer ({args.reviewer})"
                    fact["review"] = {"by": args.reviewer, "mode": "human", "on": on}
                    seg["facts"].append(fact)
                    stats["added"] += 1
        doc["provenance"] = {**doc.get("provenance", {}),
                             "reviewed_by": args.reviewer,
                             "reviewed_on": on,
                             "review_kind": "reconciliation of the second annotation",
                             "iaa_must_use": "data/gold/annotator_b",
                             "iaa_note": "Agreement is computed on the RAW second "
                                         "annotation, never on this file: this one "
                                         "carries the gold owner's judgement, so it "
                                         "is no longer independent of the gold."}
        out[DEST / p.name] = doc

    missing = set(dec) - seen
    if missing:
        print(f"  !! {len(missing)} decision(s) name a fact that is not in "
              f"{RAW.name}: {sorted(missing)[:5]}")
        raise SystemExit("nothing written")
    if unknown:
        print("  !! ignored unknown edit fields:", sorted(unknown))

    print(f"{sum(stats[k] for k in ('accepted','edited','rejected'))} decision(s) applied: "
          + " · ".join(f"{k} {stats[k]}" for k in
                       ("accepted", "edited", "rejected", "added", "untouched") if stats[k]))
    if stats["untouched"]:
        print(f"  {stats['untouched']} fact(s) still have no decision — the second "
              f"annotation is only partly adjudicated")
    if args.dry_run:
        print("  --dry-run: nothing written")
        return

    DEST.mkdir(exist_ok=True)
    for path, doc in out.items():
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  wrote {DEST}  ({len(out)} file(s)); {RAW} left untouched")


if __name__ == "__main__":
    main()
