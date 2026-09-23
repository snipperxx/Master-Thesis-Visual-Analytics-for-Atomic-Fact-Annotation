"""Top the gold segment sample back up to its per-document cap after the re-parse.

Why a top-up and not a redraw
-----------------------------
Four evaluation documents could not offer their full quota when the sample was
drawn on 2026-07-31, because the stale parse showed them as having no recitals:

    train-000019  3 -> 5    train-000116  4 -> 5
    train-000121  2 -> 4    train-000148  2 -> 5      (cap 5; 000121 has only 4)

Redrawing all twelve documents would invalidate the 277 gold facts already
pre-annotated on the surviving 51 segments — roughly 13 h of adjudication thrown
away — so the existing picks are kept and only the shortfall is drawn.

What that costs, stated plainly: every newly available segment in those four
documents is a recital (they are exactly the segments the stale parse hid), so
the top-up is not a fresh stratified draw. It moves the sample's recital share
from 0.471 toward the corpus value, which the re-parse corrected from 0.484 to
0.628 — i.e. the top-up reduces the deviation rather than introducing one, but
the added segments are recitals by construction and not by chance. Recorded in
the sample under `topup`.

The IAA subsample grows from 8 to 9 segments (15 % of 59). The original eight are
left exactly as they were — they were fixed in advance and must not be revisited
— and the ninth is drawn from the newly added segments only.

Usage:
    python -m scripts.topup_gold_sample            # dry run
    python -m scripts.topup_gold_sample --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import date
from pathlib import Path

from scripts.sample_gold_segments import segments_of

PARSED = Path("data/parsed")
LISTS = Path("data/doc_lists")
SAMPLE_PATH = Path("data/gold/segment_sample.json")
TOPUP_SEED = 20260806


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--seed", type=int, default=TOPUP_SEED)
    args = ap.parse_args()

    sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    manifest = json.loads((LISTS / "eurlex_split_manifest.json").read_text(encoding="utf-8"))
    target_share = manifest["corpus_profile"]["recital_share"]
    cap = sample["per_doc_cap"]
    eval_ids = manifest["eval_doc_ids"]

    have: dict[str, set[str]] = {}
    for seg in sample["segments"]:
        have.setdefault(seg["doc_id"], set()).add(seg["section_path"])

    rng = random.Random(args.seed)
    additions: list[dict] = []

    for doc_id in eval_ids:
        doc = json.loads((PARSED / f"{doc_id}.json").read_text(encoding="utf-8"))["document"]
        all_segs = segments_of(doc)
        taken = have.get(doc_id, set())
        want = min(cap, len(all_segs)) - len(taken)
        if want <= 0:
            continue
        pool = [s for s in all_segs if s["section_path"] not in taken]
        # Deterministic candidate order before sampling, so the seed fully
        # determines the outcome regardless of dict/JSON ordering.
        pool.sort(key=lambda s: s["section_path"])
        chosen = rng.sample(pool, min(want, len(pool)))
        order = {s["section_path"]: i for i, s in enumerate(all_segs)}
        for s in sorted(chosen, key=lambda s: order[s["section_path"]]):
            additions.append({
                "segment_id": f"{doc_id}::{s['section_path']}",
                "doc_id": doc_id,
                "doc_title": (doc.get("title") or "")[:120],
                "section_path": s["section_path"],
                "kind": s["kind"],
                "number": s["number"],
                "char_start": s["char_start"],
                "char_end": s["char_end"],
                "n_words": s["n_words"],
                "added_in_topup": True,
            })
        print(f"{doc_id}: {len(taken)} -> {len(taken) + len(chosen)}  "
              f"(of {len(all_segs)} available)  + "
              + ", ".join(s["section_path"] for s in
                          sorted(chosen, key=lambda s: order[s["section_path"]])))

    if not additions:
        print("nothing to add; sample already at cap")
        return 0

    merged = sample["segments"] + additions
    doc_order = {d: i for i, d in enumerate(eval_ids)}
    merged.sort(key=lambda s: (doc_order.get(s["doc_id"], 999), s["section_path"]))

    n_rec = sum(1 for s in merged if s["kind"] == "recital")
    n_new_iaa = max(1, round(sample["iaa_subsample"]["share"] * len(merged)))
    existing_iaa = list(sample["iaa_subsample"]["segment_ids"])
    extra = n_new_iaa - len(existing_iaa)
    added_iaa: list[str] = []
    if extra > 0:
        added_iaa = rng.sample(sorted(s["segment_id"] for s in additions),
                              min(extra, len(additions)))

    print(f"\nsegments {len(sample['segments'])} -> {len(merged)}  "
          f"(recitals {sample['n_recital_segments']} -> {n_rec}, "
          f"share {sample['recital_share']} -> {round(n_rec / len(merged), 3)}, "
          f"corpus {target_share})")
    print(f"words {sample['n_words_total']} -> {sum(s['n_words'] for s in merged)}")
    print(f"IAA {len(existing_iaa)} -> {len(existing_iaa) + len(added_iaa)} "
          f"(added: {', '.join(added_iaa) or 'none'})")
    print(f"\n{len(additions)} new segment(s) need pre-annotation.")

    if not args.write:
        print("Dry run. Re-run with --write to apply.")
        return 0

    sample["segments"] = merged
    sample["n_segments"] = len(merged)
    sample["n_recital_segments"] = n_rec
    sample["n_article_segments"] = len(merged) - n_rec
    sample["recital_share"] = round(n_rec / len(merged), 3)
    sample["corpus_recital_share"] = target_share
    sample["n_words_total"] = sum(s["n_words"] for s in merged)
    sample["split_manifest_sha1_eval"] = manifest["sha1_eval"]
    sample["iaa_subsample"]["segment_ids"] = sorted(existing_iaa + added_iaa)
    sample["iaa_subsample"]["n_segments"] = len(existing_iaa) + len(added_iaa)
    sample["topup"] = {
        "on": date.today().isoformat(),
        "seed": args.seed,
        "reason": ("Four documents could not fill their quota in the 2026-07-31 draw "
                   "because the then-current data/parsed showed them with zero "
                   "recitals. The original 51 picks are kept so the existing gold "
                   "stays valid; only the shortfall was drawn."),
        "caveat": ("Every newly available segment in those documents is a recital, "
                   "so the added segments are recitals by construction rather than "
                   "by chance. This moves the sample's recital share toward the "
                   "corpus value (0.628) rather than away from it, but the top-up "
                   "is not itself a stratified random draw."),
        "added_segment_ids": [s["segment_id"] for s in additions],
        "iaa_added_segment_ids": sorted(added_iaa),
        "iaa_note": ("The original eight IAA segments were fixed in advance and are "
                     "unchanged; the ninth was drawn from the added segments only."),
    }
    blob = json.dumps(sample, indent=2, ensure_ascii=False)
    SAMPLE_PATH.write_text(blob, encoding="utf-8")
    print(f"wrote {SAMPLE_PATH}  sha1 {hashlib.sha1(blob.encode()).hexdigest()[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
