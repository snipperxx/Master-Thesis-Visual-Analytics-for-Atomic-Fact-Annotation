"""Recompute the frozen split's descriptive statistics after the re-parse.

The dev/eval **document assignment is not touched** — it stays exactly the
partition drawn on 2026-07-31, so the existing gold remains valid. What is
recomputed is every count derived from the parsed text (`recital_count`,
`article_count`, `segment_count`, `n_words`, `char_length`) plus the three
profiles, because those were computed from `data/parsed` before the recital
fallback reached it and therefore understate the corpus by ~35%.

The honest consequence is recorded in the manifest itself, under
`stratification_caveat`: the split was allocated over
(act_type x size_bucket) strata using the stale segment counts, so 21 of 50
documents would fall in a different size bucket today. The partition is still a
valid *random* assignment — nothing about it depends on the outcome being
measured — it is just no longer optimally *stratified*. That belongs in the
thesis as a limitation, not as a reason to redraw (redrawing discards 277 gold
facts and ~13 h of adjudication).

Reads `src.eurlex_parse`-shaped documents from data/parsed and reuses
`scripts.make_eurlex_split.doc_features` / `size_bucket` so the field
definitions cannot drift from the freeze that produced them.

Usage:
    python -m scripts.refresh_split_profiles            # dry run
    python -m scripts.refresh_split_profiles --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path

from scripts.make_eurlex_split import doc_features, size_bucket

PARSED = Path("data/parsed")
LISTS = Path("data/doc_lists")
MANIFEST = LISTS / "eurlex_split_manifest.json"

CAVEAT = (
    "The dev/eval partition was allocated over (act_type x size_bucket) strata "
    "on 2026-07-31 using segment counts from a data/parsed that predated the "
    "2026-06-16 'Whereas' recital fallback. Recomputed 2026-08-06, 21 of 50 "
    "documents fall in a different size_bucket, and the corpus holds 549 "
    "segments rather than the 395 the allocation saw. The partition is "
    "therefore a valid RANDOM assignment but not an optimally STRATIFIED one. "
    "It is kept deliberately: the document assignment cannot depend on any "
    "measured outcome, and redrawing it would invalidate the gold annotation. "
    "Report as a limitation. See doc/CHANGELOG.md 2026-08-06."
)


def profile(rows: list[dict]) -> dict:
    n_seg = sum(r["segment_count"] for r in rows)
    return {
        "n_docs": len(rows),
        "n_segments": n_seg,
        "n_recitals": sum(r["recital_count"] for r in rows),
        "n_articles": sum(r["article_count"] for r in rows),
        "recital_share": round(sum(r["recital_count"] for r in rows) / max(1, n_seg), 3),
        "act_type": dict(Counter(r["act_type"] for r in rows)),
        "size_bucket": dict(Counter(size_bucket(r["segment_count"]) for r in rows)),
        "median_words": sorted(r["n_words"] for r in rows)[len(rows) // 2] if rows else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    lists = {
        split: json.loads((LISTS / f"eurlex_{split}.json").read_text(encoding="utf-8"))
        for split in ("dev", "eval")
    }
    assignment = {split: [d["doc_id"] for d in payload["docs"]]
                  for split, payload in lists.items()}

    fresh: dict[str, dict] = {}
    for doc_id in assignment["dev"] + assignment["eval"]:
        doc = json.loads((PARSED / f"{doc_id}.json").read_text(encoding="utf-8"))["document"]
        fresh[doc_id] = doc_features(doc)

    rows_all = [fresh[d] for d in assignment["dev"] + assignment["eval"]]
    new_profiles = {
        "corpus_profile": profile(rows_all),
        "dev_profile": profile([fresh[d] for d in assignment["dev"]]),
        "eval_profile": profile([fresh[d] for d in assignment["eval"]]),
    }

    moved = [d for d in fresh
             if size_bucket(fresh[d]["segment_count"])
             != size_bucket(next(x["segment_count"] for s in lists.values()
                                 for x in s["docs"] if x["doc_id"] == d))]

    for key, new in new_profiles.items():
        old = manifest.get(key, {})
        print(f"\n{key}")
        for field in ("n_docs", "n_segments", "n_recitals", "n_articles",
                      "recital_share", "median_words"):
            o, n = old.get(field), new.get(field)
            flag = "" if o == n else "   <-- changed"
            print(f"  {field:<16} {str(o):>10} -> {str(n):<10}{flag}")
        print(f"  {'size_bucket':<16} {str(old.get('size_bucket')):>10} -> {new['size_bucket']}")
        print(f"  {'act_type':<16} {str(old.get('act_type')):>10} -> {new['act_type']}")

    print(f"\ndocuments whose size_bucket changes: {len(moved)}")
    print(f"document assignment: dev={len(assignment['dev'])} eval={len(assignment['eval'])} "
          f"(UNCHANGED)")

    if not args.write:
        print("\nDry run. Re-run with --write to apply.")
        return 0

    # Per-document feature rows in the split lists, assignment preserved.
    hashes = {}
    for split, payload in lists.items():
        payload["docs"] = [fresh[d] for d in assignment[split]]
        payload["profiles_refreshed_on"] = date.today().isoformat()
        blob = json.dumps(payload, indent=2, ensure_ascii=False)
        (LISTS / f"eurlex_{split}.json").write_text(blob, encoding="utf-8")
        hashes[split] = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]
        print(f"wrote eurlex_{split}.json (sha1 {hashes[split]})")

    manifest.update(new_profiles)
    # The original hashes identified the frozen files; keep them as provenance
    # and record the post-refresh ones separately so neither is ambiguous.
    manifest["sha1_dev_frozen_20260731"] = manifest.get("sha1_dev")
    manifest["sha1_eval_frozen_20260731"] = manifest.get("sha1_eval")
    manifest["sha1_dev"] = hashes["dev"]
    manifest["sha1_eval"] = hashes["eval"]
    manifest["profiles_refreshed_on"] = date.today().isoformat()
    manifest["stratification_caveat"] = CAVEAT
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"wrote {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
