"""Draw the gold annotation sample from the locked EUR-Lex evaluation set.

Up to five segments per evaluation document, drawn at random within each
document, with the recital-to-article mix corrected at corpus level so the
sample's recital share matches the corpus (Section~\ref{sec:gold-sample}).
Short acts that hold fewer than five segments contribute all of them.

Also fixes, in advance, the 15 % second-annotator subsample: which segments
those are is decided here and never revisited.

Outputs
    data/gold/segment_sample.json      the frozen sample + IAA subsample

CLI:
    python scripts/sample_gold_segments.py [--per-doc 5] [--seed 20260731]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PARSED = ROOT / "data" / "parsed"
LISTS = ROOT / "data" / "doc_lists"
OUT = ROOT / "data" / "gold"


def segments_of(doc: dict) -> list[dict]:
    """Mirror src.extractor.enumerate_sections so gold and model output are
    keyed identically. Citation blocks and closing formulas are not segments."""
    segs = []
    pre = doc.get("preamble_text") or ""
    for i, rec in enumerate(doc.get("recitals") or []):
        text = pre[rec["char_start"]:rec["char_end"]]
        if text.strip():
            segs.append({
                "section_path": f"preamble.recitals[{i}]",
                "kind": "recital",
                "number": rec.get("number"),
                "char_start": rec["char_start"],
                "char_end": rec["char_end"],
                "n_words": len(text.split()),
                "text": text,
            })
    ena = doc.get("enacting_text") or ""
    for art in doc.get("articles") or []:
        text = ena[art["char_start"]:art["char_end"]]
        if text.strip():
            segs.append({
                "section_path": f"enacting.article_{art['number']}",
                "kind": "article",
                "number": art.get("number"),
                "char_start": art["char_start"],
                "char_end": art["char_end"],
                "n_words": len(text.split()),
                "text": text,
            })
    return segs


def allocate(k: int, n_rec: int, n_art: int, target_share: float) -> int:
    """How many of this document's k slots go to recitals."""
    want = round(k * target_share)
    return max(max(0, k - n_art), min(min(k, n_rec), want))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-doc", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--iaa-share", type=float, default=0.15)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "segment_sample.json"
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists -- sample already frozen. Use --force to redraw.")

    manifest = json.loads((LISTS / "eurlex_split_manifest.json").read_text(encoding="utf-8"))
    target_share = manifest["corpus_profile"]["recital_share"]
    eval_ids = manifest["eval_doc_ids"]

    rng = random.Random(args.seed)
    docs = {}
    for doc_id in eval_ids:
        d = json.loads((PARSED / f"{doc_id}.json").read_text(encoding="utf-8"))["document"]
        docs[doc_id] = {"title": d.get("title") or "", "segments": segments_of(d)}

    # 1. per-document allocation
    plan = {}
    for doc_id, d in docs.items():
        rec = [s for s in d["segments"] if s["kind"] == "recital"]
        art = [s for s in d["segments"] if s["kind"] == "article"]
        k = min(args.per_doc, len(rec) + len(art))
        plan[doc_id] = {"k": k, "n_rec": allocate(k, len(rec), len(art), target_share),
                        "rec": rec, "art": art}

    # 2. corpus-level correction of the recital share
    def share():
        tot = sum(p["k"] for p in plan.values())
        return sum(p["n_rec"] for p in plan.values()) / max(1, tot)

    for _ in range(200):
        cur = share()
        if abs(cur - target_share) < 1e-9:
            break
        if cur < target_share:
            cands = [i for i, p in plan.items()
                     if p["n_rec"] < min(p["k"], len(p["rec"]))
                     and p["k"] - p["n_rec"] > 0]
            if not cands:
                break
            plan[rng.choice(sorted(cands))]["n_rec"] += 1
            if abs(share() - target_share) > abs(cur - target_share):
                plan[sorted(cands)[0]]  # no-op; keep the move, it is within rounding
                break
        else:
            cands = [i for i, p in plan.items()
                     if p["n_rec"] > 0 and p["k"] - p["n_rec"] < len(p["art"])]
            if not cands:
                break
            plan[rng.choice(sorted(cands))]["n_rec"] -= 1
            if abs(share() - target_share) > abs(cur - target_share):
                break

    # 3. draw
    sample = []
    for doc_id in eval_ids:
        p = plan[doc_id]
        chosen = (rng.sample(sorted(p["rec"], key=lambda s: s["section_path"]), p["n_rec"])
                  + rng.sample(sorted(p["art"], key=lambda s: s["section_path"]),
                               p["k"] - p["n_rec"]))
        order = {s["section_path"]: i for i, s in enumerate(docs[doc_id]["segments"])}
        for s in sorted(chosen, key=lambda s: order[s["section_path"]]):
            sample.append({
                "segment_id": f"{doc_id}::{s['section_path']}",
                "doc_id": doc_id,
                "doc_title": docs[doc_id]["title"][:120],
                "section_path": s["section_path"],
                "kind": s["kind"],
                "number": s["number"],
                "char_start": s["char_start"],
                "char_end": s["char_end"],
                "n_words": s["n_words"],
            })

    # 4. IAA subsample, fixed in advance
    n_iaa = max(1, round(args.iaa_share * len(sample)))
    iaa = sorted(rng.sample(sorted(s["segment_id"] for s in sample), n_iaa))

    n_rec = sum(1 for s in sample if s["kind"] == "recital")
    payload = {
        "schema_version": 1,
        "frozen_on": date.today().isoformat(),
        "seed": args.seed,
        "per_doc_cap": args.per_doc,
        "split_manifest_sha1_eval": manifest["sha1_eval"],
        "n_segments": len(sample),
        "n_recital_segments": n_rec,
        "n_article_segments": len(sample) - n_rec,
        "recital_share": round(n_rec / len(sample), 3),
        "corpus_recital_share": target_share,
        "n_words_total": sum(s["n_words"] for s in sample),
        "iaa_subsample": {
            "share": args.iaa_share,
            "n_segments": n_iaa,
            "segment_ids": iaa,
            "note": "second annotator works these blind, without sight of the first annotation",
        },
        "segments": sample,
    }
    blob = json.dumps(payload, indent=2, ensure_ascii=False)
    out_path.write_text(blob, encoding="utf-8")

    print(f"{len(sample)} segments over {len(eval_ids)} docs "
          f"({n_rec} recitals / {len(sample)-n_rec} articles, "
          f"share={payload['recital_share']} vs corpus {target_share})")
    print(f"words: {payload['n_words_total']}  IAA subsample: {n_iaa} segments")
    print("sha1:", hashlib.sha1(blob.encode()).hexdigest()[:12])
    for doc_id in eval_ids:
        got = [s for s in sample if s["doc_id"] == doc_id]
        print(f"  {doc_id}: {len(got):2d}  " + ", ".join(s["section_path"] for s in got))


if __name__ == "__main__":
    main()
