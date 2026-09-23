"""Freeze the EUR-Lex corpus and produce the dev / eval split.

Run ONCE. The split is a commitment of the experimental design
(Section~\ref{sec:data-splits}): the development set is used to diagnose the
guideline, the evaluation set is touched once per guideline version.

Stratification: act type x document size, so that neither split is dominated
by one instrument type or by the very short "fixing export refunds" acts that
make up much of the corpus.

Outputs
    data/doc_lists/eurlex_dev.json
    data/doc_lists/eurlex_eval.json
    data/doc_lists/eurlex_split_manifest.json   (seed, hashes, strata table)

CLI:
    python scripts/make_eurlex_split.py [--n-eval 12] [--seed 20260731]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import random
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PARSED = ROOT / "data" / "parsed"
OUT = ROOT / "data" / "doc_lists"

ACT_TYPES = ["Regulation", "Decision", "Directive", "Recommendation"]
# "of 13 June 2001" -- the enactment date, not a citation number like "No 1905/2000"
DATE_RE = re.compile(
    r"of\s+\d{1,2}\s+"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+((?:19|20)\d{2})",
    re.I,
)


def doc_features(doc: dict) -> dict:
    title = doc.get("title") or ""
    m = DATE_RE.search(title)
    year = int(m.group(2)) if m else None

    act = "Other"
    for a in ACT_TYPES:
        if re.search(rf"\b{a}\b", title, re.I):
            act = a
            break
    body = "Other"
    for b in ("Commission", "Council", "Parliament"):
        if re.search(rf"\b{b}\b", title, re.I):
            body = b
            break

    n_rec = len(doc.get("recitals") or [])
    n_art = len(doc.get("articles") or [])
    text = (doc.get("preamble_text") or "") + (doc.get("enacting_text") or "")
    return {
        "doc_id": doc["celex"],
        "title": title,
        "year": year,
        "act_type": act,
        "body": body,
        "recital_count": n_rec,
        "article_count": n_art,
        "segment_count": n_rec + n_art,
        "n_words": len(text.split()),
        "char_length": len(text),
    }


def size_bucket(n_seg: int) -> str:
    if n_seg <= 3:
        return "S"      # short act: no recitals, one or two articles
    if n_seg <= 8:
        return "M"
    return "L"


def load_corpus() -> list[dict]:
    rows = []
    for p in sorted(PARSED.glob("train-*.json")):
        if p.suffix != ".json" or p.name.endswith(".bak.json"):
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))["document"]
        rows.append(doc_features(doc))
    return rows


def stratified_eval(rows: list[dict], n_eval: int, seed: int) -> list[str]:
    """Pick n_eval docs, allocating proportionally over (act_type, size) strata.

    Largest-remainder allocation keeps the eval set's stratum profile as close
    to the corpus profile as integer counts permit.
    """
    rng = random.Random(seed)
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        strata[(r["act_type"], size_bucket(r["segment_count"]))].append(r)

    n_total = len(rows)
    quotas = {k: len(v) * n_eval / n_total for k, v in strata.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    # largest remainder
    short = n_eval - sum(alloc.values())
    for k, _ in sorted(quotas.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True):
        if short <= 0:
            break
        if alloc[k] < len(strata[k]):
            alloc[k] += 1
            short -= 1

    picked: list[str] = []
    for k in sorted(strata):
        pool = sorted(strata[k], key=lambda r: r["doc_id"])
        picked += [r["doc_id"] for r in rng.sample(pool, alloc[k])]
    return sorted(picked)


def write_doc_list(path: Path, rows: list[dict], split: str, seed: int) -> str:
    payload = {
        "schema_version": 1,
        "corpus": "eurlex",
        "split": split,
        "frozen_on": date.today().isoformat(),
        "seed": seed,
        "n_docs": len(rows),
        "docs": rows,
    }
    blob = json.dumps(payload, indent=2, ensure_ascii=False)
    path.write_text(blob, encoding="utf-8")
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing frozen split")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "eurlex_split_manifest.json"
    if manifest_path.exists() and not args.force:
        raise SystemExit(
            f"{manifest_path} exists -- the split is already frozen. "
            "Re-running would invalidate every result measured on it. "
            "Pass --force only if you mean it."
        )

    rows = load_corpus()
    eval_ids = set(stratified_eval(rows, args.n_eval, args.seed))
    ev = [r for r in rows if r["doc_id"] in eval_ids]
    dv = [r for r in rows if r["doc_id"] not in eval_ids]

    h_dev = write_doc_list(OUT / "eurlex_dev.json", dv, "dev", args.seed)
    h_ev = write_doc_list(OUT / "eurlex_eval.json", ev, "eval", args.seed)

    def profile(rs):
        return {
            "n_docs": len(rs),
            "n_segments": sum(r["segment_count"] for r in rs),
            "n_recitals": sum(r["recital_count"] for r in rs),
            "n_articles": sum(r["article_count"] for r in rs),
            "recital_share": round(
                sum(r["recital_count"] for r in rs)
                / max(1, sum(r["segment_count"] for r in rs)), 3),
            "act_type": dict(Counter(r["act_type"] for r in rs)),
            "size_bucket": dict(Counter(size_bucket(r["segment_count"]) for r in rs)),
            "median_words": sorted(r["n_words"] for r in rs)[len(rs) // 2] if rs else 0,
        }

    manifest = {
        "schema_version": 1,
        "corpus": "eurlex",
        "frozen_on": date.today().isoformat(),
        "seed": args.seed,
        "method": "largest-remainder allocation over (act_type x size_bucket) strata",
        "size_buckets": {"S": "<=3 segments", "M": "4-8 segments", "L": ">=9 segments"},
        "sha1_dev": h_dev,
        "sha1_eval": h_ev,
        "corpus_profile": profile(rows),
        "dev_profile": profile(dv),
        "eval_profile": profile(ev),
        "eval_doc_ids": sorted(eval_ids),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding="utf-8")

    print(f"corpus {len(rows)} docs -> dev {len(dv)} / eval {len(ev)}")
    for name, pr in (("corpus", manifest["corpus_profile"]),
                     ("dev", manifest["dev_profile"]),
                     ("eval", manifest["eval_profile"])):
        print(f"  {name:6s} segs={pr['n_segments']:4d} "
              f"recital_share={pr['recital_share']:.3f} "
              f"acts={pr['act_type']} sizes={pr['size_bucket']}")
    print("eval:", ", ".join(sorted(eval_ids)))


if __name__ == "__main__":
    main()
