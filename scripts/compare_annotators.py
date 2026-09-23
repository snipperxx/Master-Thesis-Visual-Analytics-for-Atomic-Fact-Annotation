"""Compare the second annotation against the first pass, and write the figures.

    python -m scripts.compare_annotators                 # report + runs/annotator_ab_v1.json
    python -m scripts.compare_annotators --no-write      # report only

Two annotations of the same frozen 59-segment sample:

    A  data/gold/facts_hires/   first pass, claude-opus-5 pre-annotation
                                re-decomposed under data/gold/calibration/_RULES.md
    B  data/gold/annotator_b/   second annotator, blind, from the rules and the
                                segment text alone

**Two numbers, and only one of them means what it looks like.**

`jaccard_one_to_one` matches each fact to at most one partner (Hungarian over the
cosine matrix). It is what `src/annotator_compare.py` reports for the poster and
it is a **lower bound here**, because it structurally penalises whichever
annotator packages coarser: if B writes one fact covering three of A's, two of
A's count as missed even though the content is present. Do not report this
figure as the agreement of record. (`gold_compare.compare_soft` has scored
many-to-many coverage since 2026-09-23, but this script does not call it: it is
a fact-unit measure at its own `THRESHOLD`.)

`covered_a_by_b` / `covered_b_by_a` allow many-to-one — a fact counts as covered
if ANY fact on the other side reaches the threshold. That separates *content that
is missing* from *content that is packaged differently*, which is the whole
question §4.5 exists to ask.

The source-quote span statistics are the packaging-independent evidence: both
annotators anchor every fact in the same segment text, so comparing the spans
says who took larger bites without any embedding in the loop.

Segments are also reported in three classes, because they are not one condition:
the three flattened-table segments were re-asked under prompt v2.4 (rule 11), and
three segments of `train-000000` are still stamped with the superseded v2.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import gold_store as gs                       # noqa: E402
from src.alignment import (backend_name, encode_facts_joint,  # noqa: E402
                           fact_text_mode, fact_text_report)

GOLD = ROOT / "data" / "gold"
OUT = ROOT / "runs" / "annotator_ab_v1.json"

# NOT a calibrated threshold. tau = 0.78 was swept on dev
# (runs/alignment_calibration_dev_triple.json) and the gold slot gates were
# tuned separately; 0.60 has no such record anywhere in the repository. It is
# carried as a known limitation, not as a result — see doc/bugfix_log.md
# BUG-2026-08-27-B. Do not quote a figure from this script without it.
THRESHOLD = 0.60
THRESHOLD_PROVENANCE = "uncalibrated default; no dev sweep on record"

# The three flattened-table segments (re-asked under prompt v2.4, rule 11) and the
# three still carrying the superseded v2 prompt. Kept explicit rather than derived
# so a silent change to the data cannot silently change the reported split.
TABLES = ("train-000019::enacting.article_1",
          "train-000019::enacting.article_2",
          "train-000121::enacting.article_1")
STALE_PROMPT = ("train-000000::enacting.article_2",
                "train-000000::preamble.recitals[0]",
                "train-000000::preamble.recitals[2]")


def load(dirname: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for p in sorted((GOLD / dirname).glob("*.json")):
        payload = json.loads(p.read_text(encoding="utf-8"))
        for seg in payload["segments"]:
            out[seg["segment_id"]] = seg["facts"]
    if not out:
        raise SystemExit(f"no annotation under {GOLD / dirname}")
    return out


def _hungarian(cos: np.ndarray, threshold: float) -> int:
    from scipy.optimize import linear_sum_assignment
    if cos.size == 0:
        return 0
    ri, ci = linear_sum_assignment(-cos)
    return sum(1 for i, j in zip(ri, ci) if cos[i, j] >= threshold)


def _span(quote: str, folded_text: str) -> tuple[int, int] | None:
    q = gs.fold(quote)
    i = folded_text.find(q)
    return (i, i + len(q)) if i >= 0 and q else None


def per_segment(A: dict, B: dict, idx: dict, threshold: float) -> list[dict]:
    rows = []
    iaa = set(gs.load_sample()["iaa_subsample"]["segment_ids"])
    for sid in sorted(A):
        fa, fb = A[sid], B.get(sid, [])
        m = idx[sid]
        text = gs.fold(gs.segment_text(m["doc_id"], m["section_path"],
                                       m["char_start"], m["char_end"]))
        if fa and fb:
            ea, eb = encode_facts_joint(fa, fb)
            cos = ea @ eb.T
        else:
            cos = np.zeros((len(fa), len(fb)))
        matched = _hungarian(cos, threshold)
        best_a = cos.max(axis=1) if cos.size else np.zeros(len(fa))
        best_b = cos.max(axis=0) if cos.size else np.zeros(len(fb))
        sa = [_span(f["source_quote"], text) for f in fa]
        sb = [_span(f["source_quote"], text) for f in fb]
        rows.append({
            "segment_id": sid,
            "iaa": sid in iaa,
            "klass": ("table" if sid in TABLES
                      else "stale_prompt" if sid in STALE_PROMPT else "prose"),
            "n_a": len(fa), "n_b": len(fb), "matched": matched,
            "covered_a": float(np.mean(best_a >= threshold)) if len(fa) else 1.0,
            "covered_b": float(np.mean(best_b >= threshold)) if len(fb) else 1.0,
            "mean_best_a": float(np.mean(best_a)) if len(fa) else 0.0,
            "mean_best_b": float(np.mean(best_b)) if len(fb) else 0.0,
            "a_spans_inside_b": sum(1 for s in sa if s and any(
                u and u[0] <= s[0] and u[1] >= s[1] for u in sb)),
            "b_spans_inside_a": sum(1 for s in sb if s and any(
                u and u[0] <= s[0] and u[1] >= s[1] for u in sa)),
            "span_len_a": [s[1] - s[0] for s in sa if s],
            "span_len_b": [s[1] - s[0] for s in sb if s],
            "segment_chars": len(text),
        })
    return rows


def summarise(rows: list[dict], label: str) -> dict:
    na = sum(r["n_a"] for r in rows)
    nb = sum(r["n_b"] for r in rows)
    matched = sum(r["matched"] for r in rows)
    cov_a = [r["covered_a"] for r in rows for _ in range(r["n_a"])]
    cov_b = [r["covered_b"] for r in rows for _ in range(r["n_b"])]
    # covered_* are per-segment means; weight them by fact count to get the share
    n_cov_a = sum(round(r["covered_a"] * r["n_a"]) for r in rows)
    n_cov_b = sum(round(r["covered_b"] * r["n_b"]) for r in rows)
    la = [x for r in rows for x in r["span_len_a"]]
    lb = [x for r in rows for x in r["span_len_b"]]
    delta = [r["n_b"] - r["n_a"] for r in rows]
    return {
        "label": label,
        "n_segments": len(rows),
        "facts_a": na, "facts_b": nb,
        "matched_one_to_one": matched,
        "jaccard_one_to_one": round(matched / (na + nb - matched), 4) if na + nb - matched else 1.0,
        "covered_a_by_b": round(n_cov_a / na, 4) if na else None,
        "covered_b_by_a": round(n_cov_b / nb, 4) if nb else None,
        "a_spans_inside_a_b_span": round(
            sum(r["a_spans_inside_b"] for r in rows) / na, 4) if na else None,
        "b_spans_inside_an_a_span": round(
            sum(r["b_spans_inside_a"] for r in rows) / nb, 4) if nb else None,
        "median_span_chars_a": int(np.median(la)) if la else None,
        "median_span_chars_b": int(np.median(lb)) if lb else None,
        "mean_segment_chars": int(np.mean([r["segment_chars"] for r in rows])),
        "count_delta_b_minus_a": {
            "mean": round(float(np.mean(delta)), 2),
            "min": min(delta), "max": max(delta),
            "shape": dict(Counter("B coarser" if d < 0 else "B finer" if d > 0
                                  else "same count" for d in delta)),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--worst", type=int, default=10,
                    help="how many lowest-coverage segments to list")
    args = ap.parse_args()

    A, B = load("facts_hires"), load("annotator_b")
    missing = sorted(set(A) - set(B))
    idx = gs.sample_index()

    rows = per_segment(A, B, idx, args.threshold)
    blocks = [
        summarise(rows, "all 59"),
        summarise([r for r in rows if r["iaa"]], "IAA subsample (9)"),
        summarise([r for r in rows if r["klass"] == "prose"], "prose, one condition"),
        summarise([r for r in rows if r["klass"] == "table"], "flattened tables (prompt v2.4)"),
        summarise([r for r in rows if r["klass"] == "stale_prompt"], "still on prompt v2"),
    ]

    flat_a = [f for fs in A.values() for f in fs]
    flat_b = [f for fs in B.values() for f in fs]
    rep_a, rep_b = fact_text_report(flat_a), fact_text_report(flat_b)

    print(f"backend: {backend_name()}   threshold: {args.threshold} "
          f"({THRESHOLD_PROVENANCE})")
    print(f"representation: {fact_text_mode()}   "
          f"A {rep_a['nested']} nested / {rep_a['flat']} flat / "
          f"{rep_a['fell_back']} fell back   "
          f"B {rep_b['nested']} nested / {rep_b['flat']} flat / "
          f"{rep_b['fell_back']} fell back")
    if rep_a["fell_back"] or rep_b["fell_back"]:
        print("  !! some facts carry no triple and were embedded as sentences — "
              "the figures below are on a MIXED representation")
    if missing:
        print(f"  !! {len(missing)} segment(s) annotated by A only: {missing[:3]}")
    for b in blocks:
        if not b["n_segments"]:
            continue
        print(f"\n[{b['label']}]  segments {b['n_segments']}  "
              f"A {b['facts_a']}  B {b['facts_b']}")
        print(f"   covered  A by B {b['covered_a_by_b']}   B by A {b['covered_b_by_a']}"
              "     <- one-to-many, the figure that means what it says")
        print(f"   jaccard  {b['jaccard_one_to_one']} (one-to-one; LOWER BOUND, "
              "penalises the coarser annotator)")
        print(f"   spans    A inside a B span {b['a_spans_inside_a_b_span']}   "
              f"B inside an A span {b['b_spans_inside_an_a_span']}   "
              f"median {b['median_span_chars_a']} vs {b['median_span_chars_b']} chars")
        print(f"   counts   B-A {b['count_delta_b_minus_a']['shape']}")

    print(f"\nlowest first-pass coverage:")
    for r in sorted(rows, key=lambda r: r["covered_a"])[:args.worst]:
        print(f"   {r['segment_id']:<44} A={r['n_a']:>2} B={r['n_b']:>2} "
              f"covered={r['covered_a']:.2f} [{r['klass']}]"
              + ("  IAA" if r["iaa"] else ""))

    if args.no_write:
        print("\n  --no-write: nothing written")
        return
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "threshold": args.threshold,
        "threshold_provenance": THRESHOLD_PROVENANCE,
        "backend": backend_name(),
        # A similarity number means nothing without the representation it was
        # computed on (CLAUDE.md non-negotiable 4). Until 2026-08-27 every figure
        # here was silently computed on `natural_language`, because gold facts
        # carry nested `triples` and the encoder's triple mode fell through
        # without saying so (BUG-2026-08-27-B).
        "representation": fact_text_mode(),
        "representation_report": {"a": rep_a, "b": rep_b},
        "a": "data/gold/facts_hires",
        "b": "data/gold/annotator_b",
        "note": ("jaccard_one_to_one is a lower bound and is not the agreement of "
                 "record; covered_a_by_b / covered_b_by_a allow many-to-one and are "
                 "the figures to read. See the module docstring."),
        "summary": blocks,
        "per_segment": [{k: v for k, v in r.items()
                         if k not in ("span_len_a", "span_len_b")} for r in rows],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
