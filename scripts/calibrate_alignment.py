"""Sweep the alignment threshold on DEV documents and report group shapes.

`tau` was fixed at 0.78 under the one-to-one Hungarian assignment. One-to-many
linking changes what the number means: under one-to-one, tau only decided
whether the single best partner survived; under one-to-many it decides how many
partners a fact collects, so the same value produces a different structure. It
has to be re-read on dev material before the run of record.

What to look at:

    fan-out         partners per fact. A median above ~2 means tau is too low —
                    EUR-Lex facts share heavy boilerplate vocabulary, so a
                    permissive threshold links everything to everything and
                    every fact looks like a granularity split.
    one_to_many %   the granularity candidate rate. It should rise as tau falls
                    and it is the quantity that can be inflated by accident.
    orphan %        coverage: facts the other annotator came nowhere near.

**Dev only.** `--doc-list` defaults to the dev split and the script refuses to
run on any document in the locked eval list (discipline rule 1).

    python -m scripts.calibrate_alignment
    python -m scripts.calibrate_alignment --taus 0.78,0.85,0.90 --n-docs 12
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_phase2 import discover_annotators  # noqa: E402
from src.alignment import (SLOT_GATES, align_all_pairs, backend_name,  # noqa: E402
                           build_groups)

DEV_LIST = ROOT / "data" / "doc_lists" / "eurlex_dev.json"
EVAL_LIST = ROOT / "data" / "doc_lists" / "eurlex_eval.json"


def _doc_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    docs = payload["docs"] if isinstance(payload, dict) else payload
    return [d["doc_id"] if isinstance(d, dict) else str(d) for d in docs]


def sweep(doc_ids: list[str], taus: list[float], facts_root: Path,
          slot_gates: dict | None = None) -> list[dict]:
    rows = []
    for tau in taus:
        shapes: Counter = Counter()
        fan_outs: list[int] = []
        n_pairs = 0
        n_docs = 0
        for doc_id in doc_ids:
            fpa = discover_annotators(facts_root, doc_id)
            if len(fpa) < 2:
                continue
            n_docs += 1
            pairs = align_all_pairs(fpa, threshold=tau, one_to_many=True,
                                    slot_gates=slot_gates)
            n_pairs += sum(1 for p in pairs if p.status == "matched")
            for g in build_groups(pairs):
                shapes[g.shape] += 1
                fan_outs.append(g.fan_out)
        total = sum(shapes.values()) or 1
        matched_fan = [f for f in fan_outs if f]
        rows.append({
            "tau": tau, "n_docs": n_docs, "n_matched_pairs": n_pairs,
            "n_groups": total,
            "orphan_pct": round(100 * shapes["orphan"] / total, 1),
            "one_to_one_pct": round(100 * shapes["one_to_one"] / total, 1),
            "one_to_many_pct": round(100 * shapes["one_to_many"] / total, 1),
            "fan_out_median": statistics.median(matched_fan) if matched_fan else 0,
            "fan_out_p90": (statistics.quantiles(matched_fan, n=10)[8]
                            if len(matched_fan) >= 10 else max(matched_fan or [0])),
            "fan_out_max": max(fan_outs or [0]),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc-list", type=Path, default=DEV_LIST)
    ap.add_argument("--facts-root", type=Path, default=ROOT / "data" / "facts")
    ap.add_argument("--taus", default="0.70,0.78,0.82,0.86,0.90")
    ap.add_argument("--n-docs", type=int, default=10)
    ap.add_argument("--slot-gates", action="store_true",
                    help="also require the frozen per-slot gates "
                         "(subject/object >= 0.70, predicate >= 0.45)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    doc_ids = _doc_ids(args.doc_list)[:args.n_docs]
    locked = set(_doc_ids(EVAL_LIST))
    leaked = sorted(set(doc_ids) & locked)
    if leaked:
        raise SystemExit(
            "refusing to run: these are in the LOCKED EVAL list and must not be "
            f"tuned against (discipline rule 1): {', '.join(leaked)}")

    taus = [float(t) for t in args.taus.split(",")]
    print(f"backend={backend_name()}  docs={len(doc_ids)} (dev)  taus={taus}\n")
    gates = SLOT_GATES if args.slot_gates else None
    print(f"slot gates: {gates or 'off (whole-string cosine only)'}\n")
    rows = sweep(doc_ids, taus, args.facts_root, slot_gates=gates)

    print(f"{'tau':>5} {'docs':>5} {'pairs':>7} {'groups':>7} {'orphan%':>8} "
          f"{'1:1%':>7} {'1:n%':>7} {'fan med':>8} {'p90':>5} {'max':>5}")
    for r in rows:
        print(f"{r['tau']:>5.2f} {r['n_docs']:>5} {r['n_matched_pairs']:>7} "
              f"{r['n_groups']:>7} {r['orphan_pct']:>8} {r['one_to_one_pct']:>7} "
              f"{r['one_to_many_pct']:>7} {r['fan_out_median']:>8} "
              f"{r['fan_out_p90']:>5} {r['fan_out_max']:>5}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"backend": backend_name(), "split": "dev", "doc_ids": doc_ids,
             "slot_gates": gates,
             "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
