"""Validate the gold sample and report its statistics.

Reports the review status prominently: while any fact is still 'unreviewed'
the gold is a machine pre-annotation, not a human gold standard, and the
correctness numbers computed from it are not reportable.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "gold"

import sys
sys.path.insert(0, str(ROOT))
from src import gold_store as gs  # noqa: E402


def main() -> None:
    rep = gs.validate_all()
    sample = gs.load_sample()
    iaa = set(sample["iaa_subsample"]["segment_ids"])

    per_doc, per_kind, status, ops = {}, Counter(), Counter(), Counter()
    modes: Counter = Counter()
    n_cond = n_temp = n_logic = 0
    seg_counts = []
    for p in sorted((GOLD / "facts").glob("*.json")):
        payload = json.loads(p.read_text(encoding="utf-8"))
        doc = payload["doc_id"]
        n = 0
        for seg in payload["segments"]:
            k = len(seg["facts"])
            n += k
            seg_counts.append((seg["segment_id"], k))
            kind = "recital" if "recitals" in seg["section_path"] else "article"
            per_kind[kind] += k
            for f in seg["facts"]:
                status[f.get("status", "unreviewed")] += 1
                if f.get("status", "unreviewed") != "unreviewed":
                    modes[(f.get("review") or {}).get("mode", "unrecorded")] += 1
                n_cond += bool(f.get("condition"))
                n_temp += bool(f.get("temporal_context"))
                if f.get("logic_op"):
                    n_logic += 1
                    ops[f["logic_op"]] += 1
        per_doc[doc] = n

    print("=" * 68)
    print("EUR-Lex gold sample")
    print("=" * 68)
    print(f"documents          {rep.n_docs}")
    print(f"segments           {rep.n_segments}   (IAA subsample: {len(iaa)})")
    print(f"gold facts         {rep.n_facts}")
    print(f"  of which boilerplate  {rep.n_boilerplate}  (excluded from scoring by default)")
    print(f"facts per segment  mean {rep.n_facts/max(1,rep.n_segments):.1f}  "
          f"min {min(k for _, k in seg_counts)}  max {max(k for _, k in seg_counts)}")
    print(f"  recital segments {per_kind['recital']} facts | article segments {per_kind['article']} facts")
    print(f"slots filled       condition {n_cond} ({n_cond/rep.n_facts:.0%})  "
          f"temporal {n_temp} ({n_temp/rep.n_facts:.0%})  "
          f"logic {n_logic} ({n_logic/rep.n_facts:.0%}) {dict(ops)}")
    print()
    print("per document:")
    for d, n in sorted(per_doc.items()):
        print(f"  {d}  {n:3d}")
    print()
    print("densest segments:")
    for sid, k in sorted(seg_counts, key=lambda x: -x[1])[:5]:
        print(f"  {k:3d}  {sid}")
    print()
    print("review status:", dict(status))
    print("adjudicated by:", dict(modes) or "{}")
    unreviewed = status.get("unreviewed", 0)
    by_machine = modes.get("machine", 0)
    if unreviewed:
        print()
        print(f"!! {unreviewed}/{rep.n_facts} facts are still unreviewed.")
        print("!! This is a MACHINE PRE-ANNOTATION, not a human gold standard.")
        print("!! Correctness numbers computed against it are not reportable until")
        print("!! every fact has been adjudicated (status accepted/edited/rejected).")
    elif by_machine:
        # Not a gate — scoring runs fine on this. It is stated because the
        # pre-annotator and the adjudicator being the same model is something a
        # reader of the numbers has to know.
        print(f"   ({by_machine} by machine; confirm in review.html and re-apply "
              f"with --mode human to record the human sign-off)")
    print()
    print("validation:", "OK" if rep.ok else f"{len(rep.errors)} ERROR(S)")
    for e in rep.errors:
        print("  ", e)


if __name__ == "__main__":
    main()
