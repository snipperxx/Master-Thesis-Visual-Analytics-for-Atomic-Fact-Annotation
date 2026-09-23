"""Agreement between two adjudication passes over the same gold.

    python -m scripts.compare_adjudications data/gold/decisions_gemini
    python -m scripts.compare_adjudications DIR_B --a data/gold/decisions --out runs/x.json

The current gold was pre-annotated and adjudicated by the same model, which is
why discipline rule 6 refuses correctness numbers computed against it. A pass by
a model from a different family removes that circularity, and comparing the two
gives an **adjudicator-agreement** figure the Discussion otherwise has nothing
for. It is a machine--machine agreement and must be reported as one: it says
whether the annotation protocol is applied reproducibly, not whether the gold is
right.

Reported: raw agreement, Cohen's kappa over {accepted, edited, rejected}, the
confusion matrix, and the disagreements themselves --- which are the useful part,
because a fact two independent adjudicators split on is a fact the *guideline*
failed to determine, and that is evidence for the diagnosis.

One caveat is computed and printed rather than left implicit: both passes share
the decision log as a codebook, and its entries name specific segments. Agreement
on those segments is partly guaranteed by construction, so it is reported
separately from agreement on the rest.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import product
from pathlib import Path

from src.gold_store import GOLD, GOLD_FACTS

STATUSES = ("accepted", "edited", "rejected")


def load_pass(d: Path) -> tuple[dict, str]:
    if not d.exists():
        raise SystemExit(f"no such directory: {d}")
    decisions, reviewer = {}, None
    for p in sorted(d.glob("*.json")):
        payload = json.loads(p.read_text(encoding="utf-8"))
        reviewer = reviewer or payload.get("reviewer")
        for gid, v in payload["decisions"].items():
            decisions[gid] = v
    if not decisions:
        raise SystemExit(f"no decisions found under {d}")
    return decisions, reviewer or d.name


def codebook_segments() -> set[str]:
    log = json.loads((GOLD / "decision_log.json").read_text(encoding="utf-8"))
    out: set[str] = set()
    for e in log.get("entries", []):
        out.update(e.get("where") or [])
    return out


def kappa(pairs: list[tuple[str, str]]) -> float:
    """Cohen's kappa. Returns 1.0 when both passes are constant and identical."""
    n = len(pairs)
    if not n:
        return float("nan")
    obs = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    exp = sum((ca[s] / n) * (cb[s] / n) for s in set(ca) | set(cb))
    if exp == 1.0:
        return 1.0 if obs == 1.0 else 0.0
    return (obs - exp) / (1 - exp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("b", type=Path, help="the second pass (e.g. data/gold/decisions_gemini)")
    ap.add_argument("--a", type=Path, default=GOLD / "decisions",
                    help="the first pass (default: data/gold/decisions)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    A, name_a = load_pass(args.a)
    B, name_b = load_pass(args.b)
    shared = sorted(set(A) & set(B))
    if not shared:
        raise SystemExit("the two passes share no gold_id")

    seg_of = {}
    for p in GOLD_FACTS.glob("*.json"):
        for seg in json.loads(p.read_text(encoding="utf-8"))["segments"]:
            for f in seg["facts"]:
                seg_of[f["gold_id"]] = seg["segment_id"]

    cb = codebook_segments()
    pairs = [(A[g]["status"], B[g]["status"]) for g in shared]
    in_cb = [g for g in shared if seg_of.get(g) in cb]
    out_cb = [g for g in shared if seg_of.get(g) not in cb]

    agree = sum(1 for a, b in pairs if a == b)
    conf = Counter(pairs)
    disagreements = [
        {"gold_id": g, "segment_id": seg_of.get(g),
         name_a: A[g]["status"], name_b: B[g]["status"],
         f"note_{name_b}": (B[g].get("note") or "")[:180]}
        for g in shared if A[g]["status"] != B[g]["status"]
    ]

    def sub(ids):
        ps = [(A[g]["status"], B[g]["status"]) for g in ids]
        if not ps:
            return None
        return {"n": len(ps),
                "agreement": round(sum(1 for a, b in ps if a == b) / len(ps), 4),
                "kappa": round(kappa(ps), 4)}

    print(f"adjudicator A : {name_a}   ({len(A)} decisions)")
    print(f"adjudicator B : {name_b}   ({len(B)} decisions)")
    print(f"compared      : {len(shared)} facts\n")
    print(f"raw agreement : {agree}/{len(shared)} = {agree/len(shared):.1%}")
    print(f"Cohen's kappa : {kappa(pairs):.3f}\n")

    header = "A \\ B"
    print(f"{header:<12}" + "".join(f"{s:>11}" for s in STATUSES))
    for a in STATUSES:
        print(f"{a:<12}" + "".join(f"{conf.get((a, b), 0):>11}" for b in STATUSES))

    print("\nsplit by whether the shared codebook names the segment:")
    print(f"  named by a decision-log entry : {sub(in_cb)}")
    print(f"  not named (independent)       : {sub(out_cb)}")
    print("  (agreement on named segments is partly guaranteed: both passes were "
          "given the\n   convention that applies there. The second row is the "
          "honest number.)")

    print(f"\ndisagreements: {len(disagreements)}")
    for d in disagreements[:10]:
        print(f"  {d['gold_id']}")
        print(f"      {name_a}={d[name_a]}  {name_b}={d[name_b]}  {d[f'note_{name_b}']}")
    if len(disagreements) > 10:
        print(f"  ... and {len(disagreements) - 10} more")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "what": "machine--machine adjudicator agreement over the gold",
            "not": ("This is NOT inter-annotator agreement between humans and NOT "
                    "evidence that the gold is correct. It measures whether the "
                    "annotation protocol is applied reproducibly across models."),
            "adjudicator_a": name_a, "adjudicator_b": name_b,
            "n_compared": len(shared),
            "raw_agreement": round(agree / len(shared), 4),
            "cohens_kappa": round(kappa(pairs), 4),
            "confusion": {f"{a}->{b}": conf.get((a, b), 0) for a, b in product(STATUSES, STATUSES)},
            "by_codebook_coverage": {"named_in_decision_log": sub(in_cb),
                                     "not_named": sub(out_cb)},
            "disagreements": disagreements,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
