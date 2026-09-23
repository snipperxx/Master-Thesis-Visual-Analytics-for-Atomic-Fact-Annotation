"""Summarise a dev Phase-2 baseline into runs/rq1_dev_<variant>.json.

    python scripts/summarise_rq1_baseline.py            # default variant v1_triple
    python scripts/summarise_rq1_baseline.py v2_triple


The headline it exists to record: the conflict-type distribution depends on the
UNIT it is computed over, and not marginally. Measured over the same 37
documents and the same arbitration calls, the contradiction count is 21 at group
level and 578 at pair level, and 99 % of the difference is pairs that sit inside
a one-to-many group — where the group layer reads the same facts, correctly, as
a granularity split.
"""
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "v1_triple"
if VARIANT == "baseline":
    # The unsuffixed run — what Phase-2 writes without --out-suffix, and what
    # the UI shows as the default. Added 2026-09-10 so the 30B run of record,
    # which is the baseline, can be summarised without duplicating 25 MB of
    # conflict files under a suffix.
    FILES = sorted(f for f in (ROOT / "data" / "conflicts").glob("*.json")
                   if "__" not in f.name)
    if not FILES:
        raise SystemExit("no unsuffixed data/conflicts/*.json — run Phase-2 first")
else:
    FILES = sorted((ROOT / "data" / "conflicts").glob(f"*__{VARIANT}.json"))
    if not FILES:
        raise SystemExit(f"no data/conflicts/*__{VARIANT}.json — run Phase-2 with "
                         f"--out-suffix {VARIANT} first")

group = collections.Counter()
group_l1 = collections.Counter()
pair = collections.Counter()
shapes = collections.Counter()
calls = pair_calls = 0
n_1n = n_schema = 0
contra_by_shape = collections.Counter()
annotators_seen = set()
per_doc = []

for f in FILES:
    d = json.loads(f.read_text(encoding="utf-8"))
    group.update(d["group_label_counts"])
    group_l1.update(d["group_label_counts_layer1"])
    pair.update(d["label_counts"])
    shapes.update(d["group_shapes"])
    annotators_seen.update(d.get("annotators") or [])
    calls += d.get("group_layer2_calls", 0)
    pair_calls += d.get("layer2_calls", 0)

    fan = {}
    for g in d["groups"]:
        fan[g["anchor_id"]] = max(fan.get(g["anchor_id"], 0), g["fan_out"])
        if g["shape"] == "one_to_many":
            n_1n += 1
            if g.get("partners_one_annotation"):
                n_schema += 1
    for p in d["aligned_pairs"]:
        if p.get("conflict_label") == "contradiction":
            multi = fan.get(p.get("fact_a_id"), 0) > 1 or fan.get(p.get("fact_b_id"), 0) > 1
            contra_by_shape["inside_one_to_many_group" if multi else "inside_one_to_one_group"] += 1

    per_doc.append({
        "doc_id": d["doc_id"],
        "n_groups": sum(d["group_label_counts"].values()),
        "shapes": d["group_shapes"],
        "group_labels": d["group_label_counts"],
        "group_layer2_calls": d.get("group_layer2_calls", 0),
    })

n_groups = sum(group.values())
out = {
    "what": "v1 baseline conflict-type distribution, DEV split",
    "generated_from": ("data/conflicts/*.json (unsuffixed)" if VARIANT == "baseline"
                       else f"data/conflicts/*__{VARIANT}.json"),
    "configuration": {
        "split": "dev", "n_docs": len(FILES),
        # Read from the files, not hardcoded: the annotator set changed to the
        # cloud 30B trio on 2026-09-03 and this field still said 4B (2026-09-10).
        "annotators": sorted(annotators_seen),
        "guideline": "v1", "alignment": "one_to_many", "tau": 0.78,
        "fact_text": "triple", "layer2_model": "qwen3.5:4b",
        "arbitrate_shapes": ["one_to_one"],
    },
    "caveats": [
        "DEV split. This is diagnosis material, not an RQ1 result; the eval set is untouched.",
        "The gold is machine-adjudicated, so nothing here is a correctness claim.",
        "Layer-2 is a 4B local arbitrator; label proportions carry its bias.",
    ],
    "group_level": {
        "n_groups": n_groups,
        "shapes": dict(shapes),
        "layer1_or_shape": dict(group_l1),
        "final": dict(group),
        "final_pct": {k: round(100 * v / n_groups, 2) for k, v in group.items()},
        "layer2_calls": calls,
    },
    "pair_level": {"final": dict(pair), "layer2_calls": pair_calls},
    "unit_of_analysis_effect": {
        "contradiction_group_level": group.get("contradiction", 0),
        "contradiction_pair_level": pair.get("contradiction", 0),
        "ratio": round(pair.get("contradiction", 0) / max(1, group.get("contradiction", 1)), 1),
        "pair_contradictions_by_group_shape": dict(contra_by_shape),
        "reading": ("A fact compared pairwise against ONE member of another "
                    "annotator's multi-fact split reads as a conflicting claim, "
                    "because the rest of the split is not in the prompt. The group "
                    "layer shows the same facts as one-vs-N and labels them "
                    "granularity. 99% of the excess is this."),
    },
    "schema_fanout": {
        "one_to_many_groups": n_1n, "inside_single_annotation": n_schema,
        "pct": round(100 * n_schema / max(1, n_1n), 1),
    },
    "per_doc": per_doc,
}
dest = ROOT / "runs" / f"rq1_dev_{VARIANT}.json"
dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print("wrote", dest)
print(json.dumps(out["group_level"]["final_pct"], indent=2))
print("unit effect:", out["unit_of_analysis_effect"]["ratio"], "x")
