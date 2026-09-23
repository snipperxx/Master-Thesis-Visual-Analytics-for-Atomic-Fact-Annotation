"""
Phase-4 — Conflict-distribution shift between two guideline versions.

Inputs: two `data/conflicts/<doc>...json` files (or already-loaded dicts).
Output: a per-label diff + summary suitable for direct rendering in
        the UI's "Compare with v2" panel.

The proposal calls out three deliverables explicitly:
  * conflict-type distribution shift  (per-label deltas — this module)
  * total conflict count reduction    (totals — derived here too)
  * layer-1 filter rate change        (how many pairs Layer-1 caught
                                       before LLM call — derived here)

Bar-chart-friendly JSON: { "unit": "group", "labels": [...], "v1": [...],
                           "v2": [...], "delta": [...], "delta_pct": [...] }

Unit of analysis (changed 2026-08-09)
-------------------------------------
`method.tex` says "the group, not the pair, is the unit of analysis", but this
module only ever read the PAIR distribution (`label_counts`), so RQ1's primary
measure was computed over the wrong population — and over a one-to-many
alignment a fact appears in many pairs, so the pair distribution double-counts
exactly the facts granularity is about. `unit="group"` is now the default and
the unit is echoed in the output. Nothing of record moved: no v2 guideline
exists yet, so this module had never produced a reported number.

Two labels were also missing from `CONFLICT_LABELS`: `no_conflict` (an
arbitrator verdict, silently dropped from the distribution and the totals) and
`escalate` (not a conflict type — now reported separately as `unresolved`,
because a non-zero count means Layer-2 did not finish and every percentage in
the distribution is over the resolved remainder only).

Segment population (added 2026-09-23)
-------------------------------------
RQ1's Phi_v and Phi_w are computed over the segments every annotator returned
under both versions (method.tex sec:missing-segments; `src/segment_population`,
applied by `run_phase2 --segment-population` before alignment). A conflicts
file records the population in `params.segment_population`, and
`compare_conflict_files` refuses to diff two files computed over different
populations, including restricted against unrestricted. Two unrestricted files
still compare: that is the all-segments sensitivity check.
"""

from __future__ import annotations

import json
from pathlib import Path


# `no_conflict` was missing here until 2026-08-09, so every `no_conflict` verdict
# the arbitrator returned was silently dropped from the RQ1 distribution and from
# the totals — the shift was being computed over a subset of the outcomes.
CONFLICT_LABELS = ("contradiction", "granularity", "redundancy",
                   "no_conflict", "unlabeled")

# NOT a conflict type: it means Layer-2 never resolved the item. It is reported
# separately, and loudly, because a non-zero count makes every percentage in the
# distribution suspect — and because it silently hid the bug where nothing
# arbitrated the group layer at all (BUG-2026-08-09-B).
UNRESOLVED_LABEL = "escalate"

# The unit of analysis. method.tex: "The group, not the pair, is the unit of
# analysis" — so `group` is the default. `pair` is kept because the KG edges and
# the existing UI panels are pair-shaped, and because comparing the two is the
# honest way to show the choice matters.
UNITS = ("group", "pair")


def _load(path_or_doc) -> dict:
    if isinstance(path_or_doc, (str, Path)):
        return json.loads(Path(path_or_doc).read_text(encoding="utf-8"))
    return path_or_doc


def _raw_counts(doc: dict, unit: str) -> tuple[dict[str, int], bool]:
    """(counts, available). `available` is False when the file predates the unit."""
    if unit == "group":
        counts = doc.get("group_label_counts")
        if counts is None:
            groups = doc.get("groups")
            if not groups:
                return {}, False
            counts = {}
            for g in groups:
                lbl = g.get("conflict_label", "unlabeled")
                counts[lbl] = counts.get(lbl, 0) + 1
        return dict(counts), True
    return dict(doc.get("label_counts", {})), True


def _label_counts(doc: dict, unit: str = "group") -> dict[str, int]:
    """Conflict-type counts for `unit`, with every label present."""
    counts, _ = _raw_counts(doc, unit)
    counts = {k: v for k, v in counts.items() if not k.startswith("_")}
    for lbl in CONFLICT_LABELS:
        counts.setdefault(lbl, 0)
    return counts


def _unresolved(doc: dict, unit: str) -> int:
    counts, _ = _raw_counts(doc, unit)
    return int(counts.get(UNRESOLVED_LABEL, 0))


def _layer1_filter_rate(doc: dict, unit: str = "group") -> dict[str, float | int]:
    """How much did Layer-1 resolve without calling Layer-2?

    Denominator is whatever `unit` says: matched pairs, or arbitrable groups
    (a 1:n group is settled by its shape and an orphan has nothing to compare,
    so neither is ever billed to Layer-2 and neither belongs in the rate).
    """
    if unit == "group":
        groups = doc.get("groups") or []
        arbitrable = [g for g in groups if g.get("shape") == "one_to_one"]
        calls = int(doc.get("group_layer2_calls", 0))
        n = len(arbitrable)
        return {
            "n_arbitrable_groups": n,
            "n_layer2_calls": calls,
            "layer1_filter_rate": round(1 - calls / n, 3) if n else None,
        }
    pairs = doc.get("aligned_pairs", [])
    matched = [p for p in pairs if p.get("status") == "matched"]
    layer2_calls = int(doc.get("layer2_calls", 0))
    n_matched = len(matched) or 1
    return {
        "n_matched_pairs": len(matched),
        "n_layer2_calls": layer2_calls,
        "layer1_filter_rate": round(1 - layer2_calls / n_matched, 3),
    }


def _segment_population(doc: dict) -> dict | None:
    """The segment population a file was computed over; None = unrestricted."""
    return (doc.get("params") or {}).get("segment_population")


def check_populations(a: dict, b: dict, *, v1_label: str = "v1",
                      v2_label: str = "v2") -> dict:
    """Were two conflict files computed over the same segments?

    method.tex sec:missing-segments: Phi_v and Phi_w of an RQ1 comparison are
    computed over ONE population, the segments every annotator returned under
    both versions. A restricted file against an unrestricted one, or two
    restricted under different populations, differ in which segments they
    count, and a version that loses segments would move Phi through that alone
    (every triple of a segment one annotator did not return is an orphan).
    Two unrestricted files are comparable: that is the all-segments
    sensitivity figure, never the primary one.
    """
    pa, pb = _segment_population(a), _segment_population(b)
    out = {
        "restricted": bool(pa and pb),
        "match": True,
        "hash": {v1_label: (pa or {}).get("hash"), v2_label: (pb or {}).get("hash")},
        "error": None,
        "note": None,
    }
    if (pa is None) != (pb is None):
        restricted = v1_label if pa else v2_label
        out.update(match=False, error=(
            f"segment populations differ: {restricted} was computed over a "
            f"segment population and the other over all segments. A restricted "
            f"figure is only comparable with one computed over the same population "
            f"(method.tex sec:missing-segments); re-run Phase-2 on both with the "
            f"same --segment-population, or on neither for the sensitivity check"))
        return out
    if pa is None:
        out["note"] = ("both files are unrestricted (all segments): this is the "
                       "sensitivity figure of sec:missing-segments, not the "
                       "primary RQ1 figure")
        return out
    if pa.get("hash") != pb.get("hash"):
        out.update(match=False, error=(
            f"segment populations differ: {v1_label} {str(pa.get('hash'))[:12]} vs "
            f"{v2_label} {str(pb.get('hash'))[:12]}; both versions of an RQ1 "
            f"comparison must be computed over one population"))
        return out
    doc_a, doc_b = pa.get("doc") or {}, pb.get("doc") or {}
    out.update({
        "path": pa.get("path"),
        "versions": pa.get("versions"),
        "file_versions": {v1_label: pa.get("version"), v2_label: pb.get("version")},
        "n_segments": pa.get("n_segments"),
        "n_segments_universe": pa.get("n_segments_universe"),
        # Corpus-level: segments each annotator did not return, per version.
        # Reported beside the shift, because a revision could raise agreement
        # simply by making the annotators give up on the hardest segments.
        "excluded_counts": pa.get("excluded_counts"),
        # This document only.
        "doc_n_segments": doc_a.get("n_segments"),
        "doc_n_kept": doc_a.get("n_kept"),
        "doc_excluded_sections": doc_a.get("excluded_sections"),
        "doc_excluded": doc_a.get("excluded"),
        "facts_removed": {v1_label: doc_a.get("facts_removed"),
                          v2_label: doc_b.get("facts_removed")},
    })
    return out


def compare_conflict_files(
    v1, v2, *,
    v1_label: str = "v1",
    v2_label: str = "v2",
    unit: str = "group",
) -> dict:
    """Diff two Phase-2 conflict files. v1, v2 may be dicts or paths.

    `unit` selects the population the distribution is over — "group" (the unit
    of analysis, default) or "pair". The unit is echoed in the result, because a
    shift measured over pairs and one measured over groups are different claims
    and a bare bar chart cannot tell them apart.

    REFUSES (available=False, with `error` and `note`) when the two files were
    computed over different segment populations; see `check_populations`. The
    population block, with the excluded counts, is in `population` either way.
    """
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")
    a = _load(v1)
    b = _load(v2)

    population = check_populations(a, b, v1_label=v1_label, v2_label=v2_label)
    if not population["match"]:
        return {
            "unit": unit,
            "v1_label": v1_label, "v2_label": v2_label,
            "available": False,
            "error": population["error"],
            # `note` as well: it is the field the UI panels print instead of bars.
            "note": population["error"],
            "population": population,
        }

    # A file written before the group layer has no group counts. Say so instead
    # of silently reporting zeros, which would read as "no conflicts of any kind".
    avail_a = _raw_counts(a, unit)[1]
    avail_b = _raw_counts(b, unit)[1]
    if not (avail_a and avail_b):
        missing = [lbl for lbl, ok in ((v1_label, avail_a), (v2_label, avail_b)) if not ok]
        return {
            "unit": unit,
            "v1_label": v1_label, "v2_label": v2_label,
            "available": False,
            "error": (f"no {unit}-level counts in: {', '.join(missing)} — that file "
                      f"predates the group layer; re-run Phase-2 on it, or compare "
                      f"with unit='pair'"),
        }

    c_a = _label_counts(a, unit)
    c_b = _label_counts(b, unit)

    labels = list(CONFLICT_LABELS)
    v1_counts = [c_a[l] for l in labels]
    v2_counts = [c_b[l] for l in labels]
    delta = [b_i - a_i for a_i, b_i in zip(v1_counts, v2_counts)]

    def _pct(a_i: int, b_i: int) -> float:
        return round((b_i - a_i) / a_i * 100, 1) if a_i else None  # type: ignore

    delta_pct = [_pct(a_i, b_i) for a_i, b_i in zip(v1_counts, v2_counts)]

    total_a = sum(v1_counts)
    total_b = sum(v2_counts)

    unresolved_a = _unresolved(a, unit)
    unresolved_b = _unresolved(b, unit)

    return {
        "unit": unit,
        "available": True,
        "v1_label": v1_label,
        "v2_label": v2_label,
        "labels": labels,
        "v1": v1_counts,
        "v2": v2_counts,
        "delta": delta,
        "delta_pct": delta_pct,
        "totals": {
            "v1": total_a,
            "v2": total_b,
            "delta": total_b - total_a,
            "delta_pct": _pct(total_a, total_b),
        },
        # Items Layer-2 never resolved. Excluded from the distribution above
        # because `escalate` is not a conflict type; surfaced here because a
        # non-zero value invalidates every percentage in it.
        "unresolved": {
            "label": UNRESOLVED_LABEL,
            "v1": unresolved_a,
            "v2": unresolved_b,
            "warning": ("Layer-2 left items unresolved; the distribution above is "
                        "over the resolved remainder only"
                        if (unresolved_a or unresolved_b) else None),
        },
        "layer1_rate": {
            "v1": _layer1_filter_rate(a, unit),
            "v2": _layer1_filter_rate(b, unit),
        },
        # Which segments the two distributions are over, and what was excluded.
        "population": population,
    }


if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser(description="Compare two Phase-2 conflict files.")
    p.add_argument("--v1", required=True, type=Path)
    p.add_argument("--v2", required=True, type=Path)
    p.add_argument("--unit", choices=UNITS, default="group",
                   help="population the distribution is over (default: group, "
                        "the unit of analysis per method.tex)")
    args = p.parse_args()
    out = compare_conflict_files(args.v1, args.v2, unit=args.unit)
    json.dump(out, sys.stdout, indent=2, ensure_ascii=False)
