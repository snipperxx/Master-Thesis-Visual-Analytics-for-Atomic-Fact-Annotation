"""
Annotator-vs-annotator comparison views (poster Fig. 2).

Two products, both consumed by the UI's Heatmap / Stats tabs:

  * similarity_matrix(facts_a, facts_b)
        Full cosine matrix between two annotators' fact lists, plus the
        Hungarian assignment so the UI can outline matched cells.
        This is the "semantic similarity heatmap" from Schmidt et al.'s
        poster (Fig. 2 left).

  * iaa_summary(conflicts_doc)
        Per annotator-pair agreement numbers computed from an existing
        Phase-2 conflicts file (no re-embedding): matched counts,
        Jaccard IAA (poster's metric: |matched| / |union|), mean cosine,
        and per-pair conflict-label counts.

Backend note: embeddings go through src.alignment.encode_facts_joint, so
SBERT is used on the user's machine and the char-trigram TF-IDF fallback
keeps this module testable in sandboxes without torch.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from src.alignment import align_two, encode_facts_joint, _resolve_backend


def _fact_brief(f: dict) -> dict:
    loc = f.get("source_locator") or {}
    return {
        "fact_id": f.get("fact_id"),
        "text": (f.get("natural_language") or "").strip()
                or f"{f.get('subject','')} {f.get('predicate','')} {f.get('object','')}".strip(),
        "subject": f.get("subject"),
        "predicate": f.get("predicate"),
        "object": f.get("object"),
        "section_path": loc.get("section_path"),
        "quote": loc.get("quote"),
    }


def similarity_matrix(
    facts_a: list[dict],
    facts_b: list[dict],
    *,
    annotator_a: str,
    annotator_b: str,
    threshold: float = 0.78,
    aligned_pairs: list[dict] | None = None,
) -> dict[str, Any]:
    """Full cosine matrix (rows = annotator_a facts, cols = annotator_b facts).

    If `aligned_pairs` (from a Phase-2 conflicts file) is given, matched cells
    and conflict labels are taken from there so the heatmap agrees with the
    persisted pipeline output. Otherwise a fresh Hungarian assignment at
    `threshold` is computed on the fly.
    """
    emb_a, emb_b = encode_facts_joint(facts_a, facts_b)
    if len(facts_a) and len(facts_b):
        matrix = (emb_a @ emb_b.T).astype(float)
    else:
        matrix = np.zeros((len(facts_a), len(facts_b)), dtype=float)

    idx_a = {f.get("fact_id"): i for i, f in enumerate(facts_a)}
    idx_b = {f.get("fact_id"): i for i, f in enumerate(facts_b)}

    matched: list[dict] = []
    if aligned_pairs is not None:
        for p in aligned_pairs:
            if p.get("annotator_a") == annotator_a and p.get("annotator_b") == annotator_b:
                a_id, b_id = p.get("fact_a_id"), p.get("fact_b_id")
            elif p.get("annotator_a") == annotator_b and p.get("annotator_b") == annotator_a:
                a_id, b_id = p.get("fact_b_id"), p.get("fact_a_id")
            else:
                continue
            if p.get("status") != "matched" or a_id not in idx_a or b_id not in idx_b:
                continue
            matched.append({
                "ia": idx_a[a_id], "ib": idx_b[b_id],
                "fact_a_id": a_id, "fact_b_id": b_id,
                "cosine": p.get("cosine"),
                "conflict_label": p.get("conflict_label", "unlabeled"),
                "layer1_reason": p.get("layer1_reason"),
                "layer2_reason": p.get("layer2_reason"),
            })
    else:
        for p in align_two(facts_a, facts_b,
                           annotator_a=annotator_a, annotator_b=annotator_b,
                           threshold=threshold, emb_a=emb_a, emb_b=emb_b):
            if p.status != "matched":
                continue
            a_id, b_id = p.fact_a["fact_id"], p.fact_b["fact_id"]
            matched.append({
                "ia": idx_a[a_id], "ib": idx_b[b_id],
                "fact_a_id": a_id, "fact_b_id": b_id,
                "cosine": round(p.cosine, 4),
                "conflict_label": p.conflict_label,
                "layer1_reason": p.layer1_reason,
                "layer2_reason": p.layer2_reason,
            })

    return {
        "annotator_a": annotator_a,
        "annotator_b": annotator_b,
        "backend": _resolve_backend(),
        "threshold": threshold,
        "facts_a": [_fact_brief(f) for f in facts_a],
        "facts_b": [_fact_brief(f) for f in facts_b],
        "matrix": [[round(float(v), 3) for v in row] for row in matrix],
        "matched": matched,
    }


def iaa_summary(conflicts_doc: dict) -> dict[str, Any]:
    """Annotator-pair agreement table from a Phase-2 conflicts file.

    Jaccard IAA follows the poster: |matched pairs| / |facts_a ∪ facts_b|
    where the union size is n_a + n_b - n_matched (one-to-one matching).
    """
    fpa: dict[str, list] = conflicts_doc.get("facts_per_annotator", {})
    annotators = conflicts_doc.get("annotators") or sorted(fpa.keys())
    counts = {a: len(fpa.get(a, [])) for a in annotators}

    by_pair: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"n_matched": 0, "cos_sum": 0.0, "labels": defaultdict(int)})
    for p in conflicts_doc.get("aligned_pairs", []):
        key = (p.get("annotator_a"), p.get("annotator_b"))
        slot = by_pair[key]
        if p.get("status") == "matched":
            slot["n_matched"] += 1
            slot["cos_sum"] += float(p.get("cosine") or 0.0)
            slot["labels"][p.get("conflict_label", "unlabeled")] += 1

    pairs_out = []
    for i in range(len(annotators)):
        for j in range(i + 1, len(annotators)):
            a, b = annotators[i], annotators[j]
            slot = by_pair.get((a, b)) or by_pair.get((b, a)) \
                or {"n_matched": 0, "cos_sum": 0.0, "labels": {}}
            n_m = slot["n_matched"]
            union = counts.get(a, 0) + counts.get(b, 0) - n_m
            pairs_out.append({
                "a": a, "b": b,
                "n_a": counts.get(a, 0), "n_b": counts.get(b, 0),
                "n_matched": n_m,
                "jaccard": round(n_m / union, 3) if union > 0 else None,
                "mean_cosine": round(slot["cos_sum"] / n_m, 3) if n_m else None,
                "labels": dict(slot["labels"]),
            })

    return {"annotators": annotators, "counts": counts, "pairs": pairs_out}
