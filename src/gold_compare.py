"""
Gold-standard triple matching: the RQ2 matchers.

Corpus-agnostic. It began as the WebNLG comparison (hence the examples below);
the EUR-Lex RQ2 driver `src/gold_eurlex_eval.py` calls it once per gold segment.
It also builds a gold KG for side-by-side display.

Unit of analysis (both modes)
-----------------------------
* **Recall unit = gold triple.** A nested gold fact of N triples contributes N
  units; the driver flattens `triples` before calling in. Triples that are equal
  after normalisation (`triple_key`) are one unit, so a triple repeated across
  two facts of one segment is counted once.
* **Precision unit = predicted triple.** Model output is already one record per
  triple. Records equal after normalisation are one unit, so emitting the same
  triple twice is neither credited nor charged twice.

This is what makes content correctness packaging-invariant (CLAUDE.md
non-negotiable 5): three facts of one triple and one fact of three triples are
the same triple set.

Assignment: many-to-many coverage (design of record, decided 2026-09-23)
------------------------------------------------------------------------
Every (gold, prediction) pair that clears the match criterion is a **link**;
nothing is exclusive. A gold unit is *covered* if it has at least one link; a
prediction is *supported* if it has at least one link.
    recall    = covered gold units / gold units
    precision = supported predictions / predictions
So a prediction that merges three gold triples, and clears the gates against
each, covers all three and is not penalised for merging; a model that splits as
the gold does is credited identically; two predictions of one gold triple are
both supported while the gold triple counts once in recall
(`doc/methodology_v2.tex` "The resolution: annotate content, score form
separately"; `thesis/chapters/method.tex` "Correctness (RQ2)").

The greedy one-to-one assignment used until 2026-09-23 survives ONLY as the
explicit opt-in ablation `assignment="one_to_one"`. It penalises exactly the
packaging difference the gold was built to be invariant to: a coarse prediction
can absorb one gold triple and its other gold triples count as missed.

**Strict** matching is string equality after normalisation, which is an
equivalence relation on the deduplicated keys: each gold key can link to at most
the one identical predicted key. Many-to-many and one-to-one therefore coincide
for strict, and `compare` takes no assignment argument.

Strict (`compare`): a prediction links to a gold triple only if subject,
predicate AND object all agree after normalisation.

Normalisation is deliberately "strict but fair":
  * terms: lowercase, underscores->spaces, drop surrounding quotes/punct,
    strip leading articles (a/an/the).
  * predicates: split camelCase (gold uses `cityServed`), lowercase, strip
    leading modals/auxiliaries so "serves"/"shall serve" align, collapse space.

So `Aarhus_Airport | cityServed | "Aarhus, Denmark"` matches an extracted
("Aarhus Airport", "city served", "Aarhus Denmark") but NOT
("Aarhus Airport", "serves the city of", "Aarhus, Denmark") — the predicate
surface differs. That gap is exactly what the comparison is meant to expose.
"""

from __future__ import annotations

import re
from collections import Counter

from src import compare_core as cc

# Normalisation moved to src/compare_core.py on 2026-08-27 so that alignment and
# gold scoring share one definition instead of two private copies that had begun
# to drift. Re-exported here because callers and tests import them from this
# module; the implementations are byte-identical to what they replaced.
_PRED_STOP_LEAD = cc._PRED_STOP_LEAD
_ARTICLES = cc._ARTICLES
_CAMEL_RE = cc._CAMEL_RE
normalize_term = cc.normalize_term
normalize_predicate = cc.normalize_predicate
triple_key = cc.triple_key


def _gold_spo(g: dict) -> tuple[str, str, str]:
    return (g.get("subject", ""), g.get("predicate", ""), g.get("object", ""))


def compare(gold_triples: list[dict], extracted_facts: list[dict]) -> dict:
    """Strict triple match. Returns metrics + matched/missing/spurious + fact ids.

    Assignment-invariant (module docstring): a gold key links only to the
    identical predicted key, so covered gold = supported predictions = links.
    The three counts are emitted anyway so that a driver can aggregate strict
    and soft through one code path without special-casing either."""
    gold_by_key: dict[tuple, tuple] = {}
    for g in gold_triples or []:
        s, p, o = _gold_spo(g)
        if not normalize_term(s) or not normalize_term(o):
            continue
        gold_by_key.setdefault(triple_key(s, p, o), (s, p, o))

    ex_by_key: dict[tuple, dict] = {}
    for f in extracted_facts or []:
        s, p, o = f.get("subject", ""), f.get("predicate", ""), f.get("object", "")
        k = triple_key(s, p, o)
        if not k[0] or not k[2]:
            continue
        slot = ex_by_key.setdefault(k, {"spo": (s, p, o), "fact_ids": set(), "annotators": set()})
        if f.get("fact_id"):
            slot["fact_ids"].add(f["fact_id"])
        if f.get("_annotator"):
            slot["annotators"].add(f["_annotator"])

    gk, ek = set(gold_by_key), set(ex_by_key)
    matched_k, missing_k, spurious_k = gk & ek, gk - ek, ek - gk

    n_gold, n_ex, n_match = len(gk), len(ek), len(matched_k)
    precision = n_match / n_ex if n_ex else 0.0
    recall = n_match / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    matched = [{"triple": list(gold_by_key[k]),
                "fact_ids": sorted(ex_by_key[k]["fact_ids"])} for k in sorted(matched_k)]
    missing = [{"triple": list(gold_by_key[k])} for k in sorted(missing_k)]
    spurious = [{"triple": list(ex_by_key[k]["spo"]),
                 "fact_ids": sorted(ex_by_key[k]["fact_ids"])} for k in sorted(spurious_k)]
    matched_fact_ids = sorted({fid for k in matched_k for fid in ex_by_key[k]["fact_ids"]})
    spurious_fact_ids = sorted({fid for k in spurious_k for fid in ex_by_key[k]["fact_ids"]})

    return {
        "metrics": {
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "n_gold": n_gold, "n_extracted": n_ex, "n_matched": n_match,
            "n_gold_covered": n_match, "n_pred_supported": n_match,
            "n_links": n_match,
            "n_missing": len(missing_k), "n_spurious": len(spurious_k),
        },
        "matched": matched, "missing": missing, "spurious": spurious,
        "matched_fact_ids": matched_fact_ids, "spurious_fact_ids": spurious_fact_ids,
    }


def _node_id(label: str) -> str:
    return "g_" + (re.sub(r"\W+", "_", normalize_term(label))[:48] or "x")


def gold_graph(gold_triples: list[dict]) -> dict:
    """Cytoscape elements for the gold KG (mirrors kg_build's node/edge shape)."""
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    for g in gold_triples or []:
        s, p, o = _gold_spo(g)
        if not s or not o:
            continue
        si, oi = _node_id(s), _node_id(o)
        nodes.setdefault(si, {"data": {"id": si, "label": s[:34], "full_label": s,
                                       "gold": True, "conflict_label": "unlabeled"}})
        nodes.setdefault(oi, {"data": {"id": oi, "label": o[:34], "full_label": o,
                                       "gold": True, "conflict_label": "unlabeled"}})
        edges.append({"data": {"id": f"ge_{si}__{normalize_predicate(p)[:20]}__{oi}".replace(" ", "_"),
                               "source": si, "target": oi, "label": p, "gold": True,
                               "conflict_label": "unlabeled"}})
    return {"nodes": list(nodes.values()), "edges": edges,
            "summary": {"n_nodes": len(nodes), "n_edges": len(edges)}}


# ===========================================================================
# Soft (semantic) matching — headline metric next to the strict one.
#
# Slot-wise: subject/predicate/object are embedded separately (SBERT via
# src.alignment; TF-IDF fallback in torch-less sandboxes — flagged via
# `backend`, never report fallback numbers). A gold/predicted pair matches
# (is a *link*) when every slot clears its gate AND the weighted overall clears
# its gate; by default every link counts (many-to-many coverage, see the module
# docstring), and greedy 1:1 by overall similarity is the opt-in ablation
# `assignment="one_to_one"`. Numeric slots short-circuit
# via compare_core.numeric_override, which speaks only where the embedding is
# blind: different numbers -> 0.0 (embeddings are hopeless at "610.0" vs
# "4349.0"), equal numbers with a bare literal on one side -> 1.0, equal numbers
# with other content on both sides -> the embedding decides, because it can see
# "per tonne" against "per hectare". Unmatched items carry a *reason*,
# which is exactly the signal the guideline-refinement loop consumes.
# ===========================================================================

DEFAULT_THRESHOLDS = {  # tuned for SBERT MiniLM on short triple slots
    "subject": 0.70, "object": 0.70, "predicate": 0.45, "overall": 0.62,
}
TFIDF_THRESHOLDS = {  # dev-sandbox fallback only
    "subject": 0.55, "object": 0.55, "predicate": 0.35, "overall": 0.50,
}
_WEIGHTS = {"subject": 0.35, "predicate": 0.30, "object": 0.35}
# Kept as a re-export: the tokeniser now lives in compare_core, where it also
# glues EUR-Lex's space-grouped thousands back together ("1 000 000" was read as
# {1, 0}, so "1 000 000 tonnes" and "2 000 000 tonnes" shared the member 0 and
# scored 1.0 under the old intersection rule — BUG-2026-08-27-C).
_NUM_RE = cc._NUM_RE
_NO_CANDIDATE_FLOOR = 0.35  # below this overall sim we call it "no_candidate"

# How links become credit (module docstring). many_to_many is the design of
# record; one_to_one is the pre-2026-09-23 greedy matcher, kept as an ablation
# and never a default. Every result records which one produced it.
MANY_TO_MANY = "many_to_many"
ONE_TO_ONE = "one_to_one"
ASSIGNMENTS = (MANY_TO_MANY, ONE_TO_ONE)
DEFAULT_ASSIGNMENT = MANY_TO_MANY


def check_assignment(assignment: str) -> str:
    """Refuse an unknown mode loudly rather than fall back to a default: a
    misspelt ablation flag must not quietly produce the measure of record."""
    if assignment not in ASSIGNMENTS:
        raise ValueError(f"assignment={assignment!r}; expected one of {ASSIGNMENTS}")
    return assignment


def _numbers(s: str) -> set[float]:
    """Every number in the string, citations included — see
    `compare_core.numeric_override` for why they are not scrubbed here."""
    return cc.all_numbers(s)


def _dedup_gold(gold_triples: list[dict]) -> list[tuple[tuple, tuple]]:
    seen: dict[tuple, tuple] = {}
    for g in gold_triples or []:
        s, p, o = _gold_spo(g)
        if not normalize_term(s) or not normalize_term(o):
            continue
        seen.setdefault(triple_key(s, p, o), (s, p, o))
    return list(seen.items())


def _dedup_extracted(extracted_facts: list[dict]) -> list[tuple[tuple, dict]]:
    seen: dict[tuple, dict] = {}
    for f in extracted_facts or []:
        s, p, o = f.get("subject", ""), f.get("predicate", ""), f.get("object", "")
        k = triple_key(s, p, o)
        if not k[0] or not k[2]:
            continue
        slot = seen.setdefault(k, {"spo": (s, p, o), "fact_ids": set()})
        if f.get("fact_id"):
            slot["fact_ids"].add(f["fact_id"])
    return list(seen.items())


def _slot_sims(gold_items, ex_items) -> dict[str, "object"]:
    """Similarity matrices per slot, shape (n_gold, n_extracted)."""
    import numpy as np
    from src import alignment

    ng, ne = len(gold_items), len(ex_items)
    slots = ("subject", "predicate", "object")
    if ng == 0 or ne == 0:
        return {sl: np.zeros((ng, ne), dtype=np.float32) for sl in slots}

    def norm(sl: str, spo: tuple) -> str:
        i = {"subject": 0, "predicate": 1, "object": 2}[sl]
        return (normalize_predicate(spo[i]) if sl == "predicate"
                else normalize_term(spo[i])) or " "

    texts: list[str] = []
    for sl in slots:
        texts += [norm(sl, spo) for _, spo in gold_items]
        texts += [norm(sl, item["spo"]) for _, item in ex_items]
    embs = alignment.encode_texts(texts)  # ONE joint fit (TF-IDF validity)

    out: dict = {}
    off = 0
    for sl in slots:
        ge = embs[off:off + ng]; ee = embs[off + ng:off + ng + ne]; off += ng + ne
        sims = (ge @ ee.T).astype(np.float32)
        # exact-normalised-string and numeric overrides
        slot_i = {"subject": 0, "predicate": 1, "object": 2}[sl]
        for gi, (_, gspo) in enumerate(gold_items):
            gtxt = norm(sl, gspo)
            for ei, (_, eitem) in enumerate(ex_items):
                etxt = norm(sl, eitem["spo"])
                if gtxt == etxt:
                    sims[gi, ei] = 1.0
                    continue
                if sl != "predicate":
                    # The override speaks only where the embedding is blind:
                    # different numbers -> 0.0; equal numbers with a bare
                    # literal on one side -> 1.0; equal numbers with other
                    # content on both sides -> the embedding decides, because it
                    # can see "per tonne" against "per hectare".
                    # Until 2026-08-27 this matched on any *shared* number and
                    # forced 1.0 both ways (BUG-2026-08-27-C).
                    v = cc.numeric_override(gspo[slot_i], eitem["spo"][slot_i])
                    if v is not None:
                        sims[gi, ei] = v
        out[sl] = sims
    return out


def _fail_reason(sims: dict, gi: int, ei: int, th: dict, *,
                 consumed: bool = False,
                 consumed_label: str = "gold_already_matched") -> str:
    failed = [sl for sl in ("subject", "predicate", "object")
              if float(sims[sl][gi, ei]) < th[sl]]
    if not failed:
        # every slot gate passed - the pair wasn't matched either because the
        # counterpart was consumed by a better 1:1 partner (redundant
        # extraction / duplicate gold) or because the weighted overall fell
        # short. The two mean different guideline problems - distinguish them.
        #
        # Until 2026-08-27 `consumed` was only ever passed on the SPURIOUS side,
        # so a gold triple whose best prediction had been taken by another gold
        # triple was filed under `overall_below_threshold` — a threshold
        # complaint standing in for a duplicate-gold one. The missing-side
        # reason counts are what a guideline revision is authored from, so the
        # two must be told apart on both sides (BUG-2026-08-27-F).
        #
        # Under many-to-many coverage (the default since 2026-09-23) nothing is
        # ever consumed, so `consumed` is always False there and the two
        # `*_already_matched` labels can only come from the one_to_one ablation.
        # An unlinked unit whose closest counterpart passes every slot gate has
        # then, necessarily, failed the overall gate -- which under the frozen
        # SBERT gates cannot happen: 0.35*0.70 + 0.30*0.45 + 0.35*0.70 = 0.625
        # >= 0.62, so passing the three slot gates implies passing the overall
        # one. `overall_below_threshold` is reachable only under the TF-IDF
        # fallback thresholds, in either assignment mode.
        return consumed_label if consumed else "overall_below_threshold"
    return "+".join(f"{sl}_mismatch" for sl in failed)


def compare_soft(gold_triples: list[dict], extracted_facts: list[dict],
                 thresholds: dict | None = None, *,
                 assignment: str = DEFAULT_ASSIGNMENT) -> dict:
    """Semantic slot-wise matching. Same metric shape as `compare` plus reasons.

    Unit: deduplicated gold triples (recall) and deduplicated predicted triples
    (precision); see the module docstring.

    Match criterion (frozen): subject and object >= 0.70, predicate >= 0.45,
    weighted overall (0.35 / 0.30 / 0.35) >= 0.62, after the normalisation and
    `compare_core.numeric_override` of `_slot_sims`. A pair that clears all four
    gates is a *link*.

    `assignment="many_to_many"` (default, design of record): every link counts.
    `n_gold_covered` = gold units with >= 1 link, `n_pred_supported` =
    predictions with >= 1 link, recall = covered / gold, precision =
    supported / extracted. `matched` lists every link, so a gold triple or a
    prediction can appear in it more than once.

    `assignment="one_to_one"` (ablation only): the pre-2026-09-23 greedy 1:1
    assignment on the overall score. It reproduces the old output exactly;
    covered = supported = links there.

    `n_matched` is kept for legacy readers and equals `n_gold_covered` in both
    modes (gold units matched, which is how the WebNLG view labels it). It is
    NOT the precision numerator under many-to-many; use `n_pred_supported`.
    """
    from src import alignment

    check_assignment(assignment)
    backend = alignment.backend_name()
    th = dict(thresholds or (DEFAULT_THRESHOLDS if backend == "sbert" else TFIDF_THRESHOLDS))

    gold_items = _dedup_gold(gold_triples)
    ex_items = _dedup_extracted(extracted_facts)
    sims = _slot_sims(gold_items, ex_items)

    ng, ne = len(gold_items), len(ex_items)
    overall = None
    if ng and ne:
        import numpy as np
        overall = sum(_WEIGHTS[sl] * sims[sl] for sl in _WEIGHTS).astype(np.float32)

    # Every pair that clears the frozen criterion: the links.
    cands = []
    for gi in range(ng):
        for ei in range(ne):
            ov = float(overall[gi, ei]) if overall is not None else 0.0
            if (ov >= th["overall"]
                    and float(sims["subject"][gi, ei]) >= th["subject"]
                    and float(sims["predicate"][gi, ei]) >= th["predicate"]
                    and float(sims["object"][gi, ei]) >= th["object"]):
                cands.append((ov, gi, ei))
    cands.sort(key=lambda t: -t[0])       # stable: ties keep (gi, ei) order
    pairs: list[tuple[float, int, int]] = []
    if assignment == ONE_TO_ONE:
        # The pre-2026-09-23 matcher, unchanged: greedy on the overall score,
        # each gold and each prediction used at most once.
        g_used: set[int] = set(); e_used: set[int] = set()
        for ov, gi, ei in cands:
            if gi in g_used or ei in e_used:
                continue
            g_used.add(gi); e_used.add(ei); pairs.append((ov, gi, ei))
    else:
        pairs = cands
    g_cov: set[int] = {gi for _, gi, _ in pairs}      # covered gold units
    e_sup: set[int] = {ei for _, _, ei in pairs}      # supported predictions
    # Only exclusivity can make a counterpart "already matched".
    exclusive = assignment == ONE_TO_ONE

    def _simrow(gi, ei, ov):
        return {"subject": round(float(sims["subject"][gi, ei]), 3),
                "predicate": round(float(sims["predicate"][gi, ei]), 3),
                "object": round(float(sims["object"][gi, ei]), 3),
                "overall": round(ov, 3)}

    matched = [{"gold": list(gold_items[gi][1]), "pred": list(ex_items[ei][1]["spo"]),
                "fact_ids": sorted(ex_items[ei][1]["fact_ids"]),
                "sims": _simrow(gi, ei, ov)} for ov, gi, ei in sorted(pairs, key=lambda t: -t[0])]

    missing = []
    for gi in range(ng):
        if gi in g_cov:
            continue
        entry: dict = {"triple": list(gold_items[gi][1])}
        if ne and overall is not None:
            ei = int(overall[gi].argmax()); ov = float(overall[gi, ei])
            if ov < _NO_CANDIDATE_FLOOR:
                entry["reason"] = "no_candidate"
            else:
                entry["reason"] = _fail_reason(
                    sims, gi, ei, th, consumed=exclusive and ei in e_sup,
                    consumed_label="prediction_already_matched")
                entry["best"] = {"pred": list(ex_items[ei][1]["spo"]),
                                 "sims": _simrow(gi, ei, ov)}
        else:
            entry["reason"] = "no_candidate"
        missing.append(entry)

    spurious = []
    for ei in range(ne):
        if ei in e_sup:
            continue
        entry = {"triple": list(ex_items[ei][1]["spo"]),
                 "fact_ids": sorted(ex_items[ei][1]["fact_ids"])}
        if ng and overall is not None:
            gi = int(overall[:, ei].argmax()); ov = float(overall[gi, ei])
            if ov < _NO_CANDIDATE_FLOOR:
                entry["reason"] = "no_gold_counterpart"
            else:
                entry["reason"] = _fail_reason(sims, gi, ei, th,
                                               consumed=exclusive and gi in g_cov)
                entry["best_gold"] = {"triple": list(gold_items[gi][1]),
                                      "sims": _simrow(gi, ei, ov)}
        else:
            entry["reason"] = "no_gold_counterpart"
        spurious.append(entry)

    n_cov, n_sup = len(g_cov), len(e_sup)
    precision = n_sup / ne if ne else 0.0
    recall = n_cov / ng if ng else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    # How much of the credit came from fan-out: predictions linked to several
    # gold triples (merging credited) and gold triples linked by several
    # predictions (redundant coverage). Both are 0 under one_to_one. Reported so
    # that a recall gain carried by one generic prediction is visible.
    per_pred = Counter(ei for _, _, ei in pairs)
    per_gold = Counter(gi for _, gi, _ in pairs)
    return {
        "metrics": {
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "n_gold": ng, "n_extracted": ne,
            "n_gold_covered": n_cov, "n_pred_supported": n_sup, "n_links": len(pairs),
            "n_pred_multi_gold": sum(1 for c in per_pred.values() if c > 1),
            "n_gold_multi_pred": sum(1 for c in per_gold.values() if c > 1),
            "n_matched": n_cov,   # legacy alias: gold units matched (docstring)
            "n_missing": ng - n_cov, "n_spurious": ne - n_sup,
        },
        "assignment": assignment,
        "matched": matched, "missing": missing, "spurious": spurious,
        "matched_fact_ids": sorted({fid for ei in e_sup
                                    for fid in ex_items[ei][1]["fact_ids"]}),
        "spurious_fact_ids": sorted({fid for ei in range(ne) if ei not in e_sup
                                     for fid in ex_items[ei][1]["fact_ids"]}),
        "thresholds": th, "backend": backend,
    }


def compare_full(gold_triples: list[dict], extracted_facts: list[dict],
                 thresholds: dict | None = None, *,
                 assignment: str = DEFAULT_ASSIGNMENT) -> dict:
    """Strict + soft in one payload (the shape /api/gold_compare returns)."""
    soft = compare_soft(gold_triples, extracted_facts, thresholds,
                        assignment=assignment)
    return {"strict": compare(gold_triples, extracted_facts),
            "soft": soft, "backend": soft["backend"]}
