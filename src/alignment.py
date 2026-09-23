"""
Phase-2 — Step 1: Cross-annotator fact alignment.

Pipeline contract:
    facts_per_annotator (dict[str, list[AtomicFact]])
      -> encode_facts (SBERT or TF-IDF fallback)
      -> pairwise Hungarian assignment
      -> list[AlignedPair]

Encoder backend
---------------
SBERT (`all-MiniLM-L6-v2`) is the default. When `import torch` or
`sentence_transformers` fail, we fall back to sklearn char-trigram TF-IDF
so the rest of the pipeline stays testable in environments that can't
install torch (notably the Cowork Linux sandbox). Both backends produce
L2-normalised vectors so `A @ B.T` is cosine either way.

For TF-IDF correctness the two sides of any pair must share a vocabulary.
We achieve that with `encode_facts_joint(*lists)` which fits one
TfidfVectorizer on the union and returns per-list row slices. `align_two`
and `align_all_pairs` both go through `encode_facts_joint` so they stay
consistent.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from src import compare_core

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class AlignedPair:
    """One Hungarian assignment between two annotators (or an orphan)."""

    annotator_a: str
    annotator_b: str
    fact_a: dict | None
    fact_b: dict | None
    cosine: float
    status: str           # "matched" | "unmatched_a" | "unmatched_b"
    conflict_label: str = "unlabeled"
    layer1_reason: str | None = None
    layer2_reason: str | None = None


# ---------------------------------------------------------------------------
# Encoder backend resolution
# ---------------------------------------------------------------------------


_SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_BACKEND: str | None = None
_encoder = None


# What text represents a fact to the encoder. DECISION 2026-08-09: the triple.
#
# `natural_language` was the original choice and it is not a neutral one. An
# extractor annotation is one natural_language + one source_quote + N triples,
# and flattening copies that ONE sentence onto every triple — so 33-39 % of each
# annotator's dev facts are embedding-identical to another of their own facts.
# They therefore all link to the same partner at the same cosine, and the group
# comes out one-to-many for a reason that has nothing to do with content.
#
# Measured over 37 dev documents at tau = 0.78 (runs/schema_fanout_check_dev.json,
# reproduce with `python -m scripts.check_schema_fanout`), same facts, same tau,
# only this string changing:
#
#     embedding   orphan%   1:1%   1:n%   fan p90   1:n inside ONE annotation
#     NL             27.6   28.6   43.7         7                      57.0 %
#     triple         34.1   39.0   26.9         4                      18.3 %
#
# and — the check that matters, because this is exactly where the per-slot gates
# failed — the triple KEEPS the granularity population instead of cutting it:
# of 4012 directional-containment links (one side's content words >= 80 % covered
# by the other, strictly shorter) it retains 2702 (67.3 %), a higher rate than it
# retains one-to-many links in general (43.8 %). The slot gates cut those links
# preferentially, which is why they were rejected.
#
# It also removes an inconsistency: gold scoring (`gold_compare.compare_soft`)
# has always embedded the normalised slots and never `natural_language`, so
# alignment was the only stage measuring "same fact?" on a different object than
# the stage that scores against the gold.
#
# `natural_language` remains available as the ABLATION — it is what the
# pre-2026-08-09 numbers were computed on.
FACT_TEXT_MODES = ("triple", "natural_language")
_FACT_TEXT_MODE = "triple"


def set_fact_text_mode(mode: str) -> str:
    """Set the representation the encoder sees. Returns the previous mode.

    Global on purpose: it must be impossible for one half of a comparison to be
    encoded as a triple and the other as a sentence.
    """
    global _FACT_TEXT_MODE
    if mode not in FACT_TEXT_MODES:
        raise ValueError(f"fact text mode must be one of {FACT_TEXT_MODES}, got {mode!r}")
    prev, _FACT_TEXT_MODE = _FACT_TEXT_MODE, mode
    return prev


def fact_text_mode() -> str:
    """Callers must surface this next to any similarity number they report."""
    return _FACT_TEXT_MODE


def _triple_text(fact: dict) -> str:
    """The triple text of a fact, WHATEVER shape it is in.

    Deliberately not an alias for `compare_core._flat_triple`: that reads the
    flat slots only and returns `""` for a gold fact, which is precisely the
    trap `_fact_text` below was just repaired for (BUG-2026-08-27-B). A name
    this general must not be shape-specific.
    """
    return compare_core._flat_triple(fact) or compare_core._nested_triples(fact)


def _fact_text(fact: dict) -> str:
    """The string this fact is embedded as, under the current global mode.

    Delegates to `compare_core.fact_text` since 2026-08-27. The behaviour for a
    FLAT fact — model output, everything in data/facts/, everything tau = 0.78
    was swept on — is byte-identical to what it replaced.
    What changed is the NESTED shape (`triples: [...]`, i.e. the gold). It used
    to produce an empty triple string and fall through to `natural_language`
    without a word, so `fact_text_mode()` answered "triple" while the encoder
    was being handed sentences, and every gold A/B similarity number was
    computed on a representation nobody had chosen (BUG-2026-08-27-B).
    """
    return compare_core.fact_text(fact, _FACT_TEXT_MODE)


def fact_text_report(facts: list[dict]) -> dict:
    """What `_fact_text` actually did to this corpus, under the current mode.

    Report it beside any similarity number: `fell_back` > 0 means the number was
    computed on a mixed representation, and `fact_text_mode()` alone does not
    say so."""
    return compare_core.fact_text_report(facts, _FACT_TEXT_MODE)


def _resolve_backend() -> str:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    try:
        import torch  # noqa: F401
        from sentence_transformers import SentenceTransformer  # noqa: F401

        _BACKEND = "sbert"
        logger.info("Alignment backend: SBERT (%s)", _SBERT_MODEL_NAME)
    except Exception as exc:
        _BACKEND = "tfidf"
        logger.warning(
            "Alignment backend: TF-IDF fallback (sentence-transformers unavailable: %s)",
            exc.__class__.__name__,
        )
    return _BACKEND


def _get_sbert():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer(_SBERT_MODEL_NAME, device="cpu")
    return _encoder


def _encode_sbert(texts: list[str]) -> np.ndarray:
    enc = _get_sbert()
    emb = enc.encode(texts, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False)
    return emb.astype(np.float32, copy=False)


def _encode_tfidf(texts: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                          min_df=1, sublinear_tf=True)
    mat = vec.fit_transform(texts)
    mat = normalize(mat, norm="l2", axis=1, copy=False)
    return mat.toarray().astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Public encoding API
# ---------------------------------------------------------------------------


def encode_facts(facts: list[dict]) -> np.ndarray:
    """Returns (N, D) L2-normalised float32 embeddings, or (0, 384) when empty."""
    if not facts:
        return np.zeros((0, 384), dtype=np.float32)
    texts = [_fact_text(f) for f in facts]
    backend = _resolve_backend()
    return _encode_sbert(texts) if backend == "sbert" else _encode_tfidf(texts)


def encode_facts_joint(*fact_lists: list[dict]) -> list[np.ndarray]:
    """Encode multiple lists under one (shared-vocabulary) fit.

    SBERT doesn't need a shared vocabulary, but TF-IDF does — `A @ B.T`
    only makes sense when columns are the same features. Calling this
    helper is always safe; under SBERT it degenerates to per-list encoding.
    """
    backend = _resolve_backend()
    if backend == "sbert":
        return [encode_facts(fl) for fl in fact_lists]

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    all_texts: list[str] = []
    slices: list[tuple[int, int]] = []
    for fl in fact_lists:
        start = len(all_texts)
        all_texts.extend(_fact_text(f) for f in fl)
        slices.append((start, len(all_texts)))
    if not all_texts:
        return [np.zeros((0, 1), dtype=np.float32) for _ in fact_lists]

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                          min_df=1, sublinear_tf=True)
    mat = vec.fit_transform(all_texts)
    mat = normalize(mat, norm="l2", axis=1, copy=False).toarray().astype(np.float32, copy=False)
    return [mat[s:e] for (s, e) in slices]


# ---------------------------------------------------------------------------
# Alignment
#
# One-to-many by default (method.tex "One-to-many alignment"). Every fact is
# linked to EVERY fact in the other set above tau, not to its single best
# partner. That is not a tuning preference: a one-to-one assignment can only
# ever emit |A| or |B| edges, so "one annotator wrote one fact where the other
# wrote three" has nowhere to be represented — the extra two facts are forced
# out as orphans and the split becomes invisible. Granularity is the conflict
# type this thesis is about, so the assignment has to be able to express it.
#
# `one_to_many=False` restores the Hungarian assignment. It is kept because it
# is what the prior work does (schmidt2025dissecting) and is therefore the
# ablation the Discussion compares against — not as a fallback.
# ---------------------------------------------------------------------------


def _hungarian_edges(cos: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    from scipy.optimize import linear_sum_assignment

    rows_a, cols_b = linear_sum_assignment(1.0 - cos)
    return [(int(ia), int(ib), float(cos[ia, ib]))
            for ia, ib in zip(rows_a, cols_b) if cos[ia, ib] >= threshold]


def _one_to_many_edges(cos: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    ia_idx, ib_idx = np.nonzero(cos >= threshold)
    return [(int(ia), int(ib), float(cos[ia, ib])) for ia, ib in zip(ia_idx, ib_idx)]


# Slot gates, taken from gold_compare so linking and scoring use one criterion
# (method.tex: "the same grouping logic is reused to score predictions against
# the gold sample"). Whole-string cosine is too permissive on EUR-Lex, where
# every fact in a document shares the same institutional vocabulary — see
# runs/alignment_calibration_dev.json.
SLOT_GATES = {"subject": 0.70, "object": 0.70, "predicate": 0.45}


def _slot_matrix(facts_a: list[dict], facts_b: list[dict], slot: str) -> np.ndarray:
    """Per-slot cosine, on the SAME normalised strings gold_compare gates.

    Until 2026-08-27 this embedded the raw slot text while `gold_compare._slot_sims`
    embedded the normalised form, yet both applied the identical gate values —
    so the comment above SLOT_GATES ("taken from gold_compare so linking and
    scoring use one criterion") was false: same numbers, different inputs
    (BUG-2026-08-27-E). Affects the `--slot-gates` ablation only, which is not a
    default and carries no number of record.
    """
    def texts(facts: list[dict]) -> list[str]:
        return [compare_core.normalize_slot(slot, f.get(slot) or "") or " "
                for f in facts]

    ta, tb = texts(facts_a), texts(facts_b)
    emb = encode_texts(ta + tb)
    return emb[:len(ta)] @ emb[len(ta):].T


def _slot_gated_edges(facts_a: list[dict], facts_b: list[dict], cos: np.ndarray,
                      threshold: float, gates: dict) -> list[tuple[int, int, float]]:
    """Link only where the whole-string cosine AND every slot gate agree."""
    ok = cos >= threshold
    for slot, gate in gates.items():
        ok &= _slot_matrix(facts_a, facts_b, slot) >= gate
    ia_idx, ib_idx = np.nonzero(ok)
    return [(int(ia), int(ib), float(cos[ia, ib])) for ia, ib in zip(ia_idx, ib_idx)]


def align_two(
    facts_a: list[dict],
    facts_b: list[dict],
    *,
    annotator_a: str,
    annotator_b: str,
    threshold: float = 0.78,
    emb_a: np.ndarray | None = None,
    emb_b: np.ndarray | None = None,
    one_to_many: bool = True,
    slot_gates: dict | None = None,
) -> list[AlignedPair]:
    """Link two annotators' facts above `threshold`.

    Default is one-to-many: a fact may appear in several matched pairs. Pass
    `one_to_many=False` for the one-to-one Hungarian assignment (the ablation).
    Facts with no partner above the threshold come back as orphan pairs tagged
    with their best-partner similarity, under both regimes.
    """
    if not facts_a and not facts_b:
        return []
    if not facts_a:
        return [
            AlignedPair(annotator_a, annotator_b, None, fb, 0.0, "unmatched_b")
            for fb in facts_b
        ]
    if not facts_b:
        return [
            AlignedPair(annotator_a, annotator_b, fa, None, 0.0, "unmatched_a")
            for fa in facts_a
        ]

    # Joint encoding ensures TF-IDF backend has a shared vocab.
    if emb_a is None or emb_b is None:
        joint = encode_facts_joint(facts_a, facts_b)
        if emb_a is None:
            emb_a = joint[0]
        if emb_b is None:
            emb_b = joint[1]

    cos = emb_a @ emb_b.T
    if not one_to_many:
        edges = _hungarian_edges(cos, threshold)
    elif slot_gates:
        edges = _slot_gated_edges(facts_a, facts_b, cos, threshold, slot_gates)
    else:
        edges = _one_to_many_edges(cos, threshold)

    matched_a: set[int] = set()
    matched_b: set[int] = set()
    pairs: list[AlignedPair] = []

    for ia, ib, sim in edges:
        pairs.append(AlignedPair(
            annotator_a=annotator_a, annotator_b=annotator_b,
            fact_a=facts_a[ia], fact_b=facts_b[ib],
            cosine=sim, status="matched",
        ))
        matched_a.add(ia)
        matched_b.add(ib)

    for ia, fa in enumerate(facts_a):
        if ia not in matched_a:
            best_b = int(np.argmax(cos[ia])) if cos.shape[1] else -1
            best_sim = float(cos[ia, best_b]) if best_b >= 0 else 0.0
            pairs.append(AlignedPair(
                annotator_a=annotator_a, annotator_b=annotator_b,
                fact_a=fa, fact_b=None, cosine=best_sim,
                status="unmatched_a",
                layer1_reason=f"best partner cosine={best_sim:.3f} < {threshold}",
            ))

    for ib, fb in enumerate(facts_b):
        if ib not in matched_b:
            best_a = int(np.argmax(cos[:, ib])) if cos.shape[0] else -1
            best_sim = float(cos[best_a, ib]) if best_a >= 0 else 0.0
            pairs.append(AlignedPair(
                annotator_a=annotator_a, annotator_b=annotator_b,
                fact_a=None, fact_b=fb, cosine=best_sim,
                status="unmatched_b",
                layer1_reason=f"best partner cosine={best_sim:.3f} < {threshold}",
            ))

    return pairs


def align_all_pairs(
    facts_per_annotator: dict[str, list[dict]],
    *,
    threshold: float = 0.78,
    one_to_many: bool = True,
    slot_gates: dict | None = None,
) -> list[AlignedPair]:
    """Align every unordered annotator pair; see `align_two` for the regime."""
    annotators = list(facts_per_annotator.keys())
    embs = encode_facts_joint(*[facts_per_annotator[a] for a in annotators])
    emb_cache = dict(zip(annotators, embs))

    all_pairs: list[AlignedPair] = []
    for i in range(len(annotators)):
        for j in range(i + 1, len(annotators)):
            ann_a, ann_b = annotators[i], annotators[j]
            pairs = align_two(
                facts_per_annotator[ann_a],
                facts_per_annotator[ann_b],
                annotator_a=ann_a, annotator_b=ann_b,
                threshold=threshold,
                emb_a=emb_cache[ann_a], emb_b=emb_cache[ann_b],
                one_to_many=one_to_many, slot_gates=slot_gates,
            )
            all_pairs.extend(pairs)
    return all_pairs


# ---------------------------------------------------------------------------
# Groups — the unit of analysis
#
# method.tex: "A group is a fact together with the one or more partners it
# matches" and "The group, not the pair, is the unit of analysis."
#
# So a group is FACT-CENTRIC: one anchor fact plus its partners above tau. Each
# fact anchors one group, which means a1's group and b1's group can overlap.
# The thesis accepts that: the conflict-type distribution is over groups, and
# both readings ("a1 was split into b1,b2" and "b1 was merged into a1") are
# claims about the same span that a reader of the graph should see.
#
# Connected components were tried first and rejected on 2026-08-08. EUR-Lex
# facts share so much boilerplate vocabulary that the match graph is dense:
# on a dev document at tau=0.85 the components collapsed into four blobs
# holding ~95 of 150 fact slots, which erases every shape distinction the
# grouping exists to make. Transitivity is not safe on this similarity.
# ---------------------------------------------------------------------------


@dataclass
class FactGroup:
    """One anchor fact plus the partners it matches, within one annotator pair."""

    annotator_a: str
    annotator_b: str
    anchor_side: str              # "a" | "b" — which annotator the anchor is from
    anchor: dict
    partners: list[dict]
    shape: str                    # "orphan" | "one_to_one" | "one_to_many"
    cosines: list[float]
    conflict_label: str = "unlabeled"
    label_source: str = ""        # "shape" | "layer1" | "layer2"
    reason: str = ""
    # The label this group carried BEFORE Layer-2 overrode it, when it did.
    # Only set by conflict_layer2.arbitrate_group, so a shape-derived
    # `granularity` that the model disagreed with stays recoverable from the
    # output rather than being silently replaced.
    prior_label: str = ""

    @property
    def fan_out(self) -> int:
        return len(self.partners)

    @property
    def max_cosine(self) -> float:
        return max(self.cosines) if self.cosines else 0.0

    @property
    def facts_a(self) -> list[dict]:
        return [self.anchor] if self.anchor_side == "a" else list(self.partners)

    @property
    def facts_b(self) -> list[dict]:
        return [self.anchor] if self.anchor_side == "b" else list(self.partners)


def _shape_of(fan_out: int) -> str:
    if fan_out == 0:
        return "orphan"
    return "one_to_one" if fan_out == 1 else "one_to_many"


def build_groups(pairs: list[AlignedPair]) -> list[FactGroup]:
    """One group per fact: the fact plus every partner it matched.

    Orphans become groups of fan-out 0, carrying their best-partner similarity,
    so the orphan rate is read off the same structure as everything else.
    """
    # (annotator_pair, side, fact_id) -> [(partner, cosine)]
    partners: dict[tuple[str, str, str, str], list[tuple[dict, float]]] = defaultdict(list)
    anchors: dict[tuple[str, str, str, str], dict] = {}
    orphan_best: dict[tuple[str, str, str, str], float] = {}

    def key(p: AlignedPair, side: str, fact: dict, idx: int):
        return (p.annotator_a, p.annotator_b, side,
                str(fact.get("fact_id") or f"{side}{idx}"))

    for i, p in enumerate(pairs):
        if p.status == "matched" and p.fact_a is not None and p.fact_b is not None:
            ka = key(p, "a", p.fact_a, i)
            kb = key(p, "b", p.fact_b, i)
            anchors.setdefault(ka, p.fact_a)
            anchors.setdefault(kb, p.fact_b)
            partners[ka].append((p.fact_b, p.cosine))
            partners[kb].append((p.fact_a, p.cosine))
        else:
            side = "a" if p.fact_a is not None else "b"
            fact = p.fact_a if p.fact_a is not None else p.fact_b
            if fact is None:
                continue
            k = key(p, side, fact, i)
            anchors.setdefault(k, fact)
            orphan_best[k] = max(orphan_best.get(k, 0.0), p.cosine)

    groups: list[FactGroup] = []
    for k, anchor in anchors.items():
        ann_a, ann_b, side, _fid = k
        plist = partners.get(k, [])
        groups.append(FactGroup(
            annotator_a=ann_a, annotator_b=ann_b,
            anchor_side=side, anchor=anchor,
            partners=[f for f, _ in plist],
            shape=_shape_of(len(plist)),
            cosines=([c for _, c in plist] or
                     ([orphan_best[k]] if k in orphan_best else [])),
        ))
    return groups


def _partner_annotation_groups(g: FactGroup) -> list[str]:
    """The distinct annotation groups (`group_id`) the partners come from.

    One extractor annotation is `natural_language` + `source_quote` + N triples,
    and flattening it copies the SAME natural_language onto each triple
    (`extractor.py`, "Each group is ONE annotation"). Since alignment embeds
    natural_language, every triple of one annotation is embedding-identical, so
    they all link to the same partner at the same cosine and the group comes out
    one-to-many. That fan-out is the reification schema, NOT one annotator
    splitting what another merged — measured on train-000004, 29 of 36
    one-to-many groups had every partner inside a single annotation group, and
    36.5 % of dev annotation groups carry more than one triple.

    Reported, never acted on: which similarity the linking criterion should use
    is a method.tex question (CLAUDE.md blocker 1), so this only makes the
    phenomenon countable.
    """
    return sorted({(f.get("group_id") or "").strip() for f in g.partners} - {""})


def group_to_dict(g: FactGroup) -> dict:
    partner_ann_groups = _partner_annotation_groups(g)
    return {
        "annotator_a": g.annotator_a,
        "annotator_b": g.annotator_b,
        "anchor_side": g.anchor_side,
        "anchor_id": g.anchor.get("fact_id"),
        "partner_ids": [f.get("fact_id") for f in g.partners],
        "shape": g.shape,
        "fan_out": g.fan_out,
        "max_cosine": round(g.max_cosine, 4),
        "conflict_label": g.conflict_label,
        "label_source": g.label_source,
        "prior_label": g.prior_label,
        "reason": g.reason,
        # Diagnostic only — see _partner_annotation_groups. True means the whole
        # fan-out sits inside ONE annotation of one annotator, so reading this
        # group as a granularity disagreement would be reading the schema.
        "partner_annotation_groups": partner_ann_groups,
        "partners_one_annotation": bool(g.fan_out > 1 and len(partner_ann_groups) == 1),
    }


def pair_to_dict(p: AlignedPair) -> dict:
    return {
        "annotator_a": p.annotator_a,
        "annotator_b": p.annotator_b,
        "fact_a_id": p.fact_a["fact_id"] if p.fact_a else None,
        "fact_b_id": p.fact_b["fact_id"] if p.fact_b else None,
        "cosine": round(p.cosine, 4),
        "status": p.status,
        "conflict_label": p.conflict_label,
        "layer1_reason": p.layer1_reason,
        "layer2_reason": p.layer2_reason,
    }


def encode_texts(texts: list[str]) -> "np.ndarray":
    """L2-normalised embeddings for arbitrary short strings.

    One call = one joint fit, so TF-IDF cosines are only valid *within* a
    single call (SBERT doesn't care). Used by gold_compare soft matching.
    """
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    backend = _resolve_backend()
    return _encode_sbert(list(texts)) if backend == "sbert" else _encode_tfidf(list(texts))


def backend_name() -> str:
    """'sbert' or 'tfidf' — callers must surface this next to any similarity
    numbers (TF-IDF results are a dev fallback, never report them as eval)."""
    return _resolve_backend()
