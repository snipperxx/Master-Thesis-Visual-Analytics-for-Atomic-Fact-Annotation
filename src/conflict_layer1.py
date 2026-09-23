"""
Phase-2 — Step 2a: Cheap rule-based conflict labeller.

The Layer-1 filter exists to keep the Layer-2 LLM arbitrator off the
trivial cases. The proposal lists exactly the rules below; we keep the
parameters explicit and per-pair so they're tweakable from the VA tool
later (e.g. "what if I move REDUNDANCY's cosine cutoff from 0.95 to 0.92?").

Decision flow for a `matched` pair:

    cosine ≥ redundancy_cosine
    AND   subj+obj char overlap ≥ overlap_threshold
                 → REDUNDANCY  (filtered, no LLM call)

    content-quantity divergence      → escalate (likely CONTRADICTION)
    polarity flips (not/no/never)    → escalate (antonym trap)
    otherwise                        → escalate

Orphan pairs (`unmatched_a` / `unmatched_b`) are left UNLABELED. They are
candidates for GRANULARITY at the document level (handled by Layer-2 in
the cluster context) or true coverage gaps; Layer-1 deliberately does not
guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import compare_core as cc
from .alignment import AlignedPair

# ---------------------------------------------------------------------------
# Tunables (exposed via __init__ so the UI can override)
# ---------------------------------------------------------------------------


@dataclass
class Layer1Config:
    redundancy_cosine: float = 0.95     # very high → near-duplicate phrasing
    overlap_threshold: float = 0.75     # subject+object char-trigram Jaccard
    # Words whose presence on exactly one side flips the polarity of a claim.
    polarity_markers: tuple[str, ...] = (
        "not", "no", "never", "without", "fails", "failed", "unable",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_NUMBER_RE = re.compile(r"\d+[,.]?\d*")
_WORD_RE = re.compile(r"[A-Za-z]+")
# Legal-citation "No" — "Regulation (EEC) No 584/75", "No. 1418/76" — must NOT
# be read as the negation "no". Without this scrub, every fact that cites a
# regulation number trips the polarity-asymmetry escalation (observed on
# train-000014: gemma "security is ECU 20" vs phi4 "... No 584/75 shall be ...").
_CITATION_NO_RE = re.compile(r"\bno\.?\s*\d", re.IGNORECASE)

# --- Citation / reference / date numbers vs genuine content quantities -------
# Legal prose is saturated with reference numbers — "Regulation (EC) No 1260/2001",
# "Article 15(5)", "Decision 2005/136/EC", "2 June 2004". Treated as quantities
# they made the numeric-divergence rule fire ~77% false on the EUR-Lex set
# (citation/date/one-sided-truncation noise, not real value conflicts). We scrub
# these BEFORE comparing so the rule sees only content quantities (amounts,
# percentages, prices). Trade-off: bare 4-digit years are dropped too, so a rare
# 4-digit *quantity* (e.g. "2000 tonnes") is not compared numerically — acceptable
# in this corpus, where 4-digit numbers are overwhelmingly years/reg numbers.
# The patterns themselves moved to src/compare_core.py on 2026-08-27 — this
# module and gold_compare had begun to disagree about what a number is, which
# is exactly the drift the shared core exists to stop. Re-exported so anything
# importing the name keeps working.
_MONTHS = cc._MONTHS
_NOISE_NUMBER_PATTERNS = cc._REFERENCE_PATTERNS

# Fixed legal phrases where a negation marker does NOT flip the claim's polarity
# ("without prejudice to Article 3", "not later than 30 days"). Left in, they made
# the polarity rule fire ~93% false on EUR-Lex (only 2/28 were real polarity
# contradictions). Scrubbed before token extraction. Genuinely semantic negations
# such as "shall not apply" / "shall not affect" are deliberately NOT listed here.
_POLARITY_STOP_PHRASES = re.compile(
    r"without\s+prejudice|without\s+delay|not\s+later\s+than|"
    r"not\s+exceeding|not\s+less\s+than|not\s+more\s+than|"
    r"no\s+later\s+than|no\s+less\s+than|no\s+more\s+than|unless\s+otherwise",
    re.I,
)


def _scrub_noise_numbers(text: str) -> str:
    """Delegates to `compare_core.scrub_reference_numbers` since 2026-08-27.

    Same patterns, two ordering/tokenising fixes that this module wanted anyway
    (BUG-2026-08-27-D): the bare slash citation runs first, so
    `Decision 2005/136/EC` no longer leaves 136 behind as a pseudo-quantity;
    and grouped thousands survive tokenisation, so `1 350 000 tonnes` is the
    quantity 1350000 instead of the set {0, 1, 350}.
    """
    return cc.scrub_reference_numbers(text)


def _char_trigram_jaccard(a: str, b: str) -> float:
    """Char-trigram Jaccard on the concatenated subject+object — captures
    'same referent, same value' even when the function words differ.
    """

    def trigrams(s: str) -> set[str]:
        s = re.sub(r"\s+", " ", s).strip().lower()
        return {s[i : i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else set()

    sa, sb = trigrams(a), trigrams(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _numbers(s: str) -> list[str]:
    return _NUMBER_RE.findall(s or "")


def _content_numbers(s: str) -> set[float]:
    """Numbers that survive citation/date/reference scrubbing — i.e. genuine
    content quantities (amounts, percentages, prices), not legal references."""
    return cc.content_numbers(s)


def _polarity_tokens(s: str, markers: tuple[str, ...]) -> set[str]:
    s = _CITATION_NO_RE.sub(" ", s or "")
    s = _POLARITY_STOP_PHRASES.sub(" ", s)
    return {w.lower() for w in _WORD_RE.findall(s) if w.lower() in markers}


def _so_text(fact: dict) -> str:
    """Subject + object — what we compare for REDUNDANCY overlap."""
    return f"{fact.get('subject','')} {fact.get('object','')}".strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_pair(pair: AlignedPair, cfg: Layer1Config | None = None) -> AlignedPair:
    """In-place: set `conflict_label` and `layer1_reason` on the pair.

    Returns the (mutated) pair so this composes naturally in a map.
    """
    cfg = cfg or Layer1Config()
    if pair.status != "matched":
        # Orphans are left for Layer-2 / cluster-level reasoning.
        pair.layer1_reason = pair.layer1_reason or f"orphan_{pair.status}"
        return pair

    fa, fb = pair.fact_a, pair.fact_b  # both non-None when matched
    assert fa is not None and fb is not None

    so_a, so_b = _so_text(fa), _so_text(fb)
    overlap = _char_trigram_jaccard(so_a, so_b)

    # ---- REDUNDANCY ---------------------------------------------------------
    if pair.cosine >= cfg.redundancy_cosine and overlap >= cfg.overlap_threshold:
        pair.conflict_label = "redundancy"
        pair.layer1_reason = (
            f"cos={pair.cosine:.3f}≥{cfg.redundancy_cosine}; "
            f"subj+obj trigram jaccard={overlap:.2f}≥{cfg.overlap_threshold}"
        )
        return pair

    # ---- Numeric divergence (escalate, hint CONTRADICTION) -----------------
    # Compare *content quantities* only (citation/reference/date numbers scrubbed).
    # Fire only on a genuine divergence: both sides carry a quantity AND the sets
    # differ AND neither is a subset of the other. A one-sided extra number is a
    # granularity/coverage difference (one annotator carried the value, the other
    # truncated it) — that is left to the default escalation, not flagged as a
    # numeric (contradiction-hinting) mismatch.
    nums_a, nums_b = _content_numbers(so_a), _content_numbers(so_b)
    if (nums_a and nums_b and nums_a != nums_b
            and not nums_a <= nums_b and not nums_b <= nums_a):
        pair.conflict_label = "escalate"
        pair.layer1_reason = (
            f"numeric mismatch: A={sorted(nums_a)}  B={sorted(nums_b)}; "
            f"cos={pair.cosine:.3f}"
        )
        return pair

    # ---- Polarity / antonym trap (escalate) --------------------------------
    pol_a = _polarity_tokens(fa.get("natural_language", ""), cfg.polarity_markers)
    pol_b = _polarity_tokens(fb.get("natural_language", ""), cfg.polarity_markers)
    if pol_a ^ pol_b:
        pair.conflict_label = "escalate"
        pair.layer1_reason = (
            f"polarity asymmetry: A={sorted(pol_a) or '∅'} B={sorted(pol_b) or '∅'}"
        )
        return pair

    # ---- Default: escalate to LLM ------------------------------------------
    pair.conflict_label = "escalate"
    pair.layer1_reason = (
        f"no rule fired; cos={pair.cosine:.3f} overlap={overlap:.2f}"
    )
    return pair


def classify_all(pairs: list[AlignedPair], cfg: Layer1Config | None = None) -> dict[str, int]:
    """Run `classify_pair` over every pair; return label counts."""
    counts: dict[str, int] = {}
    for p in pairs:
        classify_pair(p, cfg)
        counts[p.conflict_label] = counts.get(p.conflict_label, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Group-level labelling
#
# method.tex: "A one-to-many group is labelled GRANULARITY directly from its
# shape." One annotator wrote one fact where the other wrote several, over the
# same span — that IS the granularity difference, and no arbitration call is
# needed to see it. Under the old one-to-one assignment this shape could not
# occur, which is why granularity had to be guessed at pair level and mostly
# was not found.
# ---------------------------------------------------------------------------


def classify_group(group, cfg: Layer1Config | None = None):
    """Label one FactGroup. Mutates and returns it."""
    if group.shape == "one_to_many":
        group.conflict_label = "granularity"
        group.label_source = "shape"
        group.reason = (f"one-to-many group, fan-out {group.fan_out} — one "
                        f"annotator split what the other kept whole")
        return group

    if group.shape == "orphan":
        # A fact nobody else came near. Not a conflict type: it is coverage,
        # and the orphan rate is reported separately (method.tex).
        group.conflict_label = "unlabeled"
        group.label_source = "shape"
        group.reason = f"orphan; best partner cosine={group.max_cosine:.3f}"
        return group

    # one_to_one: the pair rules apply unchanged.
    fa = group.anchor if group.anchor_side == "a" else group.partners[0]
    fb = group.partners[0] if group.anchor_side == "a" else group.anchor
    probe = AlignedPair(
        annotator_a=group.annotator_a, annotator_b=group.annotator_b,
        fact_a=fa, fact_b=fb,
        cosine=group.max_cosine, status="matched",
    )
    classify_pair(probe, cfg)
    group.conflict_label = probe.conflict_label
    group.label_source = "layer1"
    group.reason = probe.layer1_reason or ""
    return group


def classify_groups(groups: list, cfg: Layer1Config | None = None) -> dict[str, int]:
    """Run `classify_group` over every group; return label counts by shape."""
    counts: dict[str, int] = {}
    for g in groups:
        classify_group(g, cfg)
        counts[g.conflict_label] = counts.get(g.conflict_label, 0) + 1
    return counts
