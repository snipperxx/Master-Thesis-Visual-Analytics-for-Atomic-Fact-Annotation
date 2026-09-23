"""Tests for src.conflict_layer1 — rule-based labeller."""

from __future__ import annotations

from src.alignment import AlignedPair
from src.conflict_layer1 import Layer1Config, classify_pair, classify_all


def _matched_pair(fa: dict, fb: dict, cosine: float = 0.99) -> AlignedPair:
    return AlignedPair(
        annotator_a="A", annotator_b="B",
        fact_a=fa, fact_b=fb,
        cosine=cosine, status="matched",
    )


def test_redundancy_when_cosine_high_and_overlap_high(fact_factory):
    fa = fact_factory("a", "A", "the Council", "decided", "X is true")
    fb = fact_factory("b", "B", "the Council", "decided", "X is true")
    p = _matched_pair(fa, fb, cosine=0.98)
    classify_pair(p, Layer1Config(redundancy_cosine=0.95, overlap_threshold=0.6))
    assert p.conflict_label == "redundancy"
    assert "trigram jaccard" in (p.layer1_reason or "")


def test_numeric_mismatch_escalates(fact_factory):
    fa = fact_factory("a", "A", "deficit", "is", "2,3 % in 2004")
    fb = fact_factory("b", "B", "deficit", "is", "2,8 % in 2004")
    p = _matched_pair(fa, fb, cosine=0.93)  # below redundancy cutoff
    classify_pair(p, Layer1Config(redundancy_cosine=0.95))
    assert p.conflict_label == "escalate"
    assert "numeric mismatch" in (p.layer1_reason or "")


def test_polarity_asymmetry_escalates(fact_factory):
    fa = fact_factory("a", "A", "deficit", "is", "above threshold",
                      nl="The deficit is above threshold.")
    fb = fact_factory("b", "B", "deficit", "is", "above threshold",
                      nl="The deficit is not above threshold.")
    p = _matched_pair(fa, fb, cosine=0.90)
    classify_pair(p, Layer1Config(redundancy_cosine=0.95))
    assert p.conflict_label == "escalate"
    assert "polarity" in (p.layer1_reason or "")


def test_orphans_left_unlabeled(fact_factory):
    fa = fact_factory("a", "A", "x", "y", "z")
    p = AlignedPair("A", "B", fact_a=fa, fact_b=None, cosine=0.0,
                    status="unmatched_a")
    classify_pair(p)
    # Orphans are explicitly left for Layer-2 / cluster reasoning
    assert p.conflict_label == "unlabeled"
    assert "orphan" in (p.layer1_reason or "")


def test_classify_all_returns_counts(fact_factory):
    fa = fact_factory("a", "A", "x", "y", "z")
    fb = fact_factory("b", "B", "x", "y", "z")
    pairs = [_matched_pair(fa, fb, cosine=0.99)]
    counts = classify_all(pairs, Layer1Config(redundancy_cosine=0.95, overlap_threshold=0.5))
    assert counts.get("redundancy") == 1
