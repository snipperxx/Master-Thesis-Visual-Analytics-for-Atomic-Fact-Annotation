"""Tests for src.alignment — encoder + Hungarian + orphan handling."""

from __future__ import annotations

import numpy as np
import pytest

from src.alignment import (
    AlignedPair,
    align_all_pairs,
    align_two,
    encode_facts,
    encode_facts_joint,
    pair_to_dict,
)


def test_encode_empty_returns_zero_rows():
    emb = encode_facts([])
    assert emb.shape[0] == 0
    assert emb.dtype == np.float32


def test_encode_returns_l2_normalised(two_annotators):
    emb = encode_facts(two_annotators["A"])
    # Row norms should be ~1.0 (L2 normalised, both SBERT and TF-IDF backends)
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), norms


def test_align_two_identical_pair_is_matched(two_annotators):
    """Default regime is one-to-many, so a fact may keep several partners.

    The 3 facts per side are near-identical in content and at tau=0.5 several
    cross-links clear the bar. What is guaranteed is that every fact finds at
    least one partner and no link is below the threshold — not a fixed count,
    which is a property of the Hungarian assignment only (asserted below).
    """
    pairs = align_two(
        two_annotators["A"], two_annotators["B"],
        annotator_a="A", annotator_b="B",
        threshold=0.5,
    )
    matched = [p for p in pairs if p.status == "matched"]
    assert len(matched) >= 3
    assert all(p.cosine >= 0.5 for p in matched)
    assert not [p for p in pairs if p.status != "matched"], "nothing is orphaned here"


def test_align_two_one_to_one_ablation_caps_the_match_count(two_annotators):
    """The prior-work regime, kept behind `one_to_many=False`."""
    pairs = align_two(
        two_annotators["A"], two_annotators["B"],
        annotator_a="A", annotator_b="B",
        threshold=0.5, one_to_many=False,
    )
    matched = [p for p in pairs if p.status == "matched"]
    assert len(matched) == 3
    # Hungarian is an assignment: no fact appears twice on either side.
    assert len({p.fact_a["fact_id"] for p in matched}) == 3
    assert len({p.fact_b["fact_id"] for p in matched}) == 3


def test_align_two_threshold_filters(two_annotators):
    # Set threshold absurdly high — nothing should match.
    pairs = align_two(
        two_annotators["A"], two_annotators["B"],
        annotator_a="A", annotator_b="B",
        threshold=0.999999,
    )
    matched = [p for p in pairs if p.status == "matched"]
    # At most the truly identical pair survives a 1.0-ish threshold
    assert len(matched) <= 1


def test_align_two_one_sided_empty():
    pairs = align_two(
        [], [],  # both empty
        annotator_a="A", annotator_b="B",
        threshold=0.5,
    )
    assert pairs == []


def test_align_two_orphans_when_one_side_empty(two_annotators):
    pairs = align_two(
        two_annotators["A"], [],
        annotator_a="A", annotator_b="B",
        threshold=0.5,
    )
    assert all(p.status == "unmatched_a" for p in pairs)
    assert len(pairs) == 3


def test_align_all_pairs_runs_C_n_2(two_annotators):
    # Add a third annotator so we exercise the multi-pair loop.
    facts3 = list(two_annotators["A"])
    full = {**two_annotators, "C": facts3}
    pairs = align_all_pairs(full, threshold=0.5)
    # 3 annotators -> C(3,2) = 3 unordered pairs; each should produce some matches.
    annotator_pairs = {(p.annotator_a, p.annotator_b) for p in pairs}
    assert len(annotator_pairs) == 3


def test_pair_to_dict_serialisable(two_annotators):
    pairs = align_two(
        two_annotators["A"], two_annotators["B"],
        annotator_a="A", annotator_b="B",
        threshold=0.5,
    )
    import json
    json.dumps([pair_to_dict(p) for p in pairs])  # must not raise


def test_encode_facts_joint_shared_vocab(two_annotators):
    """Joint encoding must produce vectors of the same width across lists."""
    [e_a, e_b] = encode_facts_joint(two_annotators["A"], two_annotators["B"])
    assert e_a.shape[1] == e_b.shape[1], (e_a.shape, e_b.shape)
