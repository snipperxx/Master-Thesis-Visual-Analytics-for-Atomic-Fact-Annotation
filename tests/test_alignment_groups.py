"""Guards for one-to-many alignment and the group layer.

The regression this file exists to prevent: reverting to a one-to-one
assignment. Under one-to-one a fact can have at most one partner, so a
granularity split — one annotator writing one fact where another writes three —
has nowhere to be represented and the conflict-type distribution silently loses
the category the thesis is about. `test_one_to_one_cannot_express_granularity`
pins that difference.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import alignment as al
from src.conflict_layer1 import classify_group, classify_groups


def _fact(fid: str, subject: str, predicate: str, obj: str) -> dict:
    return {"fact_id": fid, "subject": subject, "predicate": predicate,
            "object": obj,
            "natural_language": f"{subject} {predicate} {obj}."}


# One merged fact on side A; the same content split in two on side B.
SPLIT_A = [_fact("a1", "the Commission", "shall fix", "the maximum export refund and the quantity")]
SPLIT_B = [_fact("b1", "the Commission", "shall fix", "the maximum export refund"),
           _fact("b2", "the Commission", "shall fix", "the quantity")]


def _align(facts_a, facts_b, **kw):
    return al.align_two(facts_a, facts_b, annotator_a="A", annotator_b="B", **kw)


def test_one_to_many_links_every_partner_above_tau():
    pairs = _align(SPLIT_A, SPLIT_B, threshold=0.5, one_to_many=True)
    matched = [p for p in pairs if p.status == "matched"]
    assert len(matched) == 2, "a1 must link to both b1 and b2"
    assert {p.fact_b["fact_id"] for p in matched} == {"b1", "b2"}


def test_one_to_one_cannot_express_granularity():
    """The ablation, and the reason the default changed."""
    pairs = _align(SPLIT_A, SPLIT_B, threshold=0.5, one_to_many=False)
    matched = [p for p in pairs if p.status == "matched"]
    assert len(matched) == 1, "Hungarian assigns a1 to exactly one partner"
    groups = al.build_groups(pairs)
    assert not [g for g in groups if g.shape == "one_to_many"]
    # ...and the second gold-split fact is pushed out as an orphan instead.
    assert [g for g in groups if g.shape == "orphan"]


def test_group_shapes_and_granularity_from_shape():
    groups = al.build_groups(_align(SPLIT_A, SPLIT_B, threshold=0.5))
    by_anchor = {g.anchor["fact_id"]: g for g in groups}
    assert by_anchor["a1"].shape == "one_to_many"
    assert by_anchor["a1"].fan_out == 2
    assert by_anchor["b1"].shape == "one_to_one"

    classify_group(by_anchor["a1"])
    assert by_anchor["a1"].conflict_label == "granularity"
    assert by_anchor["a1"].label_source == "shape", "no LLM call needed for shape"


def test_orphans_become_groups_carrying_best_similarity():
    unrelated = [_fact("b9", "the Council", "adopted", "a recommendation on fiscal policy")]
    pairs = _align(SPLIT_A, unrelated, threshold=0.99)
    groups = al.build_groups(pairs)
    assert {g.shape for g in groups} == {"orphan"}
    assert all(g.fan_out == 0 for g in groups)
    # The orphan still reports how close its best partner came.
    assert any(g.max_cosine > 0 for g in groups)

    counts = classify_groups(groups)
    assert counts.get("unlabeled") == len(groups), "orphans are coverage, not a conflict type"


def test_every_fact_anchors_exactly_one_group_per_annotator_pair():
    pairs = _align(SPLIT_A, SPLIT_B, threshold=0.5)
    groups = al.build_groups(pairs)
    anchors = [g.anchor["fact_id"] for g in groups]
    assert sorted(anchors) == ["a1", "b1", "b2"]
    assert len(anchors) == len(set(anchors))


def test_slot_gates_reject_a_boilerplate_only_match():
    """Two facts that share institutional vocabulary but assert different things.

    Whole-string cosine links them on a permissive tau; the slot gates should
    not, because neither the subject nor the object agrees. This is the failure
    that put ~40% of dev groups in the one-to-many bucket.
    """
    a = [_fact("a1", "the Commission", "shall fix", "the maximum export refund")]
    b = [_fact("b1", "the Member States", "shall notify", "the quantities exported")]

    loose = [p for p in _align(a, b, threshold=0.3, one_to_many=True)
             if p.status == "matched"]
    gated = [p for p in _align(a, b, threshold=0.3, one_to_many=True,
                               slot_gates=al.SLOT_GATES) if p.status == "matched"]
    assert loose, "the permissive threshold does link them"
    assert not gated, "the slot gates must not"


def test_group_to_dict_is_serialisable():
    groups = al.build_groups(_align(SPLIT_A, SPLIT_B, threshold=0.5))
    classify_groups(groups)
    import json
    payload = json.dumps([al.group_to_dict(g) for g in groups])
    assert "one_to_many" in payload
    assert "granularity" in payload


def test_hungarian_and_one_to_many_agree_when_the_match_is_unambiguous():
    a = [_fact("a1", "the Commission", "shall fix", "the maximum export refund")]
    b = [_fact("b1", "the Commission", "shall fix", "the maximum export refund")]
    for otm in (True, False):
        matched = [p for p in _align(a, b, threshold=0.5, one_to_many=otm)
                   if p.status == "matched"]
        assert len(matched) == 1
        assert matched[0].cosine == pytest.approx(1.0, abs=1e-3)


def test_empty_sides_still_produce_orphan_groups():
    assert al.build_groups(_align([], [], threshold=0.5)) == []
    groups = al.build_groups(_align(SPLIT_A, [], threshold=0.5))
    assert len(groups) == 1 and groups[0].shape == "orphan"


def test_slot_matrix_shape():
    m = al._slot_matrix(SPLIT_A, SPLIT_B, "subject")
    assert m.shape == (1, 2)
    assert np.all(m <= 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# What text represents a fact to the encoder (DECISION 2026-08-09)
#
# The regression: reverting to `natural_language`. One extractor annotation is
# one natural_language + N triples, and flattening copies that sentence onto
# every triple, so a third of all facts become embedding-identical to another of
# their own annotator's facts and fan out as "one-to-many" for no content
# reason. Evidence: runs/schema_fanout_check_dev.json.
# ---------------------------------------------------------------------------


def test_triple_is_the_default_representation():
    assert al.fact_text_mode() == "triple"


def test_facts_sharing_one_natural_language_are_not_embedding_identical():
    """The exact shape the flattener produces: one sentence, different slots."""
    shared = "Refunds must account for the relevant factors."
    a = {"fact_id": "a", "subject": "Refunds", "predicate": "must account for",
         "object": "market conditions", "natural_language": shared}
    b = {"fact_id": "b", "subject": "Refunds", "predicate": "must account for",
         "object": "price and cost factors", "natural_language": shared}
    assert al._fact_text(a) != al._fact_text(b)

    prev = al.set_fact_text_mode("natural_language")
    try:
        # The ablation reproduces the defect — that is what it is for.
        assert al._fact_text(a) == al._fact_text(b)
    finally:
        al.set_fact_text_mode(prev)


def test_mode_is_restorable_and_validated():
    prev = al.set_fact_text_mode("natural_language")
    assert al.fact_text_mode() == "natural_language"
    assert al.set_fact_text_mode(prev) == "natural_language"
    assert al.fact_text_mode() == "triple"
    with pytest.raises(ValueError):
        al.set_fact_text_mode("sentence")


def test_triple_mode_falls_back_when_slots_are_empty():
    """An empty string would match everything or nothing."""
    f = {"fact_id": "x", "subject": "", "predicate": "", "object": "",
         "natural_language": "Something was decided."}
    assert al._fact_text(f) == "Something was decided."
