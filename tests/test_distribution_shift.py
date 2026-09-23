"""RQ1's primary measure: the conflict-type distribution shift.

Three defects are pinned here, all found 2026-08-09 while wiring Layer-2 into the
group layer. None had produced a wrong published number only because no v2
guideline exists yet, so this module had never been run for record.

1. It read the PAIR distribution while `method.tex` makes the group the unit of
   analysis — and under one-to-many alignment a fact appears in many pairs, so
   the pair distribution double-counts exactly the facts granularity is about.
2. `no_conflict` was missing from `CONFLICT_LABELS`, so every `no_conflict`
   verdict vanished from the distribution *and* from the totals.
3. `escalate` was also invisible. It is not a conflict type — it means Layer-2
   never resolved the item — so hiding it turned "the arbitrator never ran" into
   a clean-looking distribution.
"""

from __future__ import annotations

import pytest

from src.distribution_shift import (CONFLICT_LABELS, UNRESOLVED_LABEL,
                                    compare_conflict_files)


def _doc(*, groups=None, group_counts=None, pair_counts=None,
         group_layer2_calls=0, layer2_calls=0, pairs=None):
    d = {}
    if groups is not None:
        d["groups"] = groups
    if group_counts is not None:
        d["group_label_counts"] = group_counts
    if pair_counts is not None:
        d["label_counts"] = pair_counts
    d["group_layer2_calls"] = group_layer2_calls
    d["layer2_calls"] = layer2_calls
    d["aligned_pairs"] = pairs or []
    return d


def _counts(res, label):
    return res["v1"][res["labels"].index(label)], res["v2"][res["labels"].index(label)]


# --- 1. the unit ------------------------------------------------------------

def test_group_is_the_default_unit():
    a = _doc(group_counts={"granularity": 5}, pair_counts={"granularity": 99})
    b = _doc(group_counts={"granularity": 2}, pair_counts={"granularity": 99})
    res = compare_conflict_files(a, b)
    assert res["unit"] == "group"
    assert _counts(res, "granularity") == (5, 2)


def test_pair_unit_is_still_available():
    a = _doc(group_counts={"granularity": 5}, pair_counts={"granularity": 99})
    b = _doc(group_counts={"granularity": 2}, pair_counts={"granularity": 40})
    res = compare_conflict_files(a, b, unit="pair")
    assert res["unit"] == "pair"
    assert _counts(res, "granularity") == (99, 40)


def test_group_counts_are_derived_when_only_groups_are_present():
    """Older files carry `groups` without the summary counts."""
    groups = [{"conflict_label": "granularity"}, {"conflict_label": "granularity"},
              {"conflict_label": "redundancy"}]
    res = compare_conflict_files(_doc(groups=groups), _doc(groups=groups[:1]))
    assert _counts(res, "granularity") == (2, 1)


def test_file_without_group_counts_reports_unavailable_not_zeros():
    """Silently reporting zeros would read as 'no conflicts of any kind'."""
    a = _doc(pair_counts={"granularity": 5})          # pre-group-layer file
    b = _doc(group_counts={"granularity": 2})
    res = compare_conflict_files(a, b, v1_label="old")
    assert res["available"] is False
    assert "old" in res["error"]
    # ...and the pair unit still works on the same pair of files.
    assert compare_conflict_files(a, b, unit="pair")["available"] is True


def test_unknown_unit_is_rejected():
    with pytest.raises(ValueError):
        compare_conflict_files(_doc(group_counts={}), _doc(group_counts={}),
                               unit="sentence")


# --- 2. no_conflict must be counted ----------------------------------------

def test_no_conflict_is_a_reported_label():
    assert "no_conflict" in CONFLICT_LABELS
    a = _doc(group_counts={"no_conflict": 7, "redundancy": 1})
    b = _doc(group_counts={"no_conflict": 2, "redundancy": 1})
    res = compare_conflict_files(a, b)
    assert _counts(res, "no_conflict") == (7, 2)
    assert res["totals"]["v1"] == 8       # not 1 — no_conflict is in the total
    assert res["totals"]["delta"] == -5


# --- 3. escalate is surfaced, not silently dropped --------------------------

def test_escalate_is_reported_as_unresolved_and_warns():
    a = _doc(group_counts={"redundancy": 3, "escalate": 12})
    b = _doc(group_counts={"redundancy": 3, "escalate": 0})
    res = compare_conflict_files(a, b)
    assert res["unresolved"]["label"] == UNRESOLVED_LABEL
    assert res["unresolved"]["v1"] == 12 and res["unresolved"]["v2"] == 0
    assert res["unresolved"]["warning"]                    # loud
    assert UNRESOLVED_LABEL not in res["labels"]           # not a conflict type
    assert res["totals"]["v1"] == 3                        # excluded from totals


def test_no_warning_when_everything_resolved():
    d = _doc(group_counts={"redundancy": 3})
    assert compare_conflict_files(d, d)["unresolved"]["warning"] is None


def test_private_keys_never_leak_into_the_distribution():
    """`arbitrate_groups` returns bookkeeping under `_layer2_calls`."""
    d = _doc(group_counts={"redundancy": 3, "_layer2_calls": 9})
    res = compare_conflict_files(d, d)
    assert res["totals"]["v1"] == 3


# --- layer-1 rate follows the unit -----------------------------------------

def test_layer1_rate_uses_arbitrable_groups_on_the_group_unit():
    """A 1:n group is settled by its shape and an orphan has no partner, so
    neither is ever billed to Layer-2 and neither belongs in the denominator."""
    groups = ([{"conflict_label": "redundancy", "shape": "one_to_one"}] * 10 +
              [{"conflict_label": "granularity", "shape": "one_to_many"}] * 30 +
              [{"conflict_label": "unlabeled", "shape": "orphan"}] * 60)
    d = _doc(groups=groups, group_layer2_calls=4)
    rate = compare_conflict_files(d, d)["layer1_rate"]["v1"]
    assert rate["n_arbitrable_groups"] == 10
    assert rate["layer1_filter_rate"] == 0.6      # 4 of 10 escalated


def test_layer1_rate_on_the_pair_unit_is_unchanged():
    pairs = [{"status": "matched"}] * 8 + [{"status": "unmatched_a"}] * 2
    d = _doc(pair_counts={"redundancy": 8}, pairs=pairs, layer2_calls=2)
    rate = compare_conflict_files(d, d, unit="pair")["layer1_rate"]["v1"]
    assert rate["n_matched_pairs"] == 8
    assert rate["layer1_filter_rate"] == 0.75
