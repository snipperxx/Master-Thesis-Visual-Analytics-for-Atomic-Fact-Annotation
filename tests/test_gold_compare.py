"""Regression tests for WebNLG gold-standard comparison (src/gold_compare)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import gold_compare as gc


def test_normalisation_camel_and_articles():
    assert gc.normalize_term("Aarhus_Airport") == "aarhus airport"
    assert gc.normalize_term('"Aarhus, Denmark"') == "aarhus denmark"
    assert gc.normalize_term("the Netherlands") == "netherlands"
    assert gc.normalize_predicate("cityServed") == "city served"
    assert gc.normalize_predicate("elevationAboveTheSeaLevel") == "elevation above the sea level"


def test_strict_match_pr_f1():
    gold = [
        {"subject": "Aarhus_Airport", "predicate": "cityServed", "object": "Aarhus, Denmark"},
        {"subject": "Aarhus_Airport", "predicate": "runwayLength", "object": "2776.0"},
    ]
    extracted = [
        {"subject": "Aarhus Airport", "predicate": "city served", "object": "Aarhus Denmark", "fact_id": "f1"},
        {"subject": "Aarhus Airport", "predicate": "serves the city of", "object": "Aarhus, Denmark", "fact_id": "f2"},
    ]
    r = gc.compare(gold, extracted)
    m = r["metrics"]
    assert m["n_gold"] == 2 and m["n_extracted"] == 2
    assert m["n_matched"] == 1 and m["n_missing"] == 1 and m["n_spurious"] == 1
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert r["matched_fact_ids"] == ["f1"]
    assert r["spurious_fact_ids"] == ["f2"]


def test_empty_extraction_is_all_missing():
    gold = [{"subject": "Aarhus_Airport", "predicate": "cityServed", "object": "Aarhus, Denmark"}]
    r = gc.compare(gold, [])
    assert r["metrics"]["n_matched"] == 0 and r["metrics"]["n_missing"] == 1
    assert r["metrics"]["recall"] == 0.0


def test_gold_graph_shape():
    gold = [{"subject": "Aarhus_Airport", "predicate": "cityServed", "object": "Aarhus, Denmark"}]
    g = gc.gold_graph(gold)
    assert g["summary"]["n_nodes"] == 2 and g["summary"]["n_edges"] == 1
    assert all(n["data"].get("gold") for n in g["nodes"])


# ---------------------------------------------------------------------------
# Soft (semantic) matching — regression guards for the 2026-07-02 additions.
# Backend-agnostic assertions only: exact-normalised strings force sim=1.0 and
# numeric slots short-circuit, so these hold under SBERT *and* TF-IDF.
# ---------------------------------------------------------------------------

def test_soft_numeric_short_circuit():
    gold = [{"subject": "A_Airport", "predicate": "elevationAboveTheSeaLevel",
             "object": "25.0"}]
    right = [{"subject": "A Airport", "predicate": "elevation above the sea level",
              "object": "25 metres", "fact_id": "f1"}]
    wrong = [{"subject": "A Airport", "predicate": "elevation above the sea level",
              "object": "4349 metres", "fact_id": "f2"}]
    assert gc.compare_soft(gold, right)["metrics"]["n_matched"] == 1
    r = gc.compare_soft(gold, wrong)
    assert r["metrics"]["n_matched"] == 0            # different number must not match
    assert r["missing"][0]["reason"].endswith("object_mismatch")


def test_soft_exact_slots_match_and_reasons():
    gold = [{"subject": "Alan_Bean", "predicate": "birthPlace", "object": "Wheeler, Texas"},
            {"subject": "Alan_Bean", "predicate": "was_a_crew_member_of", "object": "Apollo 12"}]
    extracted = [{"subject": "Alan Bean", "predicate": "birth place",
                  "object": "Wheeler Texas", "fact_id": "f1"}]
    r = gc.compare_soft(gold, extracted)
    assert r["metrics"]["n_matched"] == 1 and r["metrics"]["n_missing"] == 1
    assert r["matched_fact_ids"] == ["f1"]
    reasons = {m.get("reason") for m in r["missing"]}
    assert reasons and all(x for x in reasons)       # every missing carries a reason
    assert r["backend"] in ("sbert", "tfidf")


def test_soft_one_to_one_ablation_keeps_greedy_exclusivity():
    # The pre-2026-09-23 behaviour, now reachable only as an explicit ablation.
    gold = [{"subject": "X", "predicate": "locatedIn", "object": "Y"}]
    extracted = [
        {"subject": "X", "predicate": "located in", "object": "Y", "fact_id": "f1"},
        {"subject": "X", "predicate": "located in area of", "object": "Y", "fact_id": "f2"},
    ]
    r = gc.compare_soft(gold, extracted, assignment="one_to_one")
    assert r["assignment"] == "one_to_one"
    assert r["metrics"]["n_matched"] == 1            # one gold can absorb only one pred
    assert r["metrics"]["n_spurious"] == 1
    assert r["spurious"][0]["reason"] == "gold_already_matched"


def test_soft_default_credits_both_predictions_of_one_gold():
    # Same input under the design of record: nothing is exclusive.
    gold = [{"subject": "X", "predicate": "locatedIn", "object": "Y"}]
    extracted = [
        {"subject": "X", "predicate": "located in", "object": "Y", "fact_id": "f1"},
        {"subject": "X", "predicate": "located in area of", "object": "Y", "fact_id": "f2"},
    ]
    r = gc.compare_soft(gold, extracted)
    assert r["assignment"] == "many_to_many"
    m = r["metrics"]
    assert m["n_gold_covered"] == 1 and m["n_pred_supported"] == 2
    assert m["n_spurious"] == 0 and r["spurious"] == []
    assert m["precision"] == 1.0 and m["recall"] == 1.0


def test_compare_full_carries_both_modes():
    gold = [{"subject": "Berlin", "predicate": "capitalOf", "object": "Germany"}]
    r = gc.compare_full(gold, [{"subject": "Berlin", "predicate": "capital of",
                                "object": "Germany", "fact_id": "f1"}])
    assert r["strict"]["metrics"]["f1"] == 1.0
    assert r["soft"]["metrics"]["f1"] == 1.0
    assert r["backend"] in ("sbert", "tfidf")


# ---------------------------------------------------------------------------
# Many-to-many coverage (design of record, 2026-09-23).
#
# These drive `compare_soft` with hand-set slot similarities, so they test the
# counting and nothing else: which pairs are links is fixed by the fixture, not
# by an encoder, and the assertions hold under SBERT and TF-IDF alike. The
# thresholds are passed explicitly for the same reason. Slot values are chosen
# away from the gates (float32 0.70 < 0.70, for instance).
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402
import pytest  # noqa: E402

_SLOTS = ("subject", "predicate", "object")
_TH = dict(gc.DEFAULT_THRESHOLDS)


def _items(n: int, tag: str) -> list[dict]:
    """n triples whose normalised keys are pairwise distinct (so dedup keeps
    all of them) and whose fact_id is the tag plus the index."""
    return [{"subject": f"{tag}{i} subj", "predicate": f"{tag}{i} pred",
             "object": f"{tag}{i} obj", "fact_id": f"{tag}{i}"} for i in range(n)]


def _fake_sims(monkeypatch, ng: int, ne: int, pairs: dict) -> None:
    """Every slot 0.1 except `pairs`: {(gi, ei): value or (s, p, o)}."""
    sims = {sl: np.full((ng, ne), 0.1, dtype=np.float32) for sl in _SLOTS}
    for (gi, ei), v in pairs.items():
        vals = v if isinstance(v, tuple) else (v, v, v)
        for sl, x in zip(_SLOTS, vals):
            sims[sl][gi, ei] = x
    monkeypatch.setattr(gc, "_slot_sims",
                        lambda g, e: {k: a.copy() for k, a in sims.items()})


def _soft(gold, pred, **kw):
    return gc.compare_soft(gold, pred, _TH, **kw)


def test_m2m_merged_prediction_covers_every_gold_it_links(monkeypatch):
    """(a) One prediction clearing the gates against three gold triples covers
    all three, is supported, and costs nothing in precision."""
    _fake_sims(monkeypatch, 3, 1, {(0, 0): 0.9, (1, 0): 0.8, (2, 0): 0.75})
    r = _soft(_items(3, "g"), _items(1, "p"))
    m = r["metrics"]
    assert m["n_gold_covered"] == 3 and m["recall"] == 1.0
    assert m["n_pred_supported"] == 1 and m["precision"] == 1.0
    assert m["n_links"] == 3 and m["n_pred_multi_gold"] == 1
    assert r["missing"] == [] and r["spurious"] == []
    assert [x["gold"][0] for x in r["matched"]] == ["g0 subj", "g1 subj", "g2 subj"]
    # The penalty the design removes: the ablation lets the merged prediction
    # absorb one gold triple and charges the other two as missed.
    old = _soft(_items(3, "g"), _items(1, "p"), assignment="one_to_one")["metrics"]
    assert old["n_gold_covered"] == 1 and old["recall"] == pytest.approx(1 / 3, abs=1e-4)


def test_m2m_split_as_the_gold_is_credited_identically(monkeypatch):
    """(b) Three predictions matching the gold's own split score exactly what
    one merged prediction covering the same three scores."""
    _fake_sims(monkeypatch, 3, 3, {(0, 0): 0.9, (1, 1): 0.9, (2, 2): 0.9})
    split = _soft(_items(3, "g"), _items(3, "p"))["metrics"]
    _fake_sims(monkeypatch, 3, 1, {(0, 0): 0.9, (1, 0): 0.9, (2, 0): 0.9})
    merged = _soft(_items(3, "g"), _items(1, "p"))["metrics"]
    for k in ("precision", "recall", "f1", "n_gold_covered", "n_gold"):
        assert split[k] == merged[k], k
    assert split["precision"] == split["recall"] == 1.0
    assert split["n_pred_multi_gold"] == 0 and merged["n_pred_multi_gold"] == 1


def test_m2m_unrelated_prediction_is_not_supported(monkeypatch):
    """(c) A prediction with no link is unsupported and keeps its failure
    reason: no counterpart at all, or the slot gate it missed."""
    _fake_sims(monkeypatch, 2, 3, {(0, 0): 0.9,
                                    (0, 2): (0.5, 0.9, 0.9)})   # subject gate miss
    r = _soft(_items(2, "g"), _items(3, "p"))
    m = r["metrics"]
    assert m["n_pred_supported"] == 1 and m["precision"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["n_gold_covered"] == 1 and m["recall"] == 0.5
    reasons = {tuple(s["fact_ids"]): s["reason"] for s in r["spurious"]}
    assert reasons == {("p1",): "no_gold_counterpart", ("p2",): "subject_mismatch"}
    assert [x["reason"] for x in r["missing"]] == ["no_candidate"]
    assert r["matched_fact_ids"] == ["p0"]
    assert r["spurious_fact_ids"] == ["p1", "p2"]


def test_m2m_two_predictions_of_one_gold_count_it_once(monkeypatch):
    """(d) Both predictions are supported; the gold triple counts once in
    recall. Reading precision off the covered-gold count would give 0.5."""
    _fake_sims(monkeypatch, 1, 2, {(0, 0): 0.9, (0, 1): 0.8})
    r = _soft(_items(1, "g"), _items(2, "p"))
    m = r["metrics"]
    assert m["n_gold"] == 1 and m["n_gold_covered"] == 1 and m["recall"] == 1.0
    assert m["n_extracted"] == 2 and m["n_pred_supported"] == 2
    assert m["precision"] == 1.0
    assert m["n_gold_multi_pred"] == 1 and m["n_links"] == 2
    assert r["matched_fact_ids"] == ["p0", "p1"]


def test_m2m_two_predictions_of_one_gold_real_strings():
    """(d) again, through the real `_slot_sims`. Backend-independent: subject
    and predicate are equal after normalisation (forced 1.0), and the objects
    carry the gold's bare number (numeric_override forces 1.0)."""
    gold = [{"subject": "the export refund", "predicate": "is fixed at", "object": "25"}]
    pred = [{"subject": "Export refund", "predicate": "shall be fixed at",
             "object": "25 EUR", "fact_id": "f1"},
            {"subject": "export refund", "predicate": "fixed at",
             "object": "25 euro", "fact_id": "f2"}]
    m = gc.compare_soft(gold, pred)["metrics"]
    assert (m["n_gold"], m["n_gold_covered"], m["n_extracted"], m["n_pred_supported"]) \
        == (1, 1, 2, 2)
    assert m["precision"] == 1.0 and m["recall"] == 1.0
    m1 = gc.compare_soft(gold, pred, assignment="one_to_one")["metrics"]
    assert m1["n_pred_supported"] == 1 and m1["precision"] == 0.5


def test_m2m_is_the_default_and_unknown_modes_are_refused():
    assert gc.DEFAULT_ASSIGNMENT == "many_to_many"
    assert set(gc.ASSIGNMENTS) == {"many_to_many", "one_to_one"}
    with pytest.raises(ValueError):
        gc.compare_soft([], [], assignment="one-to-one")    # misspelt
    with pytest.raises(ValueError):
        gc.compare_full([], [], assignment="hungarian")


def test_strict_counts_are_assignment_invariant():
    """String equality links a gold key to at most the identical predicted key,
    so the three strict counts coincide."""
    gold = [{"subject": "Council", "predicate": "adopts", "object": "Decision"},
            {"subject": "Commission", "predicate": "fixes", "object": "refund"}]
    pred = [{"subject": "council", "predicate": "adopts", "object": "decision",
             "fact_id": "f1"},
            {"subject": "the Council", "predicate": "shall adopts",
             "object": "Decision", "fact_id": "f2"},
            {"subject": "zebra", "predicate": "eats", "object": "grass", "fact_id": "f3"}]
    m = gc.compare(gold, pred)["metrics"]
    # f1 and f2 normalise to the same key: one predicted unit.
    assert m["n_extracted"] == 2
    assert m["n_matched"] == m["n_gold_covered"] == m["n_pred_supported"] == m["n_links"] == 1


# (e) The ablation must reproduce the pre-2026-09-23 matcher exactly. The
# expected values below were produced by the OLD `compare_soft` (before the
# assignment parameter existed) on this very fixture, on 2026-09-23, and are
# frozen here. The fixture exercises every greedy decision: a merged prediction
# that wins its first gold and is then unavailable (-> prediction_already_matched),
# a gold taken by a better partner (-> gold_already_matched), a slot-gate miss
# and a no-counterpart prediction.

def _legacy_fixture(monkeypatch):
    _fake_sims(monkeypatch, 3, 5, {(0, 0): 0.9, (1, 0): 0.8, (2, 0): 0.75,
                                    (1, 1): 0.85, (0, 2): (0.5, 0.9, 0.9),
                                    (1, 4): 0.72})
    return _items(3, "g"), _items(5, "p")


_LEGACY_METRICS = {"precision": 0.4, "recall": 0.6667, "f1": 0.5,
                   "n_gold": 3, "n_extracted": 5, "n_matched": 2,
                   "n_missing": 1, "n_spurious": 3}


def test_one_to_one_ablation_reproduces_the_old_matcher(monkeypatch):
    gold, pred = _legacy_fixture(monkeypatch)
    r = _soft(gold, pred, assignment="one_to_one")
    assert {k: r["metrics"][k] for k in _LEGACY_METRICS} == _LEGACY_METRICS
    assert [(x["gold"][0], x["pred"][0], x["sims"]["overall"]) for x in r["matched"]] \
        == [("g0 subj", "p0 subj", 0.9), ("g1 subj", "p1 subj", 0.85)]
    assert [(x["triple"][0], x["reason"], x["best"]["pred"][0]) for x in r["missing"]] \
        == [("g2 subj", "prediction_already_matched", "p0 subj")]
    assert [(x["fact_ids"], x["reason"]) for x in r["spurious"]] == [
        (["p2"], "subject_mismatch"), (["p3"], "no_gold_counterpart"),
        (["p4"], "gold_already_matched")]
    assert r["spurious"][0]["best_gold"]["sims"] == {
        "subject": 0.5, "predicate": 0.9, "object": 0.9, "overall": 0.76}
    assert r["matched_fact_ids"] == ["p0", "p1"]
    assert r["spurious_fact_ids"] == ["p2", "p3", "p4"]
    # In one_to_one the new counts carry no new information.
    m = r["metrics"]
    assert m["n_gold_covered"] == m["n_pred_supported"] == m["n_links"] == 2
    assert m["n_pred_multi_gold"] == m["n_gold_multi_pred"] == 0


def test_many_to_many_on_the_legacy_fixture(monkeypatch):
    """The same fixture under the design of record, for contrast."""
    gold, pred = _legacy_fixture(monkeypatch)
    r = _soft(gold, pred)
    m = r["metrics"]
    assert m["n_links"] == 5
    assert m["n_gold_covered"] == 3 and m["recall"] == 1.0
    assert m["n_pred_supported"] == 3 and m["precision"] == 0.6    # p0, p1, p4
    assert r["missing"] == []
    assert [(x["fact_ids"], x["reason"]) for x in r["spurious"]] == [
        (["p2"], "subject_mismatch"), (["p3"], "no_gold_counterpart")]
    # `*_already_matched` is an artefact of exclusivity and must not survive it:
    # g2 and p4 carried those labels under the ablation and are now credited.
    assert not any("already_matched" in x["reason"] for x in r["spurious"])
