"""Tests for src.kg_build — graph construction + severity propagation."""

from __future__ import annotations

from src.kg_build import build_graph, _strongest_label


def test_strongest_label_picks_highest_severity():
    assert _strongest_label(["redundancy", "contradiction"]) == "contradiction"
    assert _strongest_label(["granularity", "redundancy"]) == "granularity"
    assert _strongest_label(["unlabeled"]) == "unlabeled"
    assert _strongest_label([]) == "unlabeled"


def _make_conflicts_doc(fact_factory):
    fa = fact_factory("a", "A", "the Council", "decided", "X exists",
                      sec="preamble.recitals[0]", cs=0, ce=30)
    fb = fact_factory("b", "B", "the Council", "decided", "X exists",
                      sec="preamble.recitals[0]", cs=0, ce=30)
    return {
        "doc_id": "test-doc",
        "facts_per_annotator": {"A": [fa], "B": [fb]},
        "entity_clusters": {
            "the Council": "ent_council_001",
            "X exists": "ent_x_002",
        },
        "aligned_pairs": [{
            "annotator_a": "A", "annotator_b": "B",
            "fact_a_id": "a", "fact_b_id": "b",
            "cosine": 0.99, "status": "matched",
            "conflict_label": "redundancy",
            "layer1_reason": "test", "layer2_reason": None,
        }],
    }


def test_build_graph_collapses_same_triple_across_annotators(fact_factory):
    doc = _make_conflicts_doc(fact_factory)
    g = build_graph(doc)
    # Both annotators emit the same triple — one edge.
    assert g["summary"]["n_edges"] == 1
    edge = g["edges"][0]
    assert set(edge["data"]["annotators"]) == {"A", "B"}
    assert edge["data"]["conflict_label"] == "redundancy"


def test_build_graph_propagates_severity_to_nodes(fact_factory):
    doc = _make_conflicts_doc(fact_factory)
    # Bump the pair to contradiction — both endpoints should pick it up.
    doc["aligned_pairs"][0]["conflict_label"] = "contradiction"
    g = build_graph(doc)
    labels = {n["data"]["id"]: n["data"]["conflict_label"] for n in g["nodes"]}
    assert all(v == "contradiction" for v in labels.values()), labels


def test_build_graph_skips_null_object(fact_factory):
    fa = fact_factory("a", "A", "Decision X", "is hereby abrogated", "null",
                      sec="enacting.article_2")
    doc = {
        "doc_id": "test-doc",
        "facts_per_annotator": {"A": [fa]},
        "entity_clusters": {"Decision X": "ent_decX_001"},
        "aligned_pairs": [],
    }
    g = build_graph(doc)
    # "null" object → no edge can be formed; the subject node still exists.
    assert g["summary"]["n_edges"] == 0
    assert g["summary"]["n_nodes"] == 1
