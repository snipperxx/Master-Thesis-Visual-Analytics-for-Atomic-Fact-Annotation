"""Tests for src.annotator_compare (heatmap matrix + IAA summary)."""

from src.annotator_compare import iaa_summary, similarity_matrix


def _fact(fid, text, subj="s", pred="p", obj="o", section="preamble.recitals[0]"):
    return {
        "fact_id": fid, "natural_language": text,
        "subject": subj, "predicate": pred, "object": obj,
        "source_locator": {"section_path": section, "quote": text},
    }


FACTS_A = [
    _fact("a1", "The Council decided that an excessive deficit exists in the Netherlands."),
    _fact("a2", "The deficit reached 3,2 % of GDP in 2003."),
    _fact("a3", "The Commission issued a recommendation on 2 June 2004."),
]
FACTS_B = [
    _fact("b1", "The Council decided that an excessive deficit exists in the Netherlands."),
    _fact("b2", "The general government deficit reached 3,2 % of GDP in 2003."),
]


def test_matrix_shape_and_range():
    out = similarity_matrix(FACTS_A, FACTS_B, annotator_a="A", annotator_b="B")
    assert len(out["matrix"]) == 3
    assert all(len(row) == 2 for row in out["matrix"])
    assert all(-1.001 <= v <= 1.001 for row in out["matrix"] for v in row)
    assert out["annotator_a"] == "A" and out["annotator_b"] == "B"
    assert len(out["facts_a"]) == 3 and len(out["facts_b"]) == 2


def test_identical_texts_match():
    out = similarity_matrix(FACTS_A, FACTS_B, annotator_a="A", annotator_b="B",
                            threshold=0.78)
    matched = {(m["fact_a_id"], m["fact_b_id"]) for m in out["matched"]}
    assert ("a1", "b1") in matched  # verbatim identical sentence
    for m in out["matched"]:
        i, j = m["ia"], m["ib"]
        assert abs(out["matrix"][i][j] - m["cosine"]) < 0.05


def test_persisted_pairs_take_precedence_and_handle_reversed_orientation():
    pairs = [{
        "annotator_a": "B", "annotator_b": "A",     # reversed on purpose
        "fact_a_id": "b2", "fact_b_id": "a2",
        "cosine": 0.91, "status": "matched",
        "conflict_label": "granularity",
        "layer1_reason": None, "layer2_reason": "stub",
    }]
    out = similarity_matrix(FACTS_A, FACTS_B, annotator_a="A", annotator_b="B",
                            aligned_pairs=pairs)
    assert len(out["matched"]) == 1
    m = out["matched"][0]
    assert (m["fact_a_id"], m["fact_b_id"]) == ("a2", "b2")
    assert m["ia"] == 1 and m["ib"] == 1
    assert m["conflict_label"] == "granularity"


def test_empty_side():
    out = similarity_matrix([], FACTS_B, annotator_a="A", annotator_b="B")
    assert out["matrix"] == []
    assert out["matched"] == []


def test_iaa_summary_jaccard():
    confl = {
        "annotators": ["A", "B"],
        "facts_per_annotator": {"A": FACTS_A, "B": FACTS_B + [_fact("b3", "extra")]},
        "aligned_pairs": [
            {"annotator_a": "A", "annotator_b": "B", "fact_a_id": "a1",
             "fact_b_id": "b1", "cosine": 1.0, "status": "matched",
             "conflict_label": "redundancy"},
            {"annotator_a": "A", "annotator_b": "B", "fact_a_id": "a2",
             "fact_b_id": "b2", "cosine": 0.9, "status": "matched",
             "conflict_label": "granularity"},
            {"annotator_a": "A", "annotator_b": "B", "fact_a_id": "a3",
             "fact_b_id": None, "cosine": 0.4, "status": "unmatched_a",
             "conflict_label": "unlabeled"},
        ],
    }
    out = iaa_summary(confl)
    assert out["counts"] == {"A": 3, "B": 3}
    [pair] = out["pairs"]
    assert pair["n_matched"] == 2
    # Jaccard = 2 / (3 + 3 - 2) = 0.5
    assert pair["jaccard"] == 0.5
    assert abs(pair["mean_cosine"] - 0.95) < 1e-6
    assert pair["labels"] == {"redundancy": 1, "granularity": 1}
