"""The reply checker gates what reaches the second annotation, so its errors
must stay errors. The duplicate-fact check in particular is the guard for the
2026-08-27 failure: one label used for every row of a flattened table, which
asserts the same triple twice and merges rows the source keeps apart."""

import pytest

from scripts.check_annotator_reply import check
from src import gold_store as gs

SID = "train-000019::enacting.article_2"


def _segment_text():
    m = gs.sample_index()[SID]
    return gs.segment_text(m["doc_id"], m["section_path"], m["char_start"], m["char_end"])


def fact(quote, s, p, o, **kw):
    f = {"natural_language": f"{s} {p} {o}.", "source_quote": quote,
         "condition": "", "temporal_context": "", "logic_group": "", "logic_op": "",
         "boilerplate": False,
         "triples": [{"subject": s, "predicate": p, "object": o}]}
    f.update(kw)
    return f


@pytest.fixture
def row_quotes():
    import re
    text = _segment_text()
    marks = [m.start() for m in
             re.finditer(r"In the table set out in this Article, row \d+:", text)]
    assert len(marks) >= 2, "this test needs a flattened table with >= 2 rows"
    return [text[a: (marks[i + 1] if i + 1 < len(marks) else len(text))].strip()
            for i, a in enumerate(marks)]


def test_clean_reply_has_no_errors(row_quotes):
    facts = [fact(row_quotes[0], "the first entry for U", "specifies code", "B"),
             fact(row_quotes[1], "the second entry for U", "specifies code", "B")]
    errors, _ = check(SID, facts)
    assert errors == []


def test_same_triple_twice_is_an_error(row_quotes):
    """Two rows collapsed under one label. Distinct quotes do not make it two facts."""
    facts = [fact(row_quotes[0], "the entry for U", "specifies code", "B"),
             fact(row_quotes[1], "the entry for U", "specifies code", "B")]
    errors, _ = check(SID, facts)
    assert any("same triple set" in e for e in errors)


def test_quote_not_verbatim_is_an_error(row_quotes):
    facts = [fact("a sentence that is not in this segment at all",
                  "the first entry for U", "specifies code", "B")]
    errors, _ = check(SID, facts)
    assert any("NOT verbatim" in e for e in errors)


def test_empty_slot_is_an_error(row_quotes):
    facts = [fact(row_quotes[0], "the first entry for U", "specifies code", "  ")]
    errors, _ = check(SID, facts)
    assert any("empty object" in e for e in errors)


def test_logic_group_of_one_is_an_error(row_quotes):
    facts = [fact(row_quotes[0], "the first entry for U", "covers", "Sorghum bicolor",
                  logic_group="row0", logic_op="AND")]
    errors, _ = check(SID, facts)
    assert any("single member" in e for e in errors)


def test_short_quote_warns_but_does_not_error(row_quotes):
    """A bare cell value is a bad anchor, not a malformed fact."""
    facts = [fact("B", "the first entry for U", "specifies code", "B")]
    errors, warnings = check(SID, facts)
    assert errors == []
    assert any("too short to anchor" in w for w in warnings)


def test_a_row_used_as_a_node_warns(row_quotes):
    facts = [fact(row_quotes[0], "row 0", "is in", "the table")]
    _, warnings = check(SID, facts)
    assert any("flattened row" in w for w in warnings)


def test_a_table_the_source_names_is_not_flagged(row_quotes):
    """Rule 11 forbids the ROW as a node. `the table in Part I (2)` is a real
    legal object the source text names, and must not be reported."""
    facts = [fact(row_quotes[0], "the following item", "is inserted in",
                  "the table in Part I (2)")]
    _, warnings = check(SID, facts)
    assert not any("flattened row" in w for w in warnings)
