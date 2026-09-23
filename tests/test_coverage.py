"""Tests for src.coverage."""

from __future__ import annotations

from src.coverage import compute_coverage, _merge_intervals, _container_for_section


def test_merge_intervals_overlapping():
    assert _merge_intervals([(0, 10), (5, 20), (30, 40)]) == [(0, 20), (30, 40)]


def test_merge_intervals_empty():
    assert _merge_intervals([]) == []


def test_merge_intervals_drops_degenerate():
    # zero-length spans add nothing
    assert _merge_intervals([(5, 5), (10, 12)]) == [(10, 12)]


def test_container_for_section():
    assert _container_for_section("preamble.recitals[3]") == "preamble"
    assert _container_for_section("enacting.article_1") == "enacting"
    assert _container_for_section("garbage") is None
    assert _container_for_section("") is None


def test_compute_coverage_basic(fact_factory):
    parsed = {
        "document": {
            "recitals": [{"number": "(1)", "text": "abc"}, {"number": "(2)", "text": "def"}],
            "articles": [{"number": "1", "paragraphs": ["p"]}],
            "preamble_text": "x" * 100,
            "enacting_text": "y" * 50,
        }
    }
    facts = [
        fact_factory("a", "A", "s", "p", "o", sec="preamble.recitals[0]", cs=0, ce=30),
        fact_factory("b", "A", "s", "p", "o", sec="preamble.recitals[1]", cs=40, ce=60),
    ]
    cov = compute_coverage({"A": facts}, parsed)
    assert cov["A"]["n_facts"] == 2
    # 2 of 3 sections hit
    assert cov["A"]["sections_hit_frac"] == round(2/3, 3)
    # 50 chars covered out of 150 available
    assert cov["A"]["char_coverage_frac"] == round(50/150, 3)


def test_compute_coverage_no_double_count_on_overlap(fact_factory):
    parsed = {
        "document": {
            "recitals": [{"number": "(1)"}],
            "articles": [],
            "preamble_text": "x" * 100,
            "enacting_text": "",
        }
    }
    facts = [
        fact_factory("a", "A", "s", "p", "o", sec="preamble.recitals[0]", cs=0, ce=50),
        fact_factory("b", "A", "s", "p", "o", sec="preamble.recitals[0]", cs=30, ce=70),
    ]
    cov = compute_coverage({"A": facts}, parsed)
    # Union is [0, 70] = 70 chars
    assert cov["A"]["total_chars_covered"] == 70
