"""The machine pre-review must not cry wolf.

Two ways it did. Both cost the same thing — a reviewer reading a badge that
means nothing and learning to stop reading badges — so both get a guard here.

1. **It read `triples[0]` only.** After the high-resolution pass (D-18) a fact
   holds a chain, and a chain-blind check calls two facts duplicates because
   their first links agree, misses everything the other links say, and reports
   an object as unanchored because the fact's own short quote does not repeat
   what an earlier link in the same chain introduced.

2. **It re-emitted checks the decision log had already settled.** D-13 rejected
   the boilerplate flag on entry-into-force / application-date / addressee
   clauses; D-14 accepted the reworded qualifier slots. Eleven and twelve of the
   44 warnings on `review_facts_hires.html` were those two, asking Zilong to
   re-decide what he had a written decision for. `SETTLED` is the guard: a check
   listed there is retired on purpose and must stay retired.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.machine_review_gold import SETTLED, _norm, check_fact  # noqa: E402

TEXT = ("The Commission shall extend the time-limits by 24 months. "
        "The time-limits are referred to in Article 7(2) of Directive 2002/96/EC. "
        "This Regulation shall enter into force on 14 November 2003.")


def _fact(gold_id="d::s::f01", triples=None, quote="The Commission shall extend",
          **kw) -> dict:
    return {"gold_id": gold_id, "natural_language": "x", "source_quote": quote,
            "condition": "", "temporal_context": "", "logic_group": "",
            "logic_op": "", "boilerplate": False,
            "triples": triples or [{"subject": "the Commission",
                                    "predicate": "shall extend",
                                    "object": "the time-limits"}], **kw}


def _codes(fact, facts=None, nodes=None, text=TEXT):
    facts = facts if facts is not None else [fact]
    if nodes is None:                       # the same inventory `review_root` builds
        nodes = {_norm(t[k]) for f in facts for t in f["triples"]
                 for k in ("subject", "object")} - {""}
    return [c for _, c, _ in check_fact(fact, text, facts, nodes)]


# --- 1. chains ------------------------------------------------------------

def test_same_first_triple_different_chain_is_not_a_duplicate():
    """The bug that produced 21 spurious `duplicate_triple` errors the moment the
    checker was pointed at facts_hires."""
    a = _fact("d::s::f01", [{"subject": "a", "predicate": "p", "object": "b"},
                            {"subject": "b", "predicate": "q", "object": "c"}])
    b = _fact("d::s::f02", [{"subject": "a", "predicate": "p", "object": "b"},
                            {"subject": "b", "predicate": "r", "object": "z"}])
    assert "duplicate_fact" not in _codes(a, [a, b])
    assert "duplicate_fact" not in _codes(b, [a, b])


def test_an_identical_chain_is_still_a_duplicate():
    chain = [{"subject": "a", "predicate": "p", "object": "b"},
             {"subject": "b", "predicate": "q", "object": "c"}]
    a, b = _fact("d::s::f01", [dict(t) for t in chain]), _fact("d::s::f02", chain)
    assert "duplicate_fact" in _codes(a, [a, b])


def test_a_broken_chain_link_is_an_error():
    f = _fact(triples=[{"subject": "a", "predicate": "p", "object": "b"},
                       {"subject": "unrelated", "predicate": "q", "object": "c"}])
    assert "chain_does_not_connect" in _codes(f)


def test_nolink_declares_a_two_proposition_span():
    """train-000041::preamble.recitals[3]::f01 asserts 'the information must be
    disregarded' and 'the goods are not of marketable quality' from one span; they
    share no entity by nature. `build_gold_hires` knows that from the spec, so the
    fact carries the declaration and the checker must honour it — otherwise three
    deliberate facts sit at the top of the queue as errors forever."""
    f = _fact(triples=[{"subject": "a", "predicate": "p", "object": "b"},
                       {"subject": "unrelated", "predicate": "q", "object": "c"}],
              nolink=True)
    assert "chain_does_not_connect" not in _codes(f)


def test_every_triple_of_a_chain_is_checked_not_just_the_first():
    f = _fact(triples=[{"subject": "a", "predicate": "p", "object": "b"},
                       {"subject": "b", "predicate": "q", "object": ""}])
    assert "empty_slot" in _codes(f)


# --- 2. settled checks ----------------------------------------------------

def test_settled_checks_are_never_emitted():
    """A decision-log entry is the end of a question, not a suggestion."""
    f = _fact(quote="This Regulation shall enter into force on 14 November 2003",
              temporal_context="on 14 November 2003",
              condition="when the time-limits are extended")
    emitted = set(_codes(f))
    assert emitted.isdisjoint(SETTLED), sorted(emitted & set(SETTLED))


def test_the_closing_formula_is_still_expected_to_be_boilerplate():
    """D-13 narrowed the boilerplate check to D-03's contentless closing formula.
    Narrowed, not deleted."""
    f = _fact(quote="This Regulation shall be binding in its entirety")
    codes = _codes(f, text="This Regulation shall be binding in its entirety.")
    assert "closing_formula_not_boilerplate" in codes


def test_a_reworded_qualifier_passes_but_an_invented_one_does_not():
    """D-14: the qualifier slots may be reworded; they may not import content the
    segment does not carry."""
    ok = _fact(condition="when the time-limits are extended")
    assert "qualifier_adds_content" not in _codes(ok)
    bad = _fact(condition="when the harvest fails")
    assert "qualifier_adds_content" in _codes(bad)


# --- 3. R1b: an entity of the segment must not hide inside a predicate ----

def test_a_node_of_the_segment_inside_a_predicate_is_flagged():
    """`may extend the time-limits by` -> 24 months, while a sibling fact
    predicates OF `the time-limits`: one entity, two KG nodes (D-19)."""
    a = _fact("d::s::f01", [{"subject": "Cyprus",
                             "predicate": "may extend the time-limits by",
                             "object": "24 months"}])
    b = _fact("d::s::f02", [{"subject": "the time-limits",
                             "predicate": "are referred to in",
                             "object": "Article 7(2)"}])
    assert "node_split_into_predicate" in _codes(a, [a, b])


def test_a_relational_noun_stays_glue():
    """R1 permits `the application of` / `a decline in` / `the purpose of
    protecting`; flagging those was the `predicate_overloaded` noise."""
    for pred in ("concerns the application of", "projects a further decline in",
                 "have the purpose of protecting", "are incurred as a result of engaging"):
        f = _fact(triples=[{"subject": "x", "predicate": pred, "object": "y"}])
        codes = _codes(f, nodes=set())
        assert "predicate_holds_a_noun_phrase" not in codes, pred
