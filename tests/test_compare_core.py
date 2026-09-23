"""Regression guards for the shared comparison core (src/compare_core.py).

Every test here pins a defect that was live on 2026-08-27 — see
`doc/bugfix_log.md` BUG-2026-08-27-B..E. The guards matter because all four
were *silent*: none raised, none failed a test, each just produced a number
that looked plausible.

Hermetic: no SBERT, no Ollama, no disk.
"""

from __future__ import annotations

import pytest

from src import alignment, compare_core as cc
from src import conflict_layer1 as l1
from src import gold_compare


# ---------------------------------------------------------------------------
# BUG-2026-08-27-C — number tokenising and the identity test
# ---------------------------------------------------------------------------

class TestNumberTokenising:
    """EUR-Lex writes '1 000 000' and '37,5 %'. A tokeniser that reads digit
    runs separately turned those into {1, 0} and {37, 5}."""

    @pytest.mark.parametrize("text, expected", [
        ("EUR 1 000", {1000.0}),
        ("1 000 000 tonnes", {1000000.0}),
        ("1,000 tonnes", {1000.0}),
        ("37,5 %", {37.5}),
        ("1 234,50 EUR", {1234.5}),
        ("2.3% of GDP", {2.3}),
        ("ECU 20 per tonne", {20.0}),
    ])
    def test_grouped_thousands_and_decimal_comma(self, text, expected):
        assert cc.all_numbers(text) == expected

    def test_thousands_separator_is_not_read_as_a_decimal_point(self):
        # "1,000" is one thousand, not one. The rule keys on exactly three
        # following digits, so "37,5" stays a decimal.
        assert cc.all_numbers("1,000") == {1000.0}
        assert cc.all_numbers("37,5") == {37.5}


class TestNumericOverride:
    """RQ2 asks 'is this the same thing?'. The rule this replaced asked only
    whether the two number sets INTERSECTED, so any shared digit — a year, a
    zero from a shattered thousands group — scored a flat 1.0."""

    @pytest.mark.parametrize("a, b", [
        ("Decision 2005/136/EC", "Decision 2005/729/EC"),   # shared year 2005
        ("EUR 1 000 per 100 kg", "EUR 1 000 per 50 kg"),    # shared 1000
        ("1 000 000 tonnes", "2 000 000 tonnes"),           # shared 0 (old)
        ("Article 15(5)", "Article 15"),                    # shared 15
        ("2.3% of GDP", "2.8% of GDP"),
    ])
    def test_different_values_never_score_identical(self, a, b):
        assert cc.numeric_override(a, b) == 0.0

    def test_equal_numbers_alone_do_not_assert_identity(self):
        # Identical strings are already forced to 1.0 by the exact-normalised
        # override in gold_compare; the numeric rule must not claim it too.
        assert cc.numeric_override("ECU 20 per tonne", "ECU 20 per tonne") is None

    def test_no_numbers_defers_to_the_embedding(self):
        assert cc.numeric_override("the Council", "the Commission") is None
        assert cc.numeric_override("the Council", "ECU 20") is None

    def test_citations_are_not_scrubbed_here(self):
        # The citation number IS the entity's identity for RQ2: two different
        # acts whose surface strings an embedding scores at ~0.95 must not match.
        assert cc.numeric_override("Regulation (EC) No 1422/95",
                                   "Regulation (EC) No 785/68") == 0.0


class TestQuantityDivergence:
    """RQ1 asks the opposite question, so it scrubs citations — and it must
    stay a HINT, never a similarity veto (see the module docstring)."""

    def test_real_quantity_conflict_fires(self):
        assert cc.quantity_divergence("deficit is 2.3% of GDP",
                                      "deficit is 2.8% of GDP")

    def test_citation_difference_does_not_fire(self):
        assert not cc.quantity_divergence("Decision 2005/136/EC applies",
                                          "Decision 2005/729/EC applies")

    def test_one_sided_extra_is_coverage_not_contradiction(self):
        assert not cc.quantity_divergence("security is ECU 20",
                                          "security is ECU 20 per tonne")

    def test_grouped_thousands_conflict_is_now_visible(self):
        # Old tokeniser: {0, 1, 350} vs {0, 2, 350} — neither a subset of the
        # other, so this fired for the wrong reason. It must fire for the right
        # one now.
        assert cc.quantity_divergence("production 1 350 000 tonnes",
                                      "production 2 350 000 tonnes")


# ---------------------------------------------------------------------------
# BUG-2026-08-27-D — the scrubber's pattern order
# ---------------------------------------------------------------------------

class TestReferenceScrubbing:
    @pytest.mark.parametrize("citation", [
        "Decision 2005/136/EC",
        "Directive 66/401",
        "Decision 85/355/EEC",
        "Regulation (EEC) No 3897/89",
        "Article 15(5)",
    ])
    def test_a_citation_leaves_no_pseudo_quantity(self, citation):
        # Behind the keyword pattern the lazy \\d+ stopped at the first group,
        # so "Decision 2005/136/EC" kept 136 and two different Decisions stayed
        # numerically comparable.
        assert cc.content_numbers(citation) == set()

    @pytest.mark.parametrize("text, expected", [
        ("a quantity of 2 000 tonnes", {2000.0}),
        ("the security of ECU 20 per tonne", {20.0}),
        ("2.3% of GDP", {2.3}),
    ])
    def test_content_quantities_survive(self, text, expected):
        assert cc.content_numbers(text) == expected


# ---------------------------------------------------------------------------
# BUG-2026-08-27-B — the representation must be declared, not inferred
# ---------------------------------------------------------------------------

FLAT = {"fact_id": "f1", "subject": "the Council", "predicate": "decided",
        "object": "X exists", "natural_language": "The Council decided X exists."}
NESTED = {"gold_id": "g1", "natural_language": "The information must be disregarded.",
          "triples": [{"subject": "the information", "predicate": "must be",
                       "object": "disregarded"},
                      {"subject": "the goods", "predicate": "are not of",
                       "object": "marketable quality"}]}
BARE = {"fact_id": "f3", "natural_language": "Only a sentence."}


class TestFactText:
    def test_flat_fact_is_unchanged(self):
        # tau = 0.78 was swept on this exact string. It must not move.
        assert cc.fact_text(FLAT, "triple") == "the Council decided X exists"

    def test_nested_fact_uses_its_triples_not_its_sentence(self):
        out = cc.fact_text(NESTED, "triple")
        assert out == ("the information must be disregarded ; "
                       "the goods are not of marketable quality")
        assert NESTED["natural_language"] not in out

    def test_only_a_fact_with_no_triple_at_all_falls_back(self):
        assert cc.fact_text(BARE, "triple") == "Only a sentence."

    def test_natural_language_mode_still_prefers_the_sentence(self):
        assert cc.fact_text(NESTED, "natural_language") == NESTED["natural_language"]
        assert cc.fact_text(FLAT, "natural_language") == FLAT["natural_language"]

    def test_an_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            cc.fact_text(FLAT, "sentence")

    def test_shape_detection(self):
        assert cc.fact_shape(FLAT) == "flat"
        assert cc.fact_shape(NESTED) == "nested"
        assert cc.fact_shape(BARE) == "empty"

    def test_report_counts_what_actually_happened(self):
        rep = cc.fact_text_report([FLAT, NESTED, BARE], "triple")
        assert rep == {"mode": "triple", "n": 3, "flat": 1, "nested": 1,
                       "fell_back": 1, "mixed": True}

    def test_alignment_delegates_and_reports(self):
        prev = alignment.set_fact_text_mode("triple")
        try:
            assert alignment._fact_text(NESTED) == cc.fact_text(NESTED, "triple")
            assert alignment.fact_text_report([NESTED])["nested"] == 1
        finally:
            alignment.set_fact_text_mode(prev)


# ---------------------------------------------------------------------------
# One definition, not three
# ---------------------------------------------------------------------------

class TestSingleSourceOfTruth:
    def test_gold_compare_reexports_the_shared_normalisers(self):
        assert gold_compare.normalize_term is cc.normalize_term
        assert gold_compare.normalize_predicate is cc.normalize_predicate
        assert gold_compare._PRED_STOP_LEAD is cc._PRED_STOP_LEAD

    def test_layer1_uses_the_shared_scrubber(self):
        assert l1._NOISE_NUMBER_PATTERNS is cc._REFERENCE_PATTERNS
        assert l1._content_numbers("Decision 2005/136/EC") == set()

    def test_predicate_stop_list_is_the_full_set(self):
        # A shorter list was written into the core by mistake during the move.
        # Every one of these must still be stripped or RQ2 predicate similarity
        # moves for the whole corpus.
        for w in ("would", "could", "am", "does", "do", "did", "hereby",
                  "to", "the", "a", "an"):
            assert w in cc._PRED_STOP_LEAD

    def test_alignment_slot_gates_use_normalised_text(self):
        # The gates are gold_compare's numbers; they have to see gold_compare's
        # strings. Until 2026-08-27 alignment fed them raw slot text.
        assert cc.normalize_slot("predicate", "shall be applied") == "applied"
        assert cc.normalize_slot("subject", "The Council") == "council"


# ---------------------------------------------------------------------------
# BUG-2026-08-27-C, second half — the override may only speak where the
# embedding is blind. Equality alone is not identity.
# ---------------------------------------------------------------------------

class TestNumericOverrideDirection:
    @pytest.mark.parametrize("a, b", [
        ("ECU 20 per tonne", "ECU 20 per hectare"),   # changed unit
        ("2 June 2004", "2 July 2004"),               # changed month
        ("Article 3 of Annex I", "Article 3 of Annex II"),
    ])
    def test_equal_numbers_with_other_content_defer_to_the_embedding(self, a, b):
        # Forcing 1.0 here claims identity from digits alone and hides the very
        # difference the slot gate exists to catch.
        assert cc.numeric_override(a, b) is None

    def test_equal_numbers_with_a_bare_literal_still_assert(self):
        # A bare gold literal has no other content, so numeric equality is the
        # whole truth — this is what lets "25.0" match "25 metres".
        assert cc.numeric_override("25.0", "25 metres") == 1.0
        assert cc.numeric_override("2 000", "2 000 tonnes") == 1.0

    def test_different_numbers_always_veto(self):
        assert cc.numeric_override("25.0", "4349 metres") == 0.0
        assert cc.numeric_override("EUR 1 000", "EUR 1 500") == 0.0

    def test_bare_number_detection(self):
        assert cc.is_bare_number("25.0")      # normalises to "25 0"
        assert cc.is_bare_number("2 000")
        assert not cc.is_bare_number("25 metres")
        assert not cc.is_bare_number("the Council")
        assert not cc.is_bare_number("")


# ---------------------------------------------------------------------------
# BUG-2026-08-27-F — the missing side never learned its counterpart was taken
# ---------------------------------------------------------------------------

class TestFailReasonSymmetry:
    def test_a_taken_prediction_is_not_reported_as_a_threshold_failure(self):
        # Two near-identical gold triples, one prediction. The prediction is
        # consumed by whichever gold wins; the loser's every slot gate still
        # passes, so before the fix it was filed under `overall_below_threshold`
        # — a threshold complaint standing in for a duplicate-gold one, in the
        # counts a guideline revision is authored from.
        gold = [{"subject": "the Council", "predicate": "adopted",
                 "object": "the Decision"},
                {"subject": "the Council", "predicate": "adopted",
                 "object": "the Decision"}]
        pred = [{"subject": "the Council", "predicate": "adopted",
                 "object": "the Decision", "fact_id": "f1"}]
        r = gold_compare.compare_soft(gold, pred)
        reasons = [m["reason"] for m in r["missing"]]
        assert "overall_below_threshold" not in reasons

    def test_both_consumed_labels_are_reachable_and_distinct(self):
        import numpy as np
        sims = {sl: np.ones((1, 1), dtype=float)
                for sl in ("subject", "predicate", "object")}
        th = {"subject": 0.7, "predicate": 0.45, "object": 0.7, "overall": 0.62}
        assert gold_compare._fail_reason(sims, 0, 0, th, consumed=False) == \
            "overall_below_threshold"
        assert gold_compare._fail_reason(sims, 0, 0, th, consumed=True) == \
            "gold_already_matched"
        assert gold_compare._fail_reason(
            sims, 0, 0, th, consumed=True,
            consumed_label="prediction_already_matched") == "prediction_already_matched"
