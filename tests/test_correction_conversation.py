"""Detector defects are corrected IN CONVERSATION, capped at 3 turns.

What changed on 2026-09-07, and why each property is pinned here:

1. **One conversation per section.** The prompt is sent once and each
   complaint is appended to the same `/api/chat` exchange. The old path
   re-sent the whole section prompt with the error glued on, so the model
   never saw the rejected JSON as its own turn — and the section text was
   paid for on every attempt. The conversation is rebuilt per section, so
   nothing accumulates across sections.
2. **Every defect is fed back.** An unanchorable `source_quote` and a triple
   the schema rejects used to be dropped on a `logger.warning`, silently, with
   no retry — the section simply under-reported.
3. **A cut-off answer is not called malformed.** `done_reason == "length"`
   means the text was truncated; saying "your JSON is invalid" sends the model
   after a syntax error it did not make.
4. **A retry never costs already-good facts.** If a later turn returns fewer
   facts, the earlier result is kept.
"""

from __future__ import annotations

import json

import pytest

import src.extractor as EX
from src.extractor import Section
from src.providers import resolve
from src.schema import ParsedDocument

SECTION_TEXT = "The Commission shall adopt implementing acts before 1 July 2001."


@pytest.fixture()
def doc() -> ParsedDocument:
    return ParsedDocument(
        celex="test-0001", title="Test Regulation",
        citations=[], recitals=[], articles=[], annexes=[],
        concluding_formulas="", preamble_text="", enacting_text=SECTION_TEXT,
        linearized_table_rows=[],
    )


@pytest.fixture()
def section() -> Section:
    return Section(section_path="enacting.article_1", text=SECTION_TEXT,
                   container_text=SECTION_TEXT, base_offset=0)


def _answer(quote=SECTION_TEXT, subject="the Commission", n=1):
    return json.dumps({"facts": [
        {"natural_language": "The Commission shall adopt implementing acts.",
         "source_quote": quote,
         "triples": [{"subject": subject, "predicate": "shall adopt",
                      "object": "implementing acts"}]}
    ] * n})


class _Scripted:
    """Replays a list of (raw, done_reason) answers, recording each request."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls: list[list[dict]] = []

    def __call__(self, resolved, messages, *, headers=None):
        # Copy: the caller keeps mutating the same list.
        self.calls.append([dict(m) for m in messages])
        return self.answers.pop(0)


def _run(monkeypatch, answers, doc, section, retries=2):
    fake = _Scripted(answers)
    monkeypatch.setattr(EX, "_converse_json", fake)
    facts = EX._extract_one_section(
        resolve("stub-model", default_base_url="http://x"), doc, section,
        "<<SECTION_TEXT>>", guideline_version="v1", retries=retries)
    return facts, fake


def test_prompt_is_sent_once_and_the_error_continues_the_conversation(
        monkeypatch, doc, section):
    facts, fake = _run(monkeypatch, [
        (json.dumps({"facts": [{"natural_language": "n", "source_quote": SECTION_TEXT,
                                "triples": []}]}), "stop"),
        (_answer(), "stop"),
    ], doc, section)

    assert len(facts) == 1
    assert len(fake.calls) == 2
    # Turn 1: just the prompt.
    assert [m["role"] for m in fake.calls[0]] == ["user"]
    # Turn 2: same prompt object, plus the model's answer and the complaint —
    # NOT the prompt re-rendered with the error appended.
    assert [m["role"] for m in fake.calls[1]] == ["user", "assistant", "user"]
    assert fake.calls[1][0]["content"] == fake.calls[0][0]["content"]
    assert "triples" in fake.calls[1][2]["content"]
    # The section text appears once, in the first turn only.
    assert SECTION_TEXT not in fake.calls[1][2]["content"]


def test_conversation_is_capped_at_retries_plus_one_turns(monkeypatch, doc, section):
    bad = (json.dumps({"facts": [{"natural_language": "n",
                                  "source_quote": SECTION_TEXT,
                                  "triples": []}]}), "stop")
    with pytest.raises(EX.SectionUnparsed):
        _run(monkeypatch, [bad, bad, bad], doc, section, retries=2)


def test_unanchorable_quote_is_fed_back_not_dropped(monkeypatch, doc, section):
    """Property 2. The first answer quotes text that is not in the section."""
    facts, fake = _run(monkeypatch, [
        (_answer(quote="The Council shall do something else entirely."), "stop"),
        (_answer(), "stop"),
    ], doc, section)

    assert len(facts) == 1                       # recovered on the second turn
    assert len(fake.calls) == 2
    complaint = fake.calls[1][2]["content"]
    assert "source_quote" in complaint
    assert "CHARACTER-FOR-CHARACTER" in complaint


def test_schema_rejection_is_fed_back_with_a_short_reason(monkeypatch, doc, section):
    """Property 2, other half: a triple pydantic refuses."""
    facts, fake = _run(monkeypatch, [
        (json.dumps({"facts": [
            {"natural_language": "n", "source_quote": SECTION_TEXT,
             "triples": [{"subject": "s", "predicate": {"bad": 1}, "object": "o"}]}
        ]}), "stop"),
        (_answer(), "stop"),
    ], doc, section)
    assert len(facts) == 1
    # The model must be told which field and why, and briefly: the point of the
    # bound is that pydantic's multi-line report and its docs URL never reach
    # the conversation. (The `object` field is the one exception — it carries a
    # deliberate worked example; see the empty-object test below.)
    complaint = fake.calls[1][2]["content"]
    assert "predicate" in complaint
    assert len(complaint) < 600
    assert "https://errors.pydantic.dev" not in complaint


def test_truncated_answer_is_reported_as_cut_off_not_malformed(
        monkeypatch, doc, section):
    """Property 3."""
    facts, fake = _run(monkeypatch, [
        ('{"facts": [{"natural_language": "The Commissio', "length"),
        (_answer(), "stop"),
    ], doc, section)

    assert len(facts) == 1
    complaint = fake.calls[1][2]["content"]
    assert "cut off" in complaint
    assert "FEWER" in complaint
    assert "not valid JSON" not in complaint


def test_a_later_turn_never_costs_already_good_facts(monkeypatch, doc, section):
    """Property 4: turn 1 yields 2 facts + 1 defect, turn 2 yields 1 fact."""
    two_good_one_bad = json.dumps({"facts": [
        {"natural_language": "a", "source_quote": SECTION_TEXT,
         "triples": [{"subject": "the Commission", "predicate": "shall adopt",
                      "object": "implementing acts"},
                     {"subject": "the Commission", "predicate": "shall act",
                      "object": "before 1 July 2001"}]},
        {"natural_language": "b", "source_quote": "not in the section at all",
         "triples": [{"subject": "x", "predicate": "y", "object": "z"}]},
    ]})
    facts, fake = _run(monkeypatch, [
        (two_good_one_bad, "stop"),
        (_answer(), "stop"),          # model over-corrects down to one fact
    ], doc, section)

    assert len(facts) == 2, "the richer earlier turn must be kept"


def test_mechanism_defects_never_cost_a_turn(monkeypatch, doc, section):
    """The exclusion hook: a defect the best possible annotator would also hit
    is logged, not retried. The pattern list ships empty on purpose — this
    pins the wiring so adding a pattern is a one-line change."""
    monkeypatch.setattr(EX, "MECHANISM_DEFECT_PATTERNS", ("could not be found",))
    facts, fake = _run(monkeypatch, [
        (_answer(quote="The Council shall do something else entirely."), "stop"),
        (_answer(), "stop"),
    ], doc, section)

    assert len(fake.calls) == 1, "a mechanism defect must not trigger a retry"
    assert facts == []


def test_a_truncated_answer_is_echoed_elided_not_in_full(monkeypatch, doc, section):
    """A cut-off answer has no defect inside it to look at, and replaying all of
    it crowds the context with the pattern that overran: on `train-000124
    preamble.recitals[0]` the second turn arrived with prompt_eval 14455 rather
    than 6205 and truncated again in the same place."""
    huge = '{"facts": [' + '{"natural_language": "x",' * 900
    facts, fake = _run(monkeypatch, [
        (huge, "length"),
        (_answer(), "stop"),
    ], doc, section)

    echoed = fake.calls[1][1]["content"]
    assert len(echoed) < len(huge) / 4
    assert "elided" in echoed
    assert echoed.startswith(huge[:200])          # it still sees what it began


def test_a_normally_rejected_answer_is_echoed_in_full(monkeypatch, doc, section):
    """The other half: to repair malformed JSON the model must see all of it."""
    bad = json.dumps({"facts": [{"natural_language": "n",
                                 "source_quote": SECTION_TEXT, "triples": []}]})
    facts, fake = _run(monkeypatch, [(bad, "stop"), (_answer(), "stop")],
                       doc, section)
    assert fake.calls[1][1]["content"] == bad


def test_one_bad_group_costs_that_group_not_the_answer(monkeypatch, doc, section):
    """BUG-2026-09-07-D: `train-000124 preamble.recitals[0]` returned 65 groups
    and the 65th lacked an `object`; the strict parser discarded all 65."""
    good = {"natural_language": "n", "source_quote": SECTION_TEXT,
            "triples": [{"subject": "the Commission", "predicate": "shall adopt",
                         "object": "implementing acts"}]}
    bad = {"natural_language": "n", "source_quote": SECTION_TEXT,
           "triples": [{"subject": "the Commission", "predicate": "shall adopt"}]}
    facts, fake = _run(monkeypatch, [
        (json.dumps({"facts": [good, good, bad]}), "stop"),
        (json.dumps({"facts": [good, good, good]}), "stop"),
    ], doc, section)

    # Turn 1 banks the two good groups and complains only about the third.
    complaint = fake.calls[1][2]["content"]
    assert "facts[2]" in complaint and "object" in complaint
    assert len(facts) == 3, "turn 2 fixed it, so all three groups stand"


def test_an_answer_with_no_usable_group_is_still_an_error(monkeypatch, doc, section):
    """Tolerance must not turn a wholly bad answer into 'the model found
    nothing' — that is BUG-2026-09-07-B's silent under-count."""
    bad = {"natural_language": "n", "source_quote": SECTION_TEXT, "triples": []}
    with pytest.raises(EX.SectionUnparsed):
        _run(monkeypatch, [(json.dumps({"facts": [bad]}), "stop")] * 3,
             doc, section)


def test_a_very_large_answer_is_echoed_elided_even_when_complete(
        monkeypatch, doc, section):
    """The 57 871-character case: echoed whole, it took turn 2 to prompt_eval
    20 909 and the model rewrote all 65 groups instead of fixing one."""
    good = {"natural_language": "n", "source_quote": SECTION_TEXT,
            "triples": [{"subject": "the Commission", "predicate": "shall adopt",
                         "object": "implementing acts"}]}
    bad = {"natural_language": "n", "source_quote": SECTION_TEXT,
           "triples": [{"subject": "s", "predicate": "p"}]}
    huge = json.dumps({"facts": [good] * 60 + [bad]})
    assert len(huge) > EX.MAX_ECHO_CHARS

    facts, fake = _run(monkeypatch, [(huge, "stop"), (_answer(), "stop")],
                       doc, section)
    echoed = fake.calls[1][1]["content"]
    assert "elided" in echoed and len(echoed) < EX.MAX_ECHO_CHARS
    assert len(facts) == 60, "the 60 good groups are kept over the shorter retry"


def test_an_empty_object_is_told_the_convention_not_just_the_schema(
        monkeypatch, doc, section):
    """9 of 11 defects on 2026-09-07 were an empty `object`, and the model
    fixed none of them in three turns: "must be a non-empty string" states the
    schema and teaches nothing about a passive that has no object ("The
    security ... shall be released"). The convention comes from the gold,
    where 0 of 556 triples have an empty object."""
    facts, fake = _run(monkeypatch, [
        (json.dumps({"facts": [{"natural_language": "n",
                                "source_quote": SECTION_TEXT,
                                "triples": [{"subject": "the security",
                                             "predicate": "shall be released",
                                             "object": ""}]}]}), "stop"),
        (_answer(), "stop"),
    ], doc, section)

    complaint = fake.calls[1][2]["content"]
    assert "shall enter into force on" in complaint      # the worked example
    assert "Do not invent an object" in complaint
    assert len(facts) == 1


def test_a_raw_control_character_in_a_string_does_not_kill_the_answer(
        monkeypatch, doc, section):
    """A literal control character inside a quoted passage cost a section on
    2026-09-09: `json.loads` is strict by default, so one stray byte threw
    away annotations that were otherwise fine (BUG-2026-09-09-B). The two
    modes differ ONLY on this case, so nothing that already parsed can move.
    """
    payload = json.dumps({"facts": [{"natural_language": "n",
                                     "source_quote": SECTION_TEXT,
                                     "triples": [{"subject": "a",
                                                  "predicate": "b",
                                                  "object": "c"}]}]})
    # Put a bare control character inside a string value, the way a model does
    # when it copies a line break into a quote instead of escaping it.
    broken = payload.replace('"natural_language": "n"',
                             '"natural_language": "n\x0btail"')
    facts, _ = _run(monkeypatch, [(broken, "stop")], doc, section)

    assert len(facts) == 1, "the answer survives the stray control character"
    assert facts[0].natural_language.startswith("n")


def test_a_genuinely_malformed_answer_is_still_rejected(
        monkeypatch, doc, section):
    """The lenient parse must not become a blanket 'accept anything': a
    truncated or otherwise broken document still has to fail, or the retry
    ladder stops seeing the errors it exists to correct."""
    facts, fake = _run(monkeypatch, [
        ('{"facts": [{"natural_language": "n", "source_q', "stop"),
        (_answer(), "stop"),
    ], doc, section)

    complaint = fake.calls[1][-1]["content"]
    assert "not valid JSON" in complaint, "the break was fed back, not accepted"
    assert len(facts) == 1, "and the corrected turn is what gets kept"


def test_a_failed_correction_turn_does_not_discard_facts_that_already_parsed(
        monkeypatch, doc, section):
    """The rule "a retry never costs already-good facts" used to hold only
    when the later turn returned FEWER facts. If the later turn failed to
    PARSE, the loop fell through the `except ExtractionError` branch's
    `continue` straight to SectionUnparsed, which never consulted
    `best_effort` — and `train-000124 preamble.recitals[0]` lost 71
    already-parsed facts because turn 3 came back as malformed JSON
    (BUG-2026-09-09-C).
    """
    good = json.loads(_answer())
    good["facts"].append({"natural_language": "n2",
                          "source_quote": SECTION_TEXT,
                          "triples": [{"subject": "the Commission",
                                       "predicate": "shall adopt",
                                       "object": ""}]})
    facts, _ = _run(monkeypatch, [
        (json.dumps(good), "stop"),      # parses; one empty-object defect
        ('{"facts": [{"natural', "stop"),  # correction turn 2: broken JSON
        ('not json at all', "stop"),       # correction turn 3: broken too
    ], doc, section)

    assert len(facts) == 1, "the fact that already parsed survives"
    assert facts[0].predicate == "shall adopt"
    assert facts[0].object == "implementing acts"


def test_a_section_that_never_parsed_at_all_is_still_unparsed(
        monkeypatch, doc, section):
    """The fallback must not turn every failure into a silent success: with
    nothing ever parsed there is no best_effort, and the section must still be
    recorded as lost rather than as an empty-but-clean result."""
    with pytest.raises(EX.SectionUnparsed):
        _run(monkeypatch, [
            ('{"facts": [{"natural', "stop"),
            ('still not json', "stop"),
            ('nor this', "stop"),
        ], doc, section)
