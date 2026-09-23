"""Guards for the external-adjudication round trip.

An outside model adjudicating the gold removes the circularity in the current
artefact (same model pre-annotated and adjudicated it). Two things must hold or
the exercise is worse than not doing it:

1. **The second pass must be blind.** A packet carrying the first adjudicator's
   reasoning produces a review of that reasoning, not an independent decision,
   and the agreement number computed from it is meaningless.
2. **Nothing unvalidated reaches the gold.** A silently dropped fact shrinks the
   gold; a non-verbatim quote breaks the invariant `gold_store` enforces.

And it stays *machine* provenance — an external model is a second machine
opinion, not a human sign-off.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


packet = _load("make_adjudication_packet")
imp = _load("import_external_adjudication")


FACT = {
    "gold_id": "train-000000::preamble.recitals[0]::f01",
    "source_quote": "the Council decided",
    "natural_language": "The Council decided that a deficit existed.",
    "triples": [{"subject": "the Council", "predicate": "decided",
                 "object": "that a deficit existed"}],
    "reviewer_note": "D-15: coordination left unexpanded by the pre-annotation.",
}
FLAGS = {FACT["gold_id"]: {"checks": [
    {"severity": "warn", "code": "quote_misses_object", "detail": "mechanical check"},
    {"severity": "info", "code": "carries_annotator_note", "detail": "D-15: ..."},
]}}


# --- 1. blindness -----------------------------------------------------------

def test_blind_packet_hides_the_first_adjudicators_reasoning():
    text = "\n".join(packet._fact_block(FACT, FLAGS, blind=True))
    assert "D-15" not in text
    assert "note  :" not in text
    assert "carries_annotator_note" not in text


def test_blind_packet_keeps_the_mechanical_flags():
    """Those are properties of the text, not anyone's judgement — and they are
    what makes the queue triageable."""
    text = "\n".join(packet._fact_block(FACT, FLAGS, blind=True))
    assert "quote_misses_object" in text
    assert "the Council decided" in text          # the fact itself is still there


def test_non_blind_packet_still_shows_the_note():
    text = "\n".join(packet._fact_block(FACT, FLAGS, blind=False))
    assert "D-15" in text


# --- 2. import validation ---------------------------------------------------

def _expect(seg_id="s1", text="the Council decided that a deficit existed"):
    return {seg_id: {"section_path": "p", "char_start": 0, "char_end": len(text)}}


def _patch_gold(monkeypatch, facts, seg_text):
    monkeypatch.setattr(imp, "_gold_index",
                        lambda doc_id: ({f["gold_id"]: f for f in facts},
                                        {f["gold_id"]: "s1" for f in facts}))
    monkeypatch.setattr(imp, "_seg_text", lambda d, s, e: seg_text)


def test_missing_decision_is_an_error(monkeypatch):
    """A silently dropped fact would shrink the gold."""
    _patch_gold(monkeypatch, [FACT], "the Council decided that a deficit existed")
    _, _, errors = imp.validate_doc("train-000000", {"decisions": {}}, _expect())
    assert any("no decision returned" in e for e in errors)


def test_unknown_gold_id_is_an_error(monkeypatch):
    _patch_gold(monkeypatch, [FACT], "the Council decided that a deficit existed")
    payload = {"decisions": {FACT["gold_id"]: {"status": "accepted"},
                             "train-000000::made::up::f99": {"status": "accepted"}}}
    _, _, errors = imp.validate_doc("train-000000", payload, _expect())
    assert any("unknown gold_id" in e for e in errors)


def test_non_verbatim_edited_quote_is_rejected(monkeypatch):
    _patch_gold(monkeypatch, [FACT], "the Council decided that a deficit existed")
    payload = {"decisions": {FACT["gold_id"]: {
        "status": "edited", "note": "x",
        "edits": {"source_quote": "the Council hereby resolved"}}}}
    clean, _, errors = imp.validate_doc("train-000000", payload, _expect())
    assert FACT["gold_id"] not in clean
    assert any("verbatim" in e for e in errors)


def test_verbatim_edited_quote_is_accepted(monkeypatch):
    _patch_gold(monkeypatch, [FACT], "the Council decided that a deficit existed")
    payload = {"decisions": {FACT["gold_id"]: {
        "status": "edited", "note": "narrowed",
        "edits": {"source_quote": "a deficit existed"}}}}
    clean, _, errors = imp.validate_doc("train-000000", payload, _expect())
    assert clean[FACT["gold_id"]]["edits"]["source_quote"] == "a deficit existed"
    assert not errors


def test_edits_on_a_non_edited_status_is_rejected(monkeypatch):
    _patch_gold(monkeypatch, [FACT], "the Council decided that a deficit existed")
    payload = {"decisions": {FACT["gold_id"]: {
        "status": "accepted", "edits": {"object": "something else"}}}}
    clean, _, errors = imp.validate_doc("train-000000", payload, _expect())
    assert not clean and any("edits present" in e for e in errors)


def test_unknown_slot_is_rejected(monkeypatch):
    _patch_gold(monkeypatch, [FACT], "the Council decided that a deficit existed")
    payload = {"decisions": {FACT["gold_id"]: {
        "status": "edited", "edits": {"confidence": "high"}}}}
    clean, _, errors = imp.validate_doc("train-000000", payload, _expect())
    assert not clean and any("not editable" in e for e in errors)


def test_bad_status_is_rejected(monkeypatch):
    _patch_gold(monkeypatch, [FACT], "the Council decided that a deficit existed")
    payload = {"decisions": {FACT["gold_id"]: {"status": "probably fine"}}}
    clean, _, errors = imp.validate_doc("train-000000", payload, _expect())
    assert not clean and any("bad status" in e for e in errors)


# --- 3. tolerant parsing of a chat reply ------------------------------------

def test_parses_a_fenced_block_wrapped_in_prose():
    raw = ('Sure! Here is the adjudication.\n\n```json\n'
           '{"doc_id": "d", "decisions": {"a": {"status": "accepted"}}}\n```\n'
           'Ready for the next one.')
    assert imp._parse(raw)["decisions"]["a"]["status"] == "accepted"


def test_parses_bare_json():
    assert imp._parse('{"decisions": {"a": {"status": "rejected"}}}')["decisions"]


def test_reply_without_decisions_raises():
    with pytest.raises(ValueError):
        imp._parse("I could not complete this task.")
