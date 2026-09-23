"""Tests for src.conflict_layer2 tolerant parsing + arbitration fallback.

These are hermetic: no Ollama needed. They pin the behaviour that keeps the
local-model Layer-2 path usable when the cloud arbitrator is unavailable —
tolerant JSON extraction and honest 'escalate' on failure.
"""

from __future__ import annotations

from src.alignment import AlignedPair
from src.conflict_layer2 import _parse_response, _extract_json_obj, arbitrate


def _pair(cosine=0.9, label="escalate"):
    fa = {"fact_id": "a", "subject": "X", "predicate": "is", "object": "Y",
          "natural_language": "X is Y.", "source_locator": {"quote": "X is Y"}}
    fb = {"fact_id": "b", "subject": "X", "predicate": "is", "object": "Z",
          "natural_language": "X is Z.", "source_locator": {"quote": "X is Z"}}
    p = AlignedPair("A", "B", fa, fb, cosine, "matched")
    p.conflict_label = label
    return p


# ---- _parse_response -------------------------------------------------------

def test_strict_json():
    assert _parse_response('{"label": "CONTRADICTION", "reason": "diff"}') == ("CONTRADICTION", "diff")

def test_fenced_json():
    raw = '```json\n{"label": "GRANULARITY", "reason": "split"}\n```'
    assert _parse_response(raw) == ("GRANULARITY", "split")

def test_prose_prefixed_json():
    raw = 'Sure, here is my classification:\n{"label":"REDUNDANCY","reason":"paraphrase"}'
    assert _parse_response(raw) == ("REDUNDANCY", "paraphrase")

def test_bare_label_token():
    lbl, reason = _parse_response("After analysis, this is a CONTRADICTION.")
    assert lbl == "CONTRADICTION"

def test_no_conflict_space_normalised():
    assert _parse_response('{"label": "NO CONFLICT", "reason": ""}')[0] == "NO_CONFLICT"

def test_ambiguous_multiple_labels_fails():
    lbl, diag = _parse_response("It is not a CONTRADICTION, more like REDUNDANCY or GRANULARITY.")
    assert lbl is None and diag.startswith("ambiguous_labels")

def test_empty_fails():
    assert _parse_response("")[0] is None
    assert _parse_response("   ")[0] is None

def test_out_of_vocab_label_fails():
    lbl, _ = _parse_response('{"label": "MAYBE", "reason": "x"}')
    assert lbl is None

def test_extract_json_first_balanced_span():
    obj = _extract_json_obj('noise {"label": "REDUNDANCY"} trailing {"x": 1}')
    assert obj == {"label": "REDUNDANCY"}


# ---- arbitrate() fallback behaviour ---------------------------------------

def test_arbitrate_sets_label_from_fenced_response():
    p = _pair()
    def stub(prompt, **kw):
        return '```json\n{"label":"CONTRADICTION","reason":"15 vs 20"}\n```'
    arbitrate(p, chat_fn=stub)
    assert p.conflict_label == "contradiction"
    assert "15 vs 20" in (p.layer2_reason or "")

def test_arbitrate_call_error_stays_escalate():
    p = _pair()
    def boom(prompt, **kw):
        raise ConnectionError("cloud down")
    arbitrate(p, chat_fn=boom)
    assert p.conflict_label == "escalate"          # honest: unresolved, visible
    assert "call_error" in (p.layer2_reason or "")

def test_arbitrate_empty_both_attempts_stays_escalate():
    p = _pair()
    def empty(prompt, **kw):
        return ""
    arbitrate(p, chat_fn=empty)
    assert p.conflict_label == "escalate"
    assert "empty_response" in (p.layer2_reason or "")

def test_arbitrate_skips_layer1_labelled_pair():
    p = _pair(label="redundancy")
    def stub(prompt, **kw):
        return '{"label":"CONTRADICTION"}'
    arbitrate(p, chat_fn=stub)
    assert p.conflict_label == "redundancy"        # Layer-1 decision preserved
