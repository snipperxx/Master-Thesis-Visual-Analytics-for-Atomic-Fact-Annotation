"""Layer-2 on the GROUP layer, and the request shape that keeps it non-empty.

Hermetic: no Ollama. Two bugs are pinned here, both found 2026-08-09:

1. **Nothing arbitrated groups.** `run_phase2` built the groups, labelled them
   by shape / Layer-1, then arbitrated the *pair* list — so a one-to-one group on
   which no Layer-1 rule fired kept `escalate` permanently (39 of 132 groups on
   train-000004). See `doc/bugfix_log.md`.
2. **"Empty" Layer-2 bodies.** The answer was arriving in the response's
   `thinking` channel while `response` was ""; the dispatcher read only
   `response`. Guarded by the `_chat_once` tests at the bottom.
"""

from __future__ import annotations

import json

import pytest

import src.extractor as EX
from src.alignment import AlignedPair, FactGroup, build_groups, group_to_dict
from src.conflict_layer1 import Layer1Config, classify_groups
from src import conflict_layer2 as L2


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def _fact(fid, subject="the levy", predicate="is set at", obj="ECU 10", quote=None):
    return {"fact_id": fid, "subject": subject, "predicate": predicate, "object": obj,
            "natural_language": f"{subject} {predicate} {obj}",
            "source_locator": {"quote": quote or f"{subject} {predicate} {obj}."}}


def _group(shape, *, label="escalate", n_partners=1):
    partners = [_fact(f"b{i}", obj=f"ECU {i}") for i in range(n_partners)]
    g = FactGroup(annotator_a="A", annotator_b="B", anchor_side="a",
                  anchor=_fact("a0"), partners=partners, shape=shape,
                  cosines=[0.8] * n_partners)
    g.conflict_label = label
    return g


def _chat(label="CONTRADICTION", reason="different amounts"):
    """A stub `chat_fn` that records every prompt it is asked."""
    seen = []

    def fn(prompt, **_kw):
        seen.append(prompt)
        return json.dumps({"label": label, "reason": reason})

    fn.seen = seen
    return fn


# --------------------------------------------------------------------------
# The bug: escalated groups never reached the arbitrator
# --------------------------------------------------------------------------

def test_escalated_one_to_one_group_gets_a_layer2_label():
    g = _group("one_to_one", label="escalate")
    chat = _chat("CONTRADICTION")
    counts = L2.arbitrate_groups([g], chat_fn=chat)

    assert g.conflict_label == "contradiction"
    assert g.label_source == "layer2"
    assert counts["_layer2_calls"] == 1
    assert counts["contradiction"] == 1


def test_group_layer_no_longer_leaves_escalate_behind():
    """Regression guard for the reported bug: after Layer-2, no group whose shape
    is arbitrable may still carry 'escalate'."""
    groups = [_group("one_to_one", label="escalate") for _ in range(5)]
    L2.arbitrate_groups(groups, chat_fn=_chat("REDUNDANCY"))
    assert not [g for g in groups if g.conflict_label == "escalate"]


def test_layer1_labelled_group_is_not_re_arbitrated():
    g = _group("one_to_one", label="redundancy")
    counts = L2.arbitrate_groups([g], chat_fn=_chat())
    assert counts["_layer2_calls"] == 0
    assert g.conflict_label == "redundancy"


def test_orphan_groups_are_never_billed_to_the_arbitrator():
    """An orphan has no partner to compare against, and the orphan rate is
    coverage rather than a conflict type (method.tex)."""
    g = _group("orphan", label="unlabeled", n_partners=0)
    counts = L2.arbitrate_groups([g], chat_fn=_chat(),
                                 shapes=("one_to_one", "one_to_many", "orphan"))
    assert counts["_layer2_calls"] == 0
    assert g.conflict_label == "unlabeled"


# --------------------------------------------------------------------------
# One-to-many: shape rule by default, model check only on request
# --------------------------------------------------------------------------

def test_one_to_many_is_not_arbitrated_by_default():
    """method.tex labels a one-to-many group `granularity` from its shape. The
    default must not spend a call, and must not change the label."""
    g = _group("one_to_many", label="granularity", n_partners=3)
    g.label_source = "shape"
    counts = L2.arbitrate_groups([g], chat_fn=_chat("NO_CONFLICT"))
    assert counts["_layer2_calls"] == 0
    assert g.conflict_label == "granularity" and g.label_source == "shape"


def test_one_to_many_check_is_opt_in_and_keeps_the_shape_verdict():
    g = _group("one_to_many", label="granularity", n_partners=3)
    g.label_source = "shape"
    counts = L2.arbitrate_groups([g], chat_fn=_chat("NO_CONFLICT"),
                                 shapes=("one_to_one", "one_to_many"))
    assert counts["_layer2_calls"] == 1
    assert g.conflict_label == "no_conflict"
    assert g.prior_label == "granularity"      # shape verdict recoverable
    assert g.label_source == "layer2"


def test_many_side_reaches_the_model_as_a_numbered_list():
    """The arbitrator cannot see a split unless the many side is enumerated."""
    g = _group("one_to_many", label="granularity", n_partners=3)
    chat = _chat()
    L2.arbitrate_groups([g], chat_fn=chat, shapes=("one_to_one", "one_to_many"))
    prompt = chat.seen[0]
    assert "(1) " in prompt and "(2) " in prompt and "(3) " in prompt


def test_partners_sharing_one_natural_language_are_split_out_by_slot():
    """The 4B annotators write the same `natural_language` for facts whose slots
    differ (nemotron: 10 of 25 facts on train-000004). NL-only rendering showed
    the model N identical sentences, which it labelled redundancy — a real split
    lost. The slots must appear when the NL does not distinguish the members."""
    shared = "Refunds must account for the relevant factors."
    partners = [
        {"fact_id": "b1", "subject": "Refunds", "predicate": "must account for",
         "object": "market conditions", "natural_language": shared},
        {"fact_id": "b2", "subject": "Refunds", "predicate": "must account for",
         "object": "price and cost factors", "natural_language": shared},
    ]
    text = L2._side_text(partners)
    assert "market conditions" in text and "price and cost factors" in text
    assert text.count(shared) == 2          # NL kept, slots appended
    # Distinct NL needs no slot annotation.
    distinct = [dict(partners[0], natural_language="A one."),
                dict(partners[1], natural_language="A two.")]
    assert "[" not in L2._side_text(distinct)


def test_schema_fanout_is_flagged_not_relabelled():
    """A fan-out entirely inside ONE annotation group is the reification schema
    (one annotation, N triples, one shared natural_language), not one annotator
    splitting what another merged. It must be *countable* — and the label must be
    left exactly as the shape rule set it, because the linking criterion is an
    open method question (CLAUDE.md blocker 1)."""
    g = _group("one_to_many", label="granularity", n_partners=3)
    for p in g.partners:
        p["group_id"] = "same-annotation"
    d = group_to_dict(g)
    assert d["partners_one_annotation"] is True
    assert d["partner_annotation_groups"] == ["same-annotation"]
    assert d["conflict_label"] == "granularity"          # untouched

    g.partners[0]["group_id"] = "other-annotation"
    assert group_to_dict(g)["partners_one_annotation"] is False


def test_missing_group_ids_are_not_read_as_one_annotation():
    """Older fact files have no group_id; an absent id must not masquerade as
    'all partners share one annotation'."""
    g = _group("one_to_many", label="granularity", n_partners=3)
    d = group_to_dict(g)
    assert d["partner_annotation_groups"] == []
    assert d["partners_one_annotation"] is False


def test_one_to_one_group_is_never_schema_fanout():
    g = _group("one_to_one", label="escalate", n_partners=1)
    g.partners[0]["group_id"] = "a"
    assert group_to_dict(g)["partners_one_annotation"] is False


def test_prior_label_is_serialised():
    g = _group("one_to_many", label="granularity", n_partners=2)
    L2.arbitrate_groups([g], chat_fn=_chat("REDUNDANCY"),
                        shapes=("one_to_one", "one_to_many"))
    d = group_to_dict(g)
    assert d["prior_label"] == "granularity" and d["conflict_label"] == "redundancy"
    json.dumps(d)  # must stay JSON-serialisable


# --------------------------------------------------------------------------
# Failure is visible, never guessed
# --------------------------------------------------------------------------

def test_unparseable_response_keeps_the_prior_label_and_records_why():
    g = _group("one_to_one", label="escalate")
    g.reason = "no rule fired"
    L2.arbitrate_groups([g], chat_fn=lambda prompt, **kw: "")
    assert g.conflict_label == "escalate"          # not guessed
    assert "layer2:" in g.reason and "empty_response" in g.reason


def test_call_error_does_not_crash_the_batch():
    def boom(prompt, **_kw):
        raise RuntimeError("connection refused")

    groups = [_group("one_to_one"), _group("one_to_one")]
    counts = L2.arbitrate_groups(groups, chat_fn=boom)
    assert counts["escalate"] == 2
    assert all("call_error" in g.reason for g in groups)


# --------------------------------------------------------------------------
# The shared cache: a 1:1 group and its pair are one question
# --------------------------------------------------------------------------

def test_group_and_its_pair_share_one_call():
    fa, fb = _fact("a0"), _fact("b0", obj="ECU 99")
    pair = AlignedPair("A", "B", fa, fb, 0.8, "matched")
    pair.conflict_label = "escalate"
    groups = build_groups([pair])
    classify_groups(groups, Layer1Config())

    chat = _chat("CONTRADICTION")
    cache: dict = {}
    L2.arbitrate_groups(groups, chat_fn=chat, cache=cache)
    L2.arbitrate_all([pair], chat_fn=chat, cache=cache)

    # 2 groups (one anchored on each side) + 1 pair = 3 arbitrations, but they
    # are all the same comparison, so the model is asked exactly once.
    assert len(chat.seen) == 1
    assert len(cache) == 1
    assert pair.conflict_label == "contradiction"
    assert all(g.conflict_label == "contradiction" for g in groups)


def test_cache_key_is_side_order_independent():
    a, b = [_fact("a0")], [_fact("b0")]
    assert L2._cache_key(a, b) == L2._cache_key(b, a)


def test_different_comparisons_are_not_confused():
    a, b, c = [_fact("a0")], [_fact("b0")], [_fact("c0")]
    assert L2._cache_key(a, b) != L2._cache_key(a, c)


# --------------------------------------------------------------------------
# Request shape — the "empty Layer-2 body" root cause
# --------------------------------------------------------------------------

class _FakeResp:
    """A streamed Ollama response: NDJSON chunks, `done` on the last one.

    `_chat_once` streams since 2026-09-07 — the DBvis gateway kills any socket
    that has sent no bytes for ~50s (BUG-2026-08-31-D) — so the double must
    speak NDJSON, not a single JSON body.
    """

    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        pass

    def close(self):
        pass

    def iter_lines(self, decode_unicode=False):
        # Split the answer over two chunks so concatenation is exercised.
        resp = self._payload.get("response", "")
        think = self._payload.get("thinking", "")
        half = len(resp) // 2
        yield json.dumps({"response": resp[:half], "thinking": think})
        yield json.dumps({"response": resp[half:], "thinking": "",
                          "done": True, "done_reason": "stop"})


def _capture(monkeypatch, payload):
    """Point `_chat_once` at a fake Ollama; return the captured request body."""
    seen = {}

    def fake_post(url, json=None, timeout=None, headers=None, stream=None):
        seen["url"] = url
        seen["body"] = json
        seen["stream_kwarg"] = stream
        return _FakeResp(payload)

    monkeypatch.setattr(EX.requests, "post", fake_post)
    return seen


def test_answer_in_the_thinking_channel_is_recovered(monkeypatch):
    """qwen3.5 / nemotron route the answer to `thinking` unless think=False is
    sent. Reading only `response` is what produced "25/25 empty" bodies."""
    _capture(monkeypatch, {"response": "",
                           "thinking": '{"label": "GRANULARITY", "reason": "split"}'})
    raw = L2._chat_once("p", model_name="qwen3.5:4b", base_url="http://x")
    assert L2._parse_response(raw) == ("GRANULARITY", "split")


def test_response_channel_wins_when_both_are_present(monkeypatch):
    _capture(monkeypatch, {"response": '{"label": "REDUNDANCY", "reason": "r"}',
                           "thinking": "some reasoning"})
    assert L2._parse_response(L2._chat_once("p", model_name="m", base_url="u"))[0] == "REDUNDANCY"


def test_think_is_sent_and_is_never_true(monkeypatch):
    """Omitting the field is NOT the same as False (the model then reasons and
    blanks `response`), and True makes gemma3 return 400 Bad Request."""
    seen = _capture(monkeypatch, {"response": '{"label": "NO_CONFLICT", "reason": ""}'})
    L2._chat_once("p", model_name="gemma3:4b", base_url="http://x")
    assert seen["body"]["think"] is False


def test_retry_drops_format_json(monkeypatch):
    seen = _capture(monkeypatch, {"response": '{"label": "NO_CONFLICT", "reason": ""}'})
    L2._chat_once("p", model_name="m", base_url="u", format_json=True)
    assert seen["body"].get("format") == "json"
    L2._chat_once("p", model_name="m", base_url="u", format_json=False)
    assert "format" not in seen["body"]


def test_stream_truncated_without_done_is_an_error(monkeypatch):
    """A peer that closes mid-stream must raise, never return a short answer.

    Guard for BUG-2026-09-04-A: a partial body accepted as the whole answer is
    how `status: ok, fact_count: 0` records were written for calls that never
    finished.
    """
    class _Truncated(_FakeResp):
        def iter_lines(self, decode_unicode=False):
            yield json.dumps({"response": '{"label": "NO_CON'})

    monkeypatch.setattr(
        EX.requests, "post",
        lambda url, json=None, timeout=None, headers=None, stream=None: _Truncated({}),
    )
    with pytest.raises(EX.requests.ConnectionError):
        L2._chat_once("p", model_name="m", base_url="u")
