"""`extractor._post_stream`: the streamed generate call, and why it exists.

The DBvis-hosted Ollama sits behind a gateway that closes any connection which
has sent **no bytes for ~50s** (BUG-2026-08-31-D). Measured 2026-09-07 against
`cloud:qwen3.6:35b`, warm: a 400-line count-up was cut with
`RemoteDisconnected` at 50.1s with `stream: false`, and completed in 84.7s with
`stream: true` (first chunk at 22.7s). Prompt size is not the variable — 3061
prompt tokens evaluate in 1.5s — generation wall-clock is. So every generate
call in this codebase streams, and these tests pin the two properties that
makes correct: the chunks are folded back into one non-streamed-shaped body,
and a stream that stops without `done` is an ERROR, not a short answer.
"""

from __future__ import annotations

import json

import pytest
import requests

import src.extractor as EX


class _Streamed:
    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def close(self):
        pass

    def iter_lines(self, decode_unicode=False):
        yield from self._lines


def _patch(monkeypatch, lines):
    seen = {}

    def fake_post(url, json=None, timeout=None, headers=None, stream=None):
        seen["url"], seen["body"], seen["stream"] = url, json, stream
        return _Streamed(lines)

    monkeypatch.setattr(EX.requests, "post", fake_post)
    return seen


def test_chunks_are_folded_into_one_body(monkeypatch):
    seen = _patch(monkeypatch, [
        json.dumps({"response": '{"fac'}),
        "",  # keep-alive blank line
        json.dumps({"response": 'ts": []}', "done": True, "eval_count": 7}),
    ])
    body = EX._post_stream("http://x", "/api/generate", {"model": "m", "prompt": "p"})
    assert body["response"] == '{"facts": []}'
    assert body["eval_count"] == 7
    assert body["_stream_chunks"] == 2
    # stream must be requested on BOTH the HTTP call and the Ollama payload.
    assert seen["stream"] is True
    assert seen["body"]["stream"] is True


def test_thinking_channel_is_concatenated_too(monkeypatch):
    _patch(monkeypatch, [
        json.dumps({"thinking": "one "}),
        json.dumps({"thinking": "two", "done": True}),
    ])
    body = EX._post_stream("http://x", "/api/generate", {"model": "m"})
    assert body["thinking"] == "one two"
    assert body["response"] == ""


def test_stream_without_done_raises_rather_than_returning_a_short_answer(monkeypatch):
    """Guard for BUG-2026-09-04-A. A truncated body accepted as complete is how
    14 `status: ok, fact_count: 0` manifest records were written for calls that
    never returned anything."""
    _patch(monkeypatch, [json.dumps({"response": '{"facts": [{"sub'})])
    with pytest.raises(requests.ConnectionError):
        EX._post_stream("http://x", "/api/generate", {"model": "m"},
                        transient_retries=0)


def test_mid_stream_error_chunk_raises(monkeypatch):
    _patch(monkeypatch, [json.dumps({"error": "model requires more system memory"})])
    with pytest.raises(RuntimeError, match="mid-stream"):
        EX._post_stream("http://x", "/api/generate", {"model": "m"},
                        transient_retries=0)


def test_non_json_chunk_is_skipped_not_fatal(monkeypatch):
    _patch(monkeypatch, [
        "<!-- gateway noise -->",
        json.dumps({"response": "ok", "done": True}),
    ])
    assert EX._post_stream("http://x", "/api/generate", {"model": "m"})["response"] == "ok"


# ---------------------------------------------------------------------------
# The corrective message a rejected extraction sends back to the model
# ---------------------------------------------------------------------------

def test_no_triples_names_triples_not_subject():
    """An annotation with no triples must be told to fill in `triples`.

    Before 2026-09-07 the empty/missing-`triples` case fell through to the
    legacy single-triple branch, so the retry prompt said `missing required
    field "subject"` — i.e. it asked the model to FLATTEN the group instead of
    to supply the triples it had dropped. `cloud:qwen3.6:35b` burned all three
    attempts on `train-000004 enacting.article_1` that way and the section was
    lost.
    """
    raw = json.dumps({"facts": [{"natural_language": "The Commission shall act.",
                                 "source_quote": "The Commission shall act.",
                                 "triples": []}]})
    with pytest.raises(EX.ExtractionError) as err:
        EX._parse_facts_response(raw)
    assert "triples" in str(err.value)
    assert "subject" in str(err.value)  # still says what a triple contains


def test_flat_legacy_triple_still_accepted():
    """The legacy shape is a documented input; the stricter branch must not
    break it."""
    raw = json.dumps({"facts": [{"natural_language": "n", "source_quote": "q",
                                 "subject": "s", "predicate": "p", "object": "o"}]})
    groups = EX._parse_facts_response(raw)
    assert groups[0]["triples"] == [{"subject": "s", "predicate": "p", "object": "o",
                                     "condition": "", "temporal_context": ""}]


def test_a_stream_cut_mid_transfer_is_retried(monkeypatch):
    """BUG-2026-09-07-E: `ChunkedEncodingError` is a SIBLING of
    `ConnectionError` under `RequestException`, not a subclass, so a retry
    ladder that catches only `ConnectionError` treats a stream that died
    mid-transfer as fatal. `train-000124 preamble.recitals[0]` was skipped on
    "Response ended prematurely" with two retries unused."""
    attempts = {"n": 0}

    class _Dying(_Streamed):
        def iter_lines(self, decode_unicode=False):
            yield json.dumps({"response": "{"})
            raise requests.exceptions.ChunkedEncodingError("Response ended prematurely")

    def fake_post(url, json=None, timeout=None, headers=None, stream=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return _Dying({})
        return _Streamed([json_mod.dumps({"response": "ok", "done": True})])

    import json as json_mod
    monkeypatch.setattr(EX.requests, "post", fake_post)
    monkeypatch.setattr(EX.time, "sleep", lambda *_: None)
    body = EX._post_stream("http://x", "/api/generate", {"model": "m"})
    assert body["response"] == "ok"
    assert attempts["n"] == 2, "the cut stream must cost a retry, not the call"


def test_a_4xx_is_not_retried(monkeypatch):
    """The other side of the same rule: a client error is our bug, and
    retrying it just repeats it."""
    calls = {"n": 0}

    def fake_post(url, json=None, timeout=None, headers=None, stream=None):
        calls["n"] += 1
        return _Streamed([], status_code=404)

    monkeypatch.setattr(EX.requests, "post", fake_post)
    with pytest.raises(requests.HTTPError):
        EX._post_stream("http://x", "/api/chat", {"model": "m"})
    assert calls["n"] == 1
