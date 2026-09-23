"""The control console's change log must freeze with the draft it explains.

`g_v2'` is a control only if it was written before the diagnosis and not
revisited (CLAUDE.md invariant 2). The lock enforces that for the guideline
text; these tests hold the same line for the change log, which is part of the
same artefact — a reason added after the conflict views were opened is exactly
the leak the arm exists to exclude. They also guard the two things that make an
entry worth keeping at all: a reason, and a citation that resolves back into
`data/facts/<model>/<doc>.json` rather than into a scroll position.

Hermetic: an HTTP server on an ephemeral port over a tmp_path, no Ollama, no
SBERT, no reading of the real prompts/ tree.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "serve_control_console", ROOT / "scripts" / "serve_control_console.py")
scc = importlib.util.module_from_spec(_spec)
sys.modules["serve_control_console"] = scc
_spec.loader.exec_module(scc)


@pytest.fixture()
def console(tmp_path, monkeypatch):
    """A running console whose whole world is tmp_path."""
    seed = tmp_path / "extract_v1.md"
    seed.write_text("# g_v1\nR1 ...\n", encoding="utf-8")
    reader = tmp_path / "baseline_reader.html"
    reader.write_text("<h1>reader</h1>", encoding="utf-8")
    monkeypatch.setattr(scc, "ROOT", tmp_path)
    monkeypatch.setattr(scc, "READER", reader)
    monkeypatch.setattr(scc, "SEED_FROM", seed)
    monkeypatch.setattr(scc, "DRAFT", tmp_path / "extract_v2_control.md")
    monkeypatch.setattr(scc, "LOCK", tmp_path / "extract_v2_control.lock.json")
    monkeypatch.setattr(scc, "SNAPS", tmp_path / "_control_drafts")
    monkeypatch.setattr(scc, "RATIONALE",
                        tmp_path / "extract_v2_control_rationale.json")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), scc.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:%d" % srv.server_address[1]

    def call(path, payload=None, method=None):
        req = urllib.request.Request(
            base + path,
            data=None if payload is None else json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method=method or ("GET" if payload is None else "POST"))
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    try:
        yield call, tmp_path
    finally:
        srv.shutdown()
        srv.server_close()


ENTRY = {
    "op": "modify",
    "rule": "R4",
    "reason": "Nemotron drops the object on passives; g_v1 does not say what to "
              "put there.",
    "cites": [{"doc_id": "train-000004", "section": "enacting.article_3",
               "annotator": "Nemotron-30B", "group_id": "d0a70809dfcd6b47",
               "excerpt": "The refund shall be fixed."}],
}


def test_entry_records_the_citation_that_resolves_back_to_the_facts(console):
    call, tmp = console
    status, body = call("/api/rationale", {"add": ENTRY})
    assert status == 200
    e, = body["entries"]
    assert e["op"] == "modify" and e["rule"] == "R4"
    assert e["at"].endswith("Z") and e["id"]
    # doc + section + annotator + group_id: the four fields a reader needs to
    # find the annotation again in data/facts/<model>/<doc>.json
    assert e["cites"][0]["group_id"] == "d0a70809dfcd6b47"
    assert e["cites"][0]["annotator"] == "Nemotron-30B"
    assert e["cites"][0]["section"] == "enacting.article_3"
    on_disk = json.loads(
        (tmp / "extract_v2_control_rationale.json").read_text(encoding="utf-8"))
    assert on_disk["entries"] == body["entries"]


def test_a_reason_is_required_and_the_op_is_closed(console):
    call, _ = console
    status, body = call("/api/rationale", {"add": dict(ENTRY, reason="   ")})
    assert status == 400 and "reason" in body["error"]
    status, body = call("/api/rationale", {"add": dict(ENTRY, op="tweak")})
    assert status == 400
    assert call("/api/rationale")[1]["entries"] == []


def test_a_reason_written_after_the_edit_says_so(console):
    """g_v2' was frozen at 18:15 on 2026-09-18 with 3 of its 5 rule additions
    unexplained; the freeze was broken to write them. A late reason is worth
    having, but the record must not claim it was there at the time — `at` stays
    the moment of writing, and `documents` names the checkpoint it explains."""
    call, _ = console
    _, body = call("/api/rationale", {"add": dict(
        ENTRY, documents="gv2control_20260918T175730Z.md")})
    e, = body["entries"]
    assert e["written_late"] is True
    assert e["documents"] == "gv2control_20260918T175730Z.md"
    assert e["at"].startswith(datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    # an ordinary entry carries neither field — absence is the normal case
    _, body = call("/api/rationale", {"add": ENTRY})
    assert "written_late" not in body["entries"][1]


def test_an_entry_can_be_withdrawn_before_the_commit(console):
    call, _ = console
    _, body = call("/api/rationale", {"add": ENTRY})
    eid = body["entries"][0]["id"]
    assert call("/api/rationale", {"remove": eid})[1]["entries"] == []
    assert call("/api/rationale", {"remove": eid})[0] == 404


def test_a_pressed_save_always_answers_even_when_nothing_changed(console):
    """A silent button reads as a broken button; it cost Zilong a lost note."""
    call, _ = console
    for _ in range(2):
        status, body = call("/api/guideline", {"text": "same", "explicit": True})
        assert status == 200 and body["saved_at"]


def test_history_keeps_pressed_checkpoints_and_rolls_up_autosave_bursts(console):
    call, tmp = console
    # three autosaves in the same burst: one checkpoint, holding the last text
    for t in ("a", "a b", "a b c"):
        call("/api/guideline", {"text": t})
    snaps = sorted((tmp / "_control_drafts").glob("*.md"))
    assert len(snaps) == 1 and snaps[0].read_text(encoding="utf-8") == "a b c"

    # a pressed Save is a checkpoint of its own, and identical text adds none
    call("/api/guideline", {"text": "a b c d", "explicit": True})
    call("/api/guideline", {"text": "a b c d", "explicit": True})
    assert len(list((tmp / "_control_drafts").glob("*.md"))) == 2

    call("/api/rationale", {"add": ENTRY})
    _, hist = call("/api/history")
    assert hist["drafts"] == 2 and hist["notes"] == 1
    assert [i["kind"] for i in hist["items"]].count("note") == 1
    assert hist["items"][0]["at"] <= hist["items"][-1]["at"]     # oldest first
    drafts = [i for i in hist["items"] if i["kind"] == "draft"]
    assert drafts[-1]["current"] is True and drafts[-1]["d_chars"] == 2

    # no path traversal
    assert call("/api/history?name=../extract_v1.md")[0] == 404


def test_a_checkpoint_shows_what_that_save_changed(console):
    """The question is "what did this edit do", not "how old is this text"."""
    call, _ = console
    call("/api/guideline", {"text": "R1 keep\nR2 old\n", "explicit": True})
    call("/api/guideline", {"text": "R1 keep\nR2 new\n", "explicit": True})
    call("/api/guideline", {"text": "R1 keep\nR2 new\nR3 added\n",
                            "explicit": True})
    drafts = [i for i in call("/api/history")[1]["items"]
              if i["kind"] == "draft"]

    first, second = call("/api/history?name=" + drafts[0]["name"])[1], \
        call("/api/history?name=" + drafts[1]["name"])[1]
    # the first checkpoint is a change *to g_v1*, not to nothing
    assert first["base"] == "extract_v1.md" and "+R1 keep" in first["diff"]
    # the second shows only its own edit — R1 is context, R3 is not there yet
    assert "-R2 old" in second["diff"] and "+R2 new" in second["diff"]
    assert "R3 added" not in second["diff"]
    assert second["base"] == drafts[0]["name"]
    # ... while the distance to today's text is the other view, kept separate
    assert "+R3 added" in second["diff_current"]
    assert drafts[1]["add"] == 1 and drafts[1]["rem"] == 1


def test_pruning_collapses_a_burst_to_the_state_it_ended_in(console):
    """Per-keystroke autosaves from before the coalescing rule read as noise."""
    call, tmp = console
    snapdir = tmp / "_control_drafts"
    snapdir.mkdir()
    # a burst of three seconds apart, then one a day later
    for stamp, text in (("20260918T172901Z", "a"), ("20260918T172904Z", "a b"),
                        ("20260918T172907Z", "a b c"),
                        ("20260919T090000Z", "a b c d")):
        (snapdir / ("gv2control_%s.md" % stamp)).write_text(text,
                                                            encoding="utf-8")
    assert call("/api/history")[1]["prunable"] == 2

    status, body = call("/api/history", {"prune": True})
    assert status == 200 and body["kept"] == 2
    kept = sorted(p.name for p in snapdir.glob("*.md"))
    assert kept == ["gv2control_20260918T172907Z.md",
                    "gv2control_20260919T090000Z.md"]
    # the newest of the burst survives, so nothing a save produced is lost
    assert (snapdir / kept[0]).read_text(encoding="utf-8") == "a b c"
    assert call("/api/history")[1]["prunable"] == 0


def test_the_commit_freezes_the_log_with_the_draft(console):
    call, tmp = console
    call("/api/rationale", {"add": ENTRY})
    status, body = call("/api/guideline", {"text": "# g_v2'\nR1 ...\n",
                                           "commit": True})
    assert status == 200
    # the lock carries the log's digest, so a later edit of the json is visible
    assert body["lock"]["change_log"]["entries"] == 1
    assert len(body["lock"]["change_log"]["sha256"]) == 64

    # invariant 2: after the commit neither half of the artefact may move
    assert call("/api/guideline", {"text": "later thought"})[0] == 409
    assert call("/api/rationale", {"add": ENTRY})[0] == 409
    assert call("/api/rationale", {"remove": "e001-000000"})[0] == 409
    assert len(call("/api/rationale")[1]["entries"]) == 1
