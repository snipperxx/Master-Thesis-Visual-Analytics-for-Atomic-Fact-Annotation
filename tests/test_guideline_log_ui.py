"""The diagnosed arm must leave the same record as the control arm.

RQ2 compares $g_{v2}$ with $g_{v2}'$. If the interface's change log and the
control console's diverged in shape, the write-up would be comparing two
formats as much as two revisions — so both go through `src/guideline_log.py`,
and these tests hold the UI's end of that: the same entry fields, the same
"what this save changed" history, the same coalescing.

Hermetic: Flask's test client over a tmp `prompts/`, no Ollama, no SBERT.
"""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.guideline_log import DraftLog  # noqa: E402

# A guideline must NOT carry the frame's sentinels; see BUG-2026-09-18-B.
FRAME_SENTINELS = "<<DOC_TITLE>> <<SECTION_PATH>> <<SECTION_TEXT>>"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    ui_app = pytest.importorskip("ui.app")
    monkeypatch.setattr(ui_app, "PROMPTS", tmp_path)
    (tmp_path / "extract_v1.md").write_text("R1 baseline\n", encoding="utf-8")
    ui_app.app.config["TESTING"] = True
    with ui_app.app.test_client() as c:
        yield c


def test_saving_a_version_checkpoints_it(client, tmp_path):
    body = "R1 baseline\nR2 added by the diagnosis\n"
    assert client.put("/api/guidelines/v2", json={"text": body}).status_code == 200
    snaps = list((tmp_path / "_drafts" / "extract_v2").glob("*.md"))
    assert len(snaps) == 1 and snaps[0].read_text(encoding="utf-8") == body


def test_a_guideline_may_not_carry_the_frame_or_overwrite_the_baseline(client):
    """Both refusals protect invariant 3: only g_v varies between runs."""
    r = client.put("/api/guidelines/v2",
                   json={"text": "R1 x\n" + FRAME_SENTINELS})
    assert r.status_code == 400 and "schema" in r.get_json()["message"]
    assert client.put("/api/guidelines/v1",
                      json={"text": "R1 rewritten\n"}).status_code == 403


def test_a_frozen_arm_is_not_editing_material_in_this_interface(client, tmp_path):
    """One author writes both arms, so their memory of $g_{v2}'$ carries into
    $g_{v2}$ no matter what. The one carryover the tooling *can* prevent is
    re-reading the frozen control at the keyboard where $g_{v2}$ is written:
    before this, `v2_control` sat in the editor's version dropdown and Load
    pasted it in. It stays runnable — its facts are the point of the arm."""
    (tmp_path / "extract_v2_control.md").write_text("R1 ...\n", encoding="utf-8")
    (tmp_path / "extract_v2_control.lock.json").write_text(
        '{"sha256": "x"}', encoding="utf-8")

    listed = {g["version"]: g for g in
              client.get("/api/guidelines").get_json()["guidelines"]}
    assert listed["v2_control"]["locked"] is True      # selectable for a run
    assert listed["v1"]["locked"] is False

    assert client.get("/api/guidelines/v2_control").status_code == 423
    assert client.put("/api/guidelines/v2_control",
                      json={"text": "R1 edited\n"}).status_code == 423
    assert client.delete("/api/guidelines/v2_control").status_code == 423
    assert (tmp_path / "extract_v2_control.md").read_text(
        encoding="utf-8") == "R1 ...\n"


def test_the_entry_carries_what_the_interface_showed(client, tmp_path):
    r = client.post("/api/guideline_log/v2", json={"add": {
        "op": "modify", "rule": "R7",
        "reason": "granularity: qwen splits the coordination, gemma does not",
        "cites": [{"kind": "pair", "doc_id": "train-000004",
                   "pair_key": "f1|f2", "label": "granularity",
                   "annotator": "qwen3.6:35b ↔ gemma-4-31B"}]}})
    assert r.status_code == 200
    entry, = r.get_json()["entries"]
    assert entry["op"] == "modify" and entry["rule"] == "R7"
    # the pair is citable by key, which resolves in data/reviews/<doc>.json
    assert entry["cites"][0]["pair_key"] == "f1|f2"
    assert entry["cites"][0]["label"] == "granularity"
    on_disk = json.loads(
        (tmp_path / "extract_v2_rationale.json").read_text(encoding="utf-8"))
    assert on_disk["entries"] == r.get_json()["entries"]

    # same refusals as the control arm
    assert client.post("/api/guideline_log/v2",
                       json={"add": {"op": "modify", "reason": " "}}).status_code == 400
    assert client.post("/api/guideline_log/v2",
                       json={"add": {"op": "tweak", "reason": "x"}}).status_code == 400
    # and the same withdrawal
    assert client.post("/api/guideline_log/v2",
                       json={"remove": entry["id"]}).get_json()["entries"] == []


def test_history_shows_what_each_save_changed(client, tmp_path):
    for body in ("R1 baseline\n", "R1 baseline\nR2 new rule\n"):
        client.put("/api/guidelines/v2", json={"text": body})
    client.post("/api/guideline_log/v2",
                json={"add": {"op": "new", "rule": "R2", "reason": "because"}})

    hist = client.get("/api/guideline_history/v2").get_json()
    drafts = [i for i in hist["items"] if i["kind"] == "draft"]
    assert hist["notes"] == 1 and len(drafts) == 2
    assert drafts[0]["base"] == "extract_v1.md"      # first is a change to g_v1
    assert drafts[-1]["current"] is True

    detail = client.get("/api/guideline_history/v2?name=" + drafts[-1]["name"])
    body = detail.get_json()
    assert body["base"] == drafts[0]["name"]
    assert "+R2 new rule" in body["diff"]            # only this save's edit
    assert client.get("/api/guideline_history/v2?name=../extract_v1.md"
                      ).status_code == 404


def test_pruning_matches_the_control_arm(client, tmp_path):
    log = DraftLog(draft=tmp_path / "extract_v2.md",
                   seed=tmp_path / "extract_v1.md",
                   snaps_dir=tmp_path / "_drafts" / "extract_v2", prefix="d_")
    t0 = datetime(2026, 9, 18, 17, 29, tzinfo=timezone.utc)
    for i, text in enumerate(("a", "a b", "a b c")):
        log.checkpoint(text, t0 + timedelta(seconds=3 * i), explicit=True)
    log.checkpoint("a b c d", t0 + timedelta(hours=3), explicit=True)

    assert client.get("/api/guideline_history/v2").get_json()["prunable"] == 2
    out = client.post("/api/guideline_history/v2", json={"prune": True}).get_json()
    assert out["kept"] == 2
    kept = sorted(p.read_text(encoding="utf-8") for p in log.snaps())
    assert kept == ["a b c", "a b c d"]     # the burst keeps what it ended in


# --------------------------------------------------------------- screenshots
#
# A reason often points at something a `pair_key` cannot carry — three
# annotators' spans side by side, the shape of a group, a colour pattern across
# a view. That view does not come back: re-running Phase-2 recomputes the
# conflicts and the layout moves. So the picture is stored with the version, and
# these tests hold the three things that make it a record rather than a decoration.

PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6360000002000100ffff03000006000557bfabd400"
    "00000049454e44ae426082")


def _attach(client, version="v2", data=PNG_1PX):
    return client.post(f"/api/guideline_shot/{version}",
                       data={"file": (io.BytesIO(data), "screen.png")},
                       content_type="multipart/form-data")


def test_a_screenshot_is_stored_with_the_version_and_named_by_its_bytes(client, tmp_path):
    r = _attach(client)
    assert r.status_code == 201
    shot = r.get_json()
    # the name the browser sent is discarded: the file is written into a
    # directory this app also serves, so the extension follows the magic bytes
    assert shot["file"].startswith("shot_") and shot["file"].endswith(".png")
    assert (tmp_path / "_shots" / "extract_v2" / shot["file"]).read_bytes() == PNG_1PX

    got = client.get(shot["url"])
    assert got.status_code == 200 and got.mimetype == "image/png"
    assert got.headers["X-Content-Type-Options"] == "nosniff"
    # a name that is not in the shots dir is not a path to follow out of it
    assert client.get("/api/guideline_shot/v2/..%2f..%2fextract_v1.md"
                      ).status_code in (404, 308)
    # v1 is the baseline, not a revision under diagnosis
    assert _attach(client, "v1").status_code == 403


def test_bytes_that_are_not_an_image_are_refused(client, tmp_path):
    r = _attach(client, data=b"<script>alert(1)</script>")
    assert r.status_code == 400
    assert not (tmp_path / "_shots" / "extract_v2").exists()


def test_the_entry_keeps_its_screenshots_and_drops_dead_ones(client, tmp_path):
    shot = _attach(client).get_json()
    r = client.post("/api/guideline_log/v2", json={"add": {
        "op": "new", "rule": "R9", "reason": "the three spans disagree here",
        "shots": [{"file": shot["file"], "caption": "qwen splits, gemma does not"},
                  {"file": "shot_20260920T000000Z.png", "caption": "never stored"}]}})
    entry, = r.get_json()["entries"]
    # an entry naming a file that is not on disk would render as a broken image
    # in the record the write-up reports from — worse than no picture at all
    assert [s["file"] for s in entry["shots"]] == [shot["file"]]
    assert entry["shots"][0]["caption"] == "qwen splits, gemma does not"
    assert json.loads((tmp_path / "extract_v2_rationale.json")
                      .read_text(encoding="utf-8"))["entries"] == [entry]
    assert client.get("/api/guideline_history/v2").get_json(
        )["items"][0]["shots"] == 1


def test_saving_writes_LF_even_on_windows(client, tmp_path):
    """BUG-2026-09-18-C, the other half.

    Normalising the textarea's CRLF on the way in is not enough: `write_text`
    re-translates every "\n" to "\r\n" on Windows, so the saved guideline came
    back 257 bytes larger than its seed and git showed every line as changed.
    The v1 -> v2 diff *is* the intervention; it has to be readable.
    """
    assert client.put("/api/guidelines/v2",
                      json={"text": "R1 a\r\nR2 b\r\n"}).status_code == 200
    raw = (tmp_path / "extract_v2.md").read_bytes()
    assert b"\r" not in raw
    assert raw == b"R1 a\nR2 b\n"


def test_logging_against_an_unchanged_draft_says_so(client, tmp_path):
    """The commonest way to lose an edit: type it, log the reason, never press
    Save. The entry is still recorded — a reason may be written before the edit
    — but the response must say the file does not carry it."""
    client.put("/api/guidelines/v2", json={"text": "R1 baseline\n"})   # == seed
    r = client.post("/api/guideline_log/v2", json={"add": {
        "op": "modify", "rule": "R1", "reason": "tightened the coverage mandate"}})
    assert r.status_code == 200
    assert r.get_json()["draft_matches_seed"] is True

    client.put("/api/guidelines/v2", json={"text": "R1 baseline\nR2 added\n"})
    r2 = client.post("/api/guideline_log/v2", json={"add": {
        "op": "new", "rule": "R2", "reason": "added the passive-object rule"}})
    assert r2.get_json()["draft_matches_seed"] is False


def test_one_press_saves_the_draft_and_files_the_lines_it_changed(client, tmp_path):
    """The control console autosaves, so its log entries always sit on a saved
    draft. Here the button has to do it: save, checkpoint, and record the diff.
    A reason without the lines it explains is an assertion; with them the record
    carries its own evidence, and the write-up reports this file."""
    client.put("/api/guidelines/v2", json={"text": "R1 baseline\n"})
    r = client.post("/api/guideline_log/v2", json={
        "text": "R1 baseline\r\nR2 no empty object slot on a passive\r\n",
        "add": {"op": "new", "rule": "R2",
                "reason": "86 validator drops in the dev pass were this slot"}})
    assert r.status_code == 200
    entry = r.get_json()["entries"][-1]

    # the file moved, in LF, without a separate Save
    saved = (tmp_path / "extract_v2.md").read_bytes()
    assert saved == b"R1 baseline\nR2 no empty object slot on a passive\n"
    # and the entry carries what moved
    assert entry["add"] == 1 and entry["rem"] == 0
    assert "+R2 no empty object slot" in entry["diff"]
    assert entry["base"] == "extract_v1.md"      # first entry: against the seed
    assert entry["unchanged"] is False
    assert (tmp_path / "_drafts" / "extract_v2" / entry["after"]).exists()

    # a second entry is measured against the FIRST entry's checkpoint, not the
    # seed — otherwise every entry would re-report the whole revision
    r2 = client.post("/api/guideline_log/v2", json={
        "text": "R1 baseline\nR2 no empty object slot on a passive\nR3 self-reference\n",
        "add": {"op": "new", "rule": "R3", "reason": "D-12, 5% of qwen's nodes"}})
    e2 = r2.get_json()["entries"][-1]
    assert e2["base"] == entry["after"]
    assert e2["add"] == 1 and "+R3 self-reference" in e2["diff"]
    # R2 appears as CONTEXT, never as an addition: the second entry reports its
    # own edit, not the revision so far
    assert "+R2 no empty" not in e2["diff"]


def test_save_and_log_obeys_the_same_refusals_as_save(client, tmp_path):
    """The log route writes the same file, so it cannot be a way around the
    guards — a second copy of them is a second place for one to go missing."""
    assert client.post("/api/guideline_log/v1", json={
        "text": "R1 rewritten\n",
        "add": {"op": "modify", "rule": "R1", "reason": "x"}}).status_code == 403
    assert (tmp_path / "extract_v1.md").read_text(encoding="utf-8") == "R1 baseline\n"

    assert client.post("/api/guideline_log/v2", json={
        "text": "R1 x\n" + FRAME_SENTINELS,
        "add": {"op": "modify", "rule": "R1", "reason": "x"}}).status_code == 400


def test_a_checkpoint_carries_the_reasons_filed_against_it(client, tmp_path):
    """Opening a save in History must show WHY, not only what.

    The lines are still on disk a year from now; the reasoning only exists in
    the log, so a History detail without it is half a record. Both arms read
    this through `DraftLog.detail`, so the control console gets it too.
    """
    client.put("/api/guidelines/v2", json={"text": "R1 baseline\n"})
    client.post("/api/guideline_log/v2", json={
        "text": "R1 baseline\nR2 no empty object on a passive\n",
        "add": {"op": "new", "rule": "R2",
                "reason": "86 validator drops in the dev pass were this slot",
                "cites": [{"kind": "annotator_segment", "doc_id": "train-000004",
                           "section": "preamble.recitals[0]",
                           "annotator": "cloud:qwen3.6:35b"}]}})

    hist = client.get("/api/guideline_history/v2").get_json()
    drafts = [i for i in hist["items"] if i["kind"] == "draft"]
    logged = [d for d in drafts if d["notes"]]
    assert len(logged) == 1, "the list must say which saves carry a reason"

    detail = client.get(
        "/api/guideline_history/v2?name=" + logged[0]["name"]).get_json()
    note, = detail["notes"]
    assert note["rule"] == "R2" and note["op"] == "new"
    assert "86 validator drops" in note["reason"]
    assert note["cites"][0]["annotator"] == "cloud:qwen3.6:35b"
    assert note["add"] == 1 and note["rem"] == 0

    # a plain Save is a checkpoint, not a recorded change: no reason, and the
    # pane must be able to say so rather than implying one exists
    client.put("/api/guidelines/v2", json={"text": "R1 baseline\nR2 x\nR3 y\n"})
    hist2 = client.get("/api/guideline_history/v2").get_json()
    plain = [i for i in hist2["items"] if i["kind"] == "draft" and not i["notes"]]
    assert plain, "a save with no logged reason must report notes == 0"
    assert client.get("/api/guideline_history/v2?name=" + plain[-1]["name"]
                      ).get_json()["notes"] == []


def test_freezing_locks_the_guideline_and_its_reasons(client, tmp_path):
    """The commit is the point after which the draft IS the artefact.

    The lock carries the sha256 of the text *and* the change log's digest:
    freezing the guideline while leaving its reasons editable would freeze the
    artefact and not the account of it, and RQ2 reports the account.
    """
    client.put("/api/guidelines/v2", json={"text": "R1 baseline\n"})
    # nothing revised yet, and no reason recorded: both are refusals
    assert client.post("/api/guideline_freeze/v2").status_code == 400
    client.post("/api/guideline_log/v2", json={
        "text": "R1 baseline\nR2 no empty object on a passive\n",
        "add": {"op": "new", "rule": "R2", "reason": "86 validator drops"}})

    r = client.post("/api/guideline_freeze/v2")
    assert r.status_code == 201
    lock = r.get_json()["lock"]
    assert lock["entries"] == 1 and lock["words"] > 0
    assert lock["change_log"]["entries"] == 1 and lock["change_log"]["sha256"]
    import hashlib
    on_disk = (tmp_path / "extract_v2.md").read_text(encoding="utf-8")
    assert lock["sha256"] == hashlib.sha256(on_disk.encode("utf-8")).hexdigest()
    assert json.loads((tmp_path / "extract_v2.lock.json")
                      .read_text(encoding="utf-8")) == lock

    # after the freeze: no edit, no re-freeze, and the editor will not open it
    assert client.put("/api/guidelines/v2",
                      json={"text": "R1 changed\n"}).status_code == 423
    assert client.post("/api/guideline_log/v2", json={
        "text": "R1 changed\n",
        "add": {"op": "modify", "rule": "R1", "reason": "x"}}).status_code == 423
    assert client.post("/api/guideline_freeze/v2").status_code == 409
    assert client.get("/api/guidelines/v2").status_code == 423
    assert client.post("/api/guideline_freeze/v1").status_code == 403
    # and it stays runnable — its facts are the point of the arm
    listed = {g["version"]: g for g in
              client.get("/api/guidelines").get_json()["guidelines"]}
    assert listed["v2"]["locked"] is True
