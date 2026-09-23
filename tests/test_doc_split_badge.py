"""The interface must not open on a locked eval document.

Invariant 1: `data/doc_lists/eurlex_eval.json` is touched once per guideline
version, only to record a result. The UI lists all 50 parsed documents in one
dropdown, sorted, and `train-000000` — the first of them — is an eval document.
So every fresh open landed on a document that is empty by design, with an
"Extract here" button over it (BUG-2026-09-20-A).

The front-end half (default selection, 🔒 in the option label, the red EVAL
badge) is driven entirely by the `split` field these tests pin.

Hermetic: Flask's test client over a tmp data root, no Ollama, no SBERT.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    ui_app = pytest.importorskip("ui.app")
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    for doc_id, title in (("train-000000", "a locked eval act"),
                          ("train-000004", "a dev act")):
        (parsed / f"{doc_id}.json").write_text(json.dumps(
            {"document": {"title": title, "source": "eurlex",
                          "preamble_text": "", "enacting_text": ""}}),
            encoding="utf-8")
    lists = tmp_path / "doc_lists"
    lists.mkdir()
    (lists / "eurlex_eval.json").write_text(
        json.dumps(["train-000000"]), encoding="utf-8")
    (lists / "phase1_dev_only.json").write_text(
        json.dumps(["train-000004"]), encoding="utf-8")
    monkeypatch.setattr(ui_app, "DATA", tmp_path)
    monkeypatch.setattr(ui_app, "PARSED", parsed)
    monkeypatch.setattr(ui_app, "_SPLIT_CACHE", {})   # built once per process
    ui_app.app.config["TESTING"] = True
    with ui_app.app.test_client() as c:
        yield c


def test_the_doc_list_says_which_split_each_document_is_in(client):
    docs = {d["doc_id"]: d for d in client.get("/api/docs").get_json()["docs"]}
    assert docs["train-000000"]["split"] == "eval"
    assert docs["train-000004"]["split"] == "dev"


def test_a_document_in_neither_list_is_not_claimed_for_one(client, tmp_path):
    """Silence, not a guess. An uploaded document is in no split, and marking it
    'dev' would licence diagnosing on something the design never sampled."""
    (tmp_path / "parsed" / "user-000001.json").write_text(json.dumps(
        {"document": {"title": "pasted text", "source": "user",
                      "preamble_text": "", "enacting_text": ""}}),
        encoding="utf-8")
    docs = {d["doc_id"]: d for d in client.get("/api/docs").get_json()["docs"]}
    assert docs["user-000001"]["split"] == ""


def test_a_locked_eval_document_cannot_be_re_extracted_from_the_interface(client):
    """Invariant 1, enforced where it can actually be broken.

    The badge and the disabled button are reminders; this is the refusal. The
    eval pass is a recorded batch run of `scripts/run_extraction.py` — an
    interactive per-clause re-run while looking at the result is tuning against
    the locked set whatever it is called.
    """
    r = client.post("/api/reextract_span", json={
        "doc_id": "train-000000", "model": "cloud:qwen3.6:35b",
        "section_path": "preamble.recitals[0]"})
    assert r.status_code == 403
    assert "eurlex_eval.json" in r.get_json()["message"]

    # a dev document is refused only for the ordinary reasons (here: no parsed
    # section by that path), never by the split guard
    r2 = client.post("/api/reextract_span", json={
        "doc_id": "train-000004", "model": "cloud:qwen3.6:35b",
        "section_path": "preamble.recitals[0]"})
    assert r2.status_code != 403
