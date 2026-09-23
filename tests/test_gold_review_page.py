"""The review page must carry a finished decision back as *finished*.

Companion to `test_apply_gold_review.py`, which guards the other half of the same
contract. There, `src: "machine"` keeps an unconfirmed decision from being
stamped human. Here, its *absence* is what makes a sweep resumable: after a
partial `--mode human` apply, `make_gold_review_page.py` runs again and the facts
Zilong already signed off must come back confirmed. While they did not
(BUG-2026-08-16-A) they returned as `unreviewed`, so the page asked for them a
second time and a download taken afterwards would have written `unreviewed` back
over his decision.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "make_gold_review_page", ROOT / "scripts" / "make_gold_review_page.py")
mgrp = importlib.util.module_from_spec(_spec)
sys.modules["make_gold_review_page"] = mgrp
_spec.loader.exec_module(mgrp)


def _fact(**kw) -> dict:
    return {"gold_id": "d::seg::f01", **kw}


def test_unreviewed_fact_gets_no_entry():
    assert mgrp.prefill_entry(_fact()) is None
    assert mgrp.prefill_entry(_fact(status="unreviewed")) is None


def test_machine_decision_is_carried_but_marked_unconfirmed():
    e = mgrp.prefill_entry(_fact(status="accepted", reviewer_note="D-12",
                                 review={"by": "claude-opus-5", "mode": "machine"}))
    assert e["status"] == "accepted"
    assert e["src"] == "machine"          # the provenance marker apply_gold_review reads
    assert e["by"] == "claude-opus-5"
    assert e["note"] == "D-12"


def test_human_decision_comes_back_confirmed():
    e = mgrp.prefill_entry(_fact(status="edited",
                                 review={"by": "annotator", "mode": "human"}))
    assert "src" not in e                 # BUG-2026-08-16-A: this is the resume path
    assert e["status"] == "edited"


def test_edits_are_not_replayed():
    """Applied edits already live in the fact's slots; replaying them as pending
    edits would send the old text back on the next download."""
    e = mgrp.prefill_entry(_fact(status="edited", review={"mode": "machine"}))
    assert e["edits"] == {}


def _facts_root(tmp_path, triples: list[dict]):
    """A one-fact gold root on a real sampled segment, so segment_text resolves."""
    seg = mgrp.gs.load_sample()["segments"][0]
    (tmp_path / "doc.json").write_text(json.dumps({
        "schema_version": 1, "doc_id": seg["doc_id"], "doc_title": seg["doc_title"],
        "segments": [{"segment_id": seg["segment_id"],
                      "section_path": seg["section_path"],
                      "facts": [{"gold_id": seg["segment_id"] + "::f01",
                                 "natural_language": "x", "source_quote": "x",
                                 "condition": "", "temporal_context": "",
                                 "logic_group": "", "logic_op": "",
                                 "boilerplate": False, "triples": triples}]}],
    }), encoding="utf-8")
    return mgrp.build_payload(tmp_path)


def test_a_whole_triple_chain_reaches_the_page(tmp_path):
    """The gold schema lets one fact carry the N triples its source span
    decomposes into (schema.py `group_id`). The page rendered `triples[0]` only,
    so a chain was invisible to the reviewer and an edit could only ever reach
    its first link."""
    chain = [{"subject": "a", "predicate": "p", "object": "b"},
             {"subject": "b", "predicate": "q", "object": "c"},
             {"subject": "c", "predicate": "r", "object": "d"}]
    payload = _facts_root(tmp_path, chain)
    assert payload["segments"][0]["facts"][0]["triples"] == chain
    assert payload["n_triples"] == 3 and payload["n_facts"] == 1


def test_a_non_default_facts_root_gets_its_own_browser_storage(tmp_path):
    """Progress lives in localStorage under one key. A preview built from another
    facts root sharing that key would mix its decisions into the real sweep."""
    payload = _facts_root(tmp_path, [{"subject": "a", "predicate": "p", "object": "b"}])
    assert payload["store_key"] != "gold-review-v1"
    assert mgrp.build_payload()["store_key"] == "gold-review-v1"


def test_a_bumped_epoch_invalidates_the_browser_copy(tmp_path):
    """Resetting a sweep has to work without touching disk: the facts are already
    `unreviewed` there, and what has to be discarded is the accept/edit/reject the
    reviewer clicked into localStorage. Keying the store by epoch is the whole
    mechanism, so a bump that did not change the key would silently do nothing."""
    one = mgrp.build_payload(mgrp.GOLD / "facts", epoch=1)["store_key"]
    two = mgrp.build_payload(mgrp.GOLD / "facts", epoch=2)["store_key"]
    assert one == "gold-review-v1" and two == "gold-review-v1#2"


def test_the_page_loads_the_machine_review_of_its_own_facts_root(tmp_path):
    """BUG-2026-08-17-A. `review_facts_hires.html` was badged with the flags
    computed on `data/gold/facts`, whose triples are a different decomposition of
    the same spans — so it showed 44 warnings, of which 23 had been settled in the
    decision log and most of the rest had been fixed by the re-decomposition being
    reviewed. A per-root review file is the fix; reading the default one for every
    root is the bug."""
    payload = _facts_root(tmp_path, [{"subject": "a", "predicate": "p", "object": "b"}])
    # tmp_path has no machine_review_<root>.json, so no flags may appear even
    # though data/gold/machine_review.json exists and covers the same segment.
    assert payload["has_review"] is False
    assert all(not f["flags"] for s in payload["segments"] for f in s["facts"])
    assert mgrp.build_payload()["has_review"] is True
