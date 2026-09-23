"""Tests for src.review_store (rule parsing + review persistence + summary)."""

from pathlib import Path

import pytest

from src.review_store import (
    load_reviews,
    parse_guideline_rules,
    save_review,
    summarize_reviews,
)

REPO_PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


def test_parse_rules_from_real_v1_guideline():
    """The live guideline's rules must be recoverable, whatever its format.

    Expectations updated 2026-09-04: the guideline body was replaced on
    2026-08-31 (a documented DECISION) with R1-R10 restated from
    data/gold/calibration/_RULES.md, so the previous assertions — 13 rules
    numbered 1..13, titled "Atomicity"/"Conditions ..." — described a text
    that no longer exists. The parser only understood the old `### N.` form
    and returned [] for months of the new one (BUG-2026-09-04-B); this test
    now pins the current form so the same silent breakage cannot recur.
    """
    rules = parse_guideline_rules("v1", REPO_PROMPTS)
    ids = [r["id"] for r in rules]
    assert ids == ["R1", "R1b", "R2", "R3", "R4", "R5", "R6", "R7", "R8",
                   "R9", "R10"]
    assert rules[0]["title"] == "THE PREDICATE CARRIES THE RELATION ONLY"
    assert rules[-1]["title"] == "FLATTENED TABLE ROWS"
    assert all(r["title"] for r in rules)        # no rule loses its title


def test_parse_rules_legacy_heading_form(tmp_path):
    """The legacy `### N. Title` form still parses — g_v2 may reuse it."""
    (tmp_path / "extract_legacy.md").write_text(
        "<guideline>\n"
        "### 1. Atomicity\nbody\n\n"
        "### 6. Conditions and temporal context\nbody\n"
        "</guideline>\n",
        encoding="utf-8",
    )
    rules = parse_guideline_rules("legacy", tmp_path)
    assert [r["id"] for r in rules] == ["1", "6"]
    assert rules[1]["title"] == "Conditions and temporal context"


def test_parse_rules_missing_version():
    assert parse_guideline_rules("nope_does_not_exist", REPO_PROMPTS) == []


def test_save_load_delete_roundtrip(tmp_path):
    root = tmp_path / "reviews"
    out = save_review(root, "doc1", "fa|fb", {
        "rules": ["6", "2", "2"],
        "note": "conditional split",
        "resolution": "agree",
        "label_at_review": "granularity",
        "annotators": ["m1", "m2"],
    })
    assert out["n_reviews"] == 1
    assert out["record"]["rules"] == ["2", "6"]          # deduped + sorted

    loaded = load_reviews(root, "doc1")
    assert "fa|fb" in loaded
    assert loaded["fa|fb"]["label_at_review"] == "granularity"

    out = save_review(root, "doc1", "fa|fb", {"delete": True})
    assert out["deleted"] is True
    assert load_reviews(root, "doc1") == {}


def test_invalid_resolution_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_review(tmp_path, "doc1", "k", {"resolution": "whatever"})
    # relabel with a valid target label is fine
    save_review(tmp_path, "doc1", "k", {"resolution": "relabel:contradiction"})


def test_summarize_across_docs(tmp_path):
    # Rule ids follow the live guideline's scheme (R1..R10 since 2026-08-31),
    # because the titles in the summary are resolved against that file.
    root = tmp_path / "reviews"
    save_review(root, "doc1", "p1", {"rules": ["R7"], "note": "split conditional",
                                     "label_at_review": "granularity"})
    save_review(root, "doc1", "p2", {"rules": ["R7", "R2"], "note": "",
                                     "label_at_review": "contradiction"})
    save_review(root, "doc2", "p3", {"rules": [], "note": "noise",
                                     "label_at_review": "redundancy",
                                     "resolution": "dismiss"})
    out = summarize_reviews(root, prompts_dir=REPO_PROMPTS, guideline_version="v1")
    assert out["n_reviews"] == 3
    assert out["n_docs"] == 2
    assert out["n_unattributed"] == 1
    top = out["rules"][0]
    assert top["id"] == "R7" and top["n_conflicts"] == 2
    assert top["by_label"] == {"granularity": 1, "contradiction": 1}
    assert top["title"]                    # resolved from the live v1 guideline
    assert any(r["id"] == "R2" and r["n_conflicts"] == 1 for r in out["rules"])
