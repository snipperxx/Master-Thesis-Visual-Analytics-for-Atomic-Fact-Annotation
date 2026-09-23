"""Gold provenance must be per fact, not per run.

`review.html` downloads ALL 306 entries — the ones a human confirmed and the ones
still carried over from the machine pass, which it marks `src: "machine"`. Before
2026-08-09 `apply_gold_review --mode human` stamped one uniform reviewer over
every entry, so confirming 200 and applying would have recorded all 306 as
human-adjudicated. That is the exact claim CLAUDE.md discipline rule 6 exists to
protect, and the only thing standing between "machine pre-annotation" and the
phrase the thesis is allowed to print.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "apply_gold_review", ROOT / "scripts" / "apply_gold_review.py")
agr = importlib.util.module_from_spec(_spec)
sys.modules["apply_gold_review"] = agr
_spec.loader.exec_module(agr)


MACHINE = {"status": "accepted", "src": "machine", "by": "claude-opus-5"}
HUMAN = {"status": "accepted", "by": "claude-opus-5"}   # src cleared on confirm


def test_machine_entry_stays_machine_in_a_human_run():
    assert agr._entry_mode(MACHINE, "human") == "machine"


def test_confirmed_entry_becomes_human():
    assert agr._entry_mode(HUMAN, "human") == "human"


def test_machine_run_never_promotes_anything():
    assert agr._entry_mode(HUMAN, "machine") == "machine"
    assert agr._entry_mode(MACHINE, "machine") == "machine"


def test_explicit_human_src_is_honoured():
    assert agr._entry_mode({"status": "edited", "src": "human"}, "human") == "human"


def test_missing_decision_is_not_treated_as_machine():
    """An added fact carries no src; it was written by the reviewer."""
    assert agr._entry_mode({}, "human") == "human"
    assert agr._entry_mode(None, "human") == "human"


def test_mixed_kind_does_not_claim_human_adjudication():
    """The phrase the thesis may print must not appear on a partial sweep."""
    mixed = agr.PROVENANCE_KIND["mixed"]
    assert "PARTIALLY" in mixed
    assert mixed != agr.PROVENANCE_KIND["human"]
    # The bare claim "human-adjudicated" must be qualified, not standalone.
    assert "remainder machine-adjudicated" in mixed


def test_all_three_kinds_are_distinct():
    kinds = set(agr.PROVENANCE_KIND.values())
    assert len(kinds) == 3
