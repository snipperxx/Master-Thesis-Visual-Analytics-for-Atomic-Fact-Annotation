"""Shared fixtures for Phase-2 unit tests.

The fixtures here invent small in-memory facts so the tests are
hermetic — they don't touch data/facts/, don't need SBERT to be
installed, and run in milliseconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `src.*` importable from tests/ without an editable install.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _f(fid, ann, subj, pred, obj, sec="preamble.recitals[0]", cs=0, ce=20, nl=None):
    """Build a fact dict compatible with what runs through Phase-2."""
    return {
        "fact_id": fid,
        "doc_id": "test-doc",
        "annotator": ann,
        "guideline_version": "v1",
        "subject": subj,
        "predicate": pred,
        "object": obj,
        "natural_language": nl or f"{subj} {pred} {obj}.",
        "source_locator": {
            "source_type": "prose",
            "section_path": sec,
            "char_start": cs,
            "char_end": ce,
            "quote": f"{subj} {pred} {obj}",
        },
        "conflict_label": "unlabeled",
        "extra": {},
    }


@pytest.fixture
def fact_factory():
    return _f


@pytest.fixture
def two_annotators():
    """Three pairs: identical / paraphrase / contradiction."""
    a = [
        _f("a1", "A", "the Council", "decided", "X exists", cs=0, ce=30),
        _f("a2", "A", "deficit", "is", "2.3% of GDP", cs=40, ce=70),
        _f("a3", "A", "Decision X", "is abrogated", "", cs=100, ce=140),
    ]
    b = [
        _f("b1", "B", "the Council", "decided", "X exists", cs=0, ce=30),
        _f("b2", "B", "deficit", "is", "2.8% of GDP", cs=40, ce=70),
        _f("b3", "B", "Decision X", "shall be abrogated", "", cs=100, ce=140),
    ]
    return {"A": a, "B": b}
