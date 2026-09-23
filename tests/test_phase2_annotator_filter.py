"""Regression guard for BUG-2026-09-10-A.

`data/facts/` holds the three retired local 4B pilot directories beside the
three cloud 30B annotators frozen on 2026-09-03. Both discovery helpers swept
every subdirectory, so a Phase-2 run — and the UI, which calls the same two
functions — would have built a SIX-annotator conflict graph and recorded it as
the v1 dev baseline, with no error anywhere. The filter is therefore a default,
and `annotators=None` is the explicit opt-out.
"""
import json

import pytest

from scripts.run_phase2 import (ANNOTATORS_OF_RECORD, discover_all_doc_ids,
                                discover_annotators)

RETIRED = "qwen3.5:4b"


def _write(root, model, doc_id, n):
    d = root / model.replace(":", "_").replace("/", "_")
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{doc_id}.json").write_text(json.dumps({
        "doc_id": doc_id, "annotator": model, "guideline_version": "v1",
        "fact_count": n,
        "facts": [{"fact_id": f"{model}-{i}", "subject": "a", "predicate": "b",
                   "object": "c"} for i in range(n)],
    }), encoding="utf-8")


@pytest.fixture
def facts_root(tmp_path):
    root = tmp_path / "facts"
    for m in ANNOTATORS_OF_RECORD:
        _write(root, m, "train-000001", 3)
    _write(root, RETIRED, "train-000001", 3)
    # a doc only the retired pilot covered
    _write(root, RETIRED, "train-000002", 3)
    _write(root, "gemma3:4b", "train-000002", 3)
    return root


def test_retired_annotators_are_excluded_by_default(facts_root):
    got = discover_annotators(facts_root, "train-000001")
    assert set(got) == set(ANNOTATORS_OF_RECORD)
    assert RETIRED not in got


def test_opt_out_still_sees_everything(facts_root):
    got = discover_annotators(facts_root, "train-000001", annotators=None)
    assert RETIRED in got
    assert len(got) == len(ANNOTATORS_OF_RECORD) + 1


def test_doc_discovery_ignores_docs_only_retired_models_covered(facts_root):
    assert discover_all_doc_ids(facts_root) == ["train-000001"]
    assert discover_all_doc_ids(facts_root, None) == ["train-000001",
                                                      "train-000002"]


def test_explicit_subset_is_honoured(facts_root):
    one = ANNOTATORS_OF_RECORD[:2]
    got = discover_annotators(facts_root, "train-000001", annotators=one)
    assert set(got) == set(one)
