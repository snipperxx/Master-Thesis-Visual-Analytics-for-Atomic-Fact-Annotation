"""Guards for the EUR-Lex RQ2 scorer.

The one that matters is segment scoping: the gold covers 59 sampled segments,
so a model fact anchored anywhere else must be set aside, not counted as a
false positive. Getting that wrong silently deflates precision for every
annotator by roughly the corpus's unsampled share, which looks like a model
result and is not one.
"""

from __future__ import annotations

import json

import pytest

from src import gold_eurlex_eval as ge
from src import gold_store as gs
from src import reextract_worker as rw


def test_eval_doc_ids_are_the_twelve_locked_docs():
    ids = ge.eval_doc_ids()
    assert len(ids) == 12
    assert len(set(ids)) == 12


def test_gold_by_segment_keys_are_section_paths():
    by_sec = ge.gold_by_segment("train-000005")
    assert by_sec, "train-000005 has sampled segments with gold facts"
    for section_path, triples in by_sec.items():
        assert section_path.startswith(("preamble.", "enacting."))
        for t in triples:
            assert t["subject"] and t["predicate"] and t["object"]
            assert t["segment_id"].endswith(section_path)


def test_boilerplate_excluded_by_default():
    # D-03: the closing formula is annotated and tagged, not scored by default.
    default = sum(len(v) for v in ge.gold_by_segment("train-000005").values())
    with_bp = sum(len(v) for v in
                  ge.gold_by_segment("train-000005", include_boilerplate=True).values())
    assert with_bp > default


# These two guards used to read whatever happened to sit under `data/facts/`
# for an eval document. The only extraction the locked eval set ever had was the
# retired local-4B pilot, so the guards silently depended on it — and archiving
# it (2026-09-20) broke them. A guard for the scorer must not require the eval
# set to have been touched; it now builds its own facts root in tmp.

@pytest.fixture()
def one_annotator(tmp_path, monkeypatch):
    """A facts root holding one annotator: two facts in a sampled segment, one
    in a segment the gold never sampled. Returns (doc, annotator, sampled)."""
    doc = "train-000005"
    sampled = sorted(ge.gold_by_segment(doc))
    assert sampled, "train-000005 has sampled segments with gold facts"
    in_scope = sampled[0]
    off_scope = "preamble.recitals[9999]"        # never sampled, by construction
    annotator = "cloud:qwen3.6:35b"              # of the frozen set, so no filter
    root = tmp_path / "facts"                    # drops it
    d = root / "cloud_qwen3.6_35b"
    d.mkdir(parents=True)
    (d / f"{doc}.json").write_text(json.dumps({"annotator": annotator, "facts": [
        {"fact_id": "a1", "subject": "the Commission", "predicate": "adopted",
         "object": "this Regulation",
         "source_locator": {"section_path": in_scope}},
        {"fact_id": "a2", "subject": "the Council", "predicate": "fixed",
         "object": "the export refunds",
         "source_locator": {"section_path": in_scope}},
        {"fact_id": "a3", "subject": "a fact", "predicate": "anchored outside",
         "object": "the sample",
         "source_locator": {"section_path": off_scope}},
    ]}), encoding="utf-8")
    monkeypatch.setattr(rw, "FACTS_ROOT", root)
    return doc, annotator, set(sampled)


def test_model_facts_are_bucketed_by_anchor(one_annotator):
    doc, annotator, _ = one_annotator
    by_ann = ge.model_facts_by_segment(doc)
    assert list(by_ann) == [annotator]
    for by_sec in by_ann.values():
        for section_path, facts in by_sec.items():
            for f in facts:
                anchor = (f.get("source_locator") or {}).get("section_path") or ""
                assert anchor == section_path


def test_out_of_scope_facts_are_counted_not_scored(one_annotator):
    """Facts from unsampled segments must not reach the confusion matrix."""
    doc, ann, sampled = one_annotator
    by_ann = ge.model_facts_by_segment(doc)
    result = ge.evaluate([doc], annotators=[ann])

    scored = result["summary"][ann]["n_facts_scored"]
    out = result["summary"][ann]["n_facts_out_of_scope"]
    in_sampled = sum(len(v) for k, v in by_ann[ann].items() if k in sampled)
    off_sampled = sum(len(v) for k, v in by_ann[ann].items() if k and k not in sampled)

    assert off_sampled == 1 and in_sampled == 2   # the fixture, not an accident
    # `n_facts_scored` counts deduplicated triples, so it cannot exceed the
    # facts that actually sit in a sampled segment — and must never include the
    # ones that do not.
    assert scored <= in_sampled
    assert out == off_sampled
    assert all(r["section_path"] in sampled for r in result["rows"])


@pytest.fixture()
def no_model_facts(tmp_path, monkeypatch):
    """An empty facts root. `train-000005` is a locked evaluation document:
    these guards need only the gold side, and must not score whatever model
    output might one day sit under data/facts for it (CLAUDE.md invariant 1)."""
    monkeypatch.setattr(rw, "FACTS_ROOT", tmp_path / "facts")


def test_result_carries_the_gold_provenance_and_reportability(no_model_facts):
    result = ge.evaluate(["train-000005"])
    st = result["gold_status"]
    assert st["adjudication_mode"] in {"human", "machine", "mixed", "unreviewed"}
    # Reportable requires BOTH a human-adjudicated gold and the real encoder.
    assert result["reportable"] == (st["adjudication_mode"] == "human"
                                    and result["backend"] == "sbert")
    if not result["reportable"]:
        assert "NOT REPORTABLE" in ge.report_markdown(result)


def test_perfect_extraction_scores_one():
    """Feed the gold back in as if a model had produced it."""
    doc = "train-000005"
    gold = ge.gold_by_segment(doc)
    section_path, triples = sorted(gold.items())[0]
    facts = [{"subject": t["subject"], "predicate": t["predicate"],
              "object": t["object"], "fact_id": f"x{i}",
              "source_locator": {"section_path": section_path}}
             for i, t in enumerate(triples)]

    strict = ge.gc.compare(triples, facts)["metrics"]
    assert strict["precision"] == pytest.approx(1.0)
    assert strict["recall"] == pytest.approx(1.0)


# --- BUG-2026-09-21-A: the scorer read the superseded coarse gold root -------
# `data/gold/facts` and `data/gold/facts_hires` describe the same 59 segments
# with the same gold_ids, so scoring against the wrong one yields plausible
# numbers rather than an error (the BUG-2026-08-17-A failure mode). The coarse
# root is still machine-adjudicated; the hi-res root is the human one of
# record. Guard: the default must be the hi-res root, and the root actually
# used must travel with the result.

def test_default_gold_root_is_the_hires_root_of_record():
    assert ge.DEFAULT_GOLD_FACTS == gs.GOLD_FACTS_HIRES
    assert ge.DEFAULT_GOLD_FACTS.name == "facts_hires"


def test_result_records_which_gold_root_was_scored(no_model_facts):
    result = ge.evaluate(["train-000005"])
    assert result["gold_facts_root"] == "data/gold/facts_hires"
    assert "facts_hires" in ge.report_markdown(result)


def test_the_two_gold_roots_are_not_interchangeable():
    """If they ever became identical this guard would be vacuous."""
    doc = "train-000005"
    hires = ge.gold_by_segment(doc, facts_root=gs.GOLD_FACTS_HIRES)
    coarse = ge.gold_by_segment(doc, facts_root=gs.GOLD_FACTS)
    assert coarse, "coarse root missing - guard would pass vacuously"
    n_hires = sum(len(v) for v in hires.values())
    n_coarse = sum(len(v) for v in coarse.values())
    assert n_hires != n_coarse, (
        f"both roots yield {n_hires} triples for {doc}; the cross-wiring this "
        f"test guards against would no longer be detectable")


def test_adjudication_mode_follows_the_root_it_was_given():
    assert ge._gold_status(gs.GOLD_FACTS_HIRES)["adjudication_mode"] == "human"
    assert ge._gold_status(gs.GOLD_FACTS)["adjudication_mode"] != "human"


# --- Many-to-many coverage through the driver (2026-09-23) ------------------
# Under many-to-many, supported predictions and covered gold triples are two
# different counts, so the aggregation must carry both. The old `_accumulate`
# derived precision AND recall from one `n_matched`, which is right only under
# one-to-one. Both the gold root and the facts root are built in tmp, for a doc
# id that exists nowhere else: no real gold, no real model output, and nothing
# of the locked evaluation set, is read.

_FAKE_DOC = "train-900001"          # not a corpus document, by construction
_SEG = "enacting.articles[1].paragraphs[1]"


@pytest.fixture()
def m2m_roots(tmp_path, monkeypatch):
    """One gold segment: a nested fact of two chained triples and a fact of
    one triple (3 gold triples). One annotator: an exact copy of the first
    triple, two distinct predictions of the third, and one unrelated
    prediction (4 predicted triples). Every similarity that matters is forced
    (equal after normalisation, or the gold's bare number), so the expected
    counts hold under SBERT and TF-IDF alike."""
    gold_root = tmp_path / "gold_facts"
    gold_root.mkdir()
    reg = "Regulation (EC) No 1/2000"
    (gold_root / f"{_FAKE_DOC}.json").write_text(json.dumps({"segments": [{
        "segment_id": f"{_FAKE_DOC}::{_SEG}", "section_path": _SEG,
        "facts": [
            {"gold_id": "G1", "status": "accepted", "review": {"mode": "human"},
             "source_quote": "q1",
             "triples": [
                 {"subject": "the Commission", "predicate": "adopts", "object": reg},
                 {"subject": reg, "predicate": "applies to", "object": "the products"}]},
            {"gold_id": "G2", "status": "accepted", "review": {"mode": "human"},
             "source_quote": "q2",
             "triples": [{"subject": "the export refund", "predicate": "is fixed at",
                          "object": "25"}]},
        ]}]}), encoding="utf-8")

    facts_root = tmp_path / "facts"
    d = facts_root / "cloud_qwen3.6_35b"
    d.mkdir(parents=True)
    loc = {"section_path": _SEG}
    (d / f"{_FAKE_DOC}.json").write_text(json.dumps({
        "annotator": "cloud:qwen3.6:35b", "facts": [
            {"fact_id": "p1", "subject": "Commission", "predicate": "adopts",
             "object": reg, "source_locator": loc},
            {"fact_id": "p3", "subject": "Export refund", "predicate": "shall be fixed at",
             "object": "25 EUR", "source_locator": loc},
            {"fact_id": "p4", "subject": "export refund", "predicate": "fixed at",
             "object": "25 euro", "source_locator": loc},
            {"fact_id": "p5", "subject": "zebra", "predicate": "eats",
             "object": "grass", "source_locator": loc},
        ]}), encoding="utf-8")
    monkeypatch.setattr(rw, "FACTS_ROOT", facts_root)
    return gold_root


def test_driver_counts_covered_gold_and_supported_predictions_separately(m2m_roots):
    r = ge.evaluate([_FAKE_DOC], facts_root=m2m_roots)
    assert r["assignment"] == "many_to_many"
    assert r["unit"]["recall"].startswith("gold triple")
    assert r["unit"]["precision"].startswith("predicted triple")
    (ann, s), = r["summary"].items()
    soft = s["soft_micro"]
    assert (soft["n_gold"], soft["n_gold_covered"]) == (3, 2)          # G1a, G2
    assert (soft["n_extracted"], soft["n_pred_supported"]) == (4, 3)   # p1, p3, p4
    assert soft["precision"] == 0.75
    assert soft["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert soft["n_links"] == 3 and soft["n_gold_multi_pred"] == 1
    # Strict: only p1 is string-equal to a gold triple after normalisation.
    strict = s["strict_micro"]
    assert (strict["n_pred_supported"], strict["n_gold_covered"]) == (1, 1)
    # The failure taxonomy survives: the chained gold triple nobody wrote and
    # the unrelated prediction both carry a reason.
    reasons = r["error_reasons"]
    assert sum(v for k, v in reasons.items() if k.startswith("spurious:")) == 1   # p5
    assert sum(v for k, v in reasons.items() if k.startswith("missing:")) == 1    # G1b
    assert not any("already_matched" in k for k in reasons)
    (err,) = r["errors"]
    assert [m["triple"][1] for m in err["missing"]] == ["applies to"]
    assert [s["fact_ids"] for s in err["spurious"]] == [["p5"]]
    md = ge.report_markdown(r)
    assert "Assignment: `many_to_many`" in md and "ablation" not in md


def test_driver_one_to_one_ablation_reproduces_the_old_aggregation(m2m_roots):
    """The old `_rate(tp, fp, fn)` with tp = n_matched: under one_to_one the new
    two-numerator aggregation must give the same figures."""
    r = ge.evaluate([_FAKE_DOC], facts_root=m2m_roots, assignment="one_to_one")
    assert r["assignment"] == "one_to_one"
    (ann, s), = r["summary"].items()
    soft = s["soft_micro"]
    tp = sum(row["soft"]["n_matched"] for row in r["rows"])
    fp = sum(row["soft"]["n_extracted"] - row["soft"]["n_matched"] for row in r["rows"])
    fn = sum(row["soft"]["n_gold"] - row["soft"]["n_matched"] for row in r["rows"])
    assert (tp, fp, fn) == (2, 2, 1)
    assert soft["precision"] == round(tp / (tp + fp), 4) == 0.5
    assert soft["recall"] == round(tp / (tp + fn), 4)
    assert soft["n_pred_supported"] == soft["n_gold_covered"] == tp
    # One of p3/p4 loses to the other under exclusivity, as it always did.
    assert r["error_reasons"].get("spurious:gold_already_matched") == 1
    assert "**ablation**" in ge.report_markdown(r)
    # Strict does not depend on the assignment at all.
    m2m = ge.evaluate([_FAKE_DOC], facts_root=m2m_roots)
    assert s["strict_micro"] == m2m["summary"][ann]["strict_micro"]


def test_driver_refuses_an_unknown_assignment(m2m_roots):
    with pytest.raises(ValueError):
        ge.evaluate([_FAKE_DOC], facts_root=m2m_roots, assignment="greedy")


def test_the_ablation_can_never_overwrite_the_result_of_record():
    out, rep = ge.default_out_paths("v2", "many_to_many")
    assert (out.name, rep.name) == ("rq2_eurlex_v2.json", "rq2_eurlex_v2_errors.md")
    out1, rep1 = ge.default_out_paths("v2", "one_to_one")
    assert out1 != out and rep1 != rep and "one_to_one" in out1.name


def _row(doc, sup, ext, cov, gold):
    from src import gold_eurlex_eval as E
    r = E._rate(sup, ext, cov, gold)
    return {"doc_id": doc, "annotator": "a", "soft": r, "strict": r}


def test_macro_f1_is_per_document_not_per_segment():
    """Pre-registered macro F1 is the mean of per-DOCUMENT F1: a document's
    segments are pooled before averaging, so one tiny segment cannot weigh as
    much as a whole document (regression guard for 2026-09-23)."""
    from src import gold_eurlex_eval as E
    rows = [
        _row("d1", 10, 10, 10, 10),   # d1, big segment: perfect
        _row("d1", 0, 1, 0, 1),       # d1, tiny segment: all wrong
        _row("d2", 0, 5, 0, 5),       # d2: all wrong
    ]
    # d1 pooled: P = R = 10/11 -> F1 = 0.9091; d2: 0 -> mean 0.4545
    assert abs(E._macro_f1(rows, "soft") - 0.4545) < 1e-3
    # the per-segment mean (the old figure) is (1 + 0 + 0) / 3
    assert abs(E._segment_macro_f1(rows, "soft") - 0.3333) < 1e-3
    assert E._macro_f1([], "soft") == 0.0
