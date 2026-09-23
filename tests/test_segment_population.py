"""RQ1's segment rule (method.tex sec:missing-segments), pre-registered 2026-09-22.

For a comparison between versions v and w, Phi_v and Phi_w are computed over
the segments every annotator returned under BOTH versions, and every other
triple is removed BEFORE alignment. Pinned here:

  * the population is the intersection over annotators AND versions;
  * a section answered with `{"facts": []}` was returned (it has no triples,
    and it stays in);
  * a failed section is removed for every annotator, in both versions;
  * the filter sits before alignment, so the orphans a missing segment causes
    are gone (filtering the output afterwards could not do that);
  * distribution_shift refuses two files computed over different populations;
  * the run's own record decides "failed", never the absence of triples, and
    a record that does not describe the file on disk is refused.

Hermetic: tmp facts roots, no parsed documents, no Ollama.
"""
from __future__ import annotations

import json
import sys

import pytest

from scripts import run_extraction
from scripts import run_phase2
from src import segment_population as sp
from src.distribution_shift import compare_conflict_files

A, B, C = "ann:A", "ann:B", "ann:C"
DOC = "train-900001"
R0, R1, ART1, ART2 = ("preamble.recitals[0]", "preamble.recitals[1]",
                      "enacting.article_1", "enacting.article_2")
SECTIONS = {DOC: [R0, R1, ART1, ART2]}

# Distinct content per section, identical across annotators, so alignment is
# decided by which sections are present and not by the encoder.
TEXT = {
    R0: ("the Council", "adopted", "Regulation (EC) No 1/2001"),
    R1: ("Member States", "shall notify", "the Commission by 1 March 2002"),
    ART1: ("export refunds", "are fixed at", "the amounts in the Annex"),
    ART2: ("this Regulation", "shall enter into force on", "14 June 2001"),
}


def _facts(fact_factory, ann, sections):
    out = []
    for i, sec in enumerate(sections):
        s, p, o = TEXT[sec]
        out.append(fact_factory(f"{ann}-{i}-{sec}", ann, s, p, o, sec=sec))
    return out


def _write(root, label, ann, doc_id, facts, *, failed=(), style="new",
           status=None, fact_count=None):
    """One facts file + its manifest record, as run_extraction writes them.

    style: "new" (failed_sections recorded, 2026-09-23), "error" (only the
    error string and incomplete_sections, i.e. every existing run), "legacy"
    (neither, pre-2026-09-04).
    """
    fp = root / sp.safe_dir(ann) / f"{doc_id}.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    failed_recs = [{"section_path": f, "cause": "unparsed"} for f in failed]
    payload = {"doc_id": doc_id, "annotator": ann, "guideline_version": label,
               "fact_count": len(facts), "facts": facts}
    if style == "new":
        payload["failed_sections"] = failed_recs
    fp.write_text(json.dumps(payload), encoding="utf-8")

    rec = {"model": ann, "doc_id": doc_id, "output_path": fp.resolve().as_posix(),
           "status": status or ("failed" if failed else "ok"),
           "fact_count": len(facts) if fact_count is None else fact_count,
           "guideline_version": label, "error": None}
    if style in ("new", "error"):
        rec["incomplete_sections"] = len(failed)
        if failed:
            rec["error"] = (
                f"INCOMPLETE: {len(failed)} section(s) skipped on persistent network "
                f"failures ({', '.join(f + ' (unparsed)' for f in failed)}); "
                f"{len(facts)} facts were written but the count under-reports — "
                f"re-run this pair")
    if style == "new":
        rec["failed_sections"] = failed_recs
    mpath = root / "manifest.json"
    manifest = (json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists()
                else {"schema_version": 1, "guideline_version": label, "runs": {}})
    manifest["runs"][f"{ann}|{doc_id}"] = rec
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    return fp


@pytest.fixture
def two_versions(tmp_path, fact_factory):
    """v1: C fails R1 (old-style record). v2: A fails ART1 (new-style record).
    Nobody writes a triple for ART2: every model answered `{"facts": []}`."""
    v1, v2 = tmp_path / "facts_v1", tmp_path / "facts_v2"
    for ann in (A, B):
        _write(v1, "v1", ann, DOC, _facts(fact_factory, ann, [R0, R1, ART1]), style="error")
    _write(v1, "v1", C, DOC, _facts(fact_factory, C, [R0, ART1]), failed=[R1], style="error")
    for ann in (B, C):
        _write(v2, "v2", ann, DOC, _facts(fact_factory, ann, [R0, R1, ART1]))
    _write(v2, "v2", A, DOC, _facts(fact_factory, A, [R0, R1]), failed=[ART1])
    return v1, v2


def _build(v1, v2=None, **kw):
    versions = [sp.VersionSource("v1", v1)] + ([sp.VersionSource("v2", v2)] if v2 else [])
    return sp.build_population(versions, SECTIONS, [A, B, C], **kw)


# --- the population ---------------------------------------------------------

def test_population_is_the_intersection_over_annotators_and_both_versions(two_versions):
    pop = _build(*two_versions)
    assert pop["segments"][DOC] == [R0, ART2]
    assert pop["n_segments_universe"] == 4 and pop["n_segments"] == 2
    assert pop["excluded_counts"] == {"v1": {A: 0, B: 0, C: 1}, "v2": {A: 1, B: 0, C: 0}}
    assert pop["excluded_by_cause"] == {"unparsed": 2}
    # Each version alone keeps what only the OTHER version lost.
    assert _build(two_versions[0])["segments"][DOC] == [R0, ART1, ART2]


def test_a_legitimately_empty_section_counts_as_returned(two_versions):
    """ART2 has no triple from anyone, and no run recorded it as failed: the
    models answered `{"facts": []}`. It is IN the population."""
    pop = _build(*two_versions)
    assert ART2 in pop["segments"][DOC]
    assert not [e for e in pop["excluded"] if e["section_path"] == ART2]


def test_old_and_new_records_agree(tmp_path, fact_factory):
    """The error string (every existing run) and failed_sections (from today)
    must yield the same exclusion."""
    got = {}
    for style in ("error", "new"):
        root = tmp_path / style
        for ann in (A, B):
            _write(root, "v1", ann, DOC, _facts(fact_factory, ann, [R0, R1]), style=style)
        _write(root, "v1", C, DOC, _facts(fact_factory, C, [R0]), failed=[R1], style=style)
        got[style] = _build(root)["segments"][DOC]
    assert got["error"] == got["new"] == [R0, ART1, ART2]


def test_error_string_count_must_match_incomplete_sections():
    err = ("INCOMPLETE: 1 section(s) skipped on persistent network failures "
           "(enacting.article_1 (unparsed)); 26 facts were written but the count "
           "under-reports — re-run this pair")
    assert sp.failed_sections_from_error(err, 1) == [
        {"section_path": "enacting.article_1", "cause": "unparsed"}]
    with pytest.raises(sp.PopulationError):
        sp.failed_sections_from_error(err, 2)


def test_legacy_record_cannot_vouch_for_an_absent_section(tmp_path, fact_factory):
    """Before 2026-09-04 a skipped section could be recorded `ok`, so for a
    record with no failure bookkeeping at all an absent section is excluded as
    unverifiable: the one inference the module makes."""
    for ann in (A, B):
        _write(tmp_path, "v1", ann, DOC, _facts(fact_factory, ann, [R0, R1, ART1, ART2]))
    _write(tmp_path, "v1", C, DOC, _facts(fact_factory, C, [R0, R1, ART1]), style="legacy")
    pop = _build(tmp_path)
    assert pop["segments"][DOC] == [R0, R1, ART1]
    assert pop["excluded_by_reason"] == {"legacy_unverified": 1}


def test_document_that_failed_as_a_whole_is_excluded_entirely(tmp_path, fact_factory):
    for ann in (A, B):
        _write(tmp_path, "v1", ann, DOC, _facts(fact_factory, ann, [R0]))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["runs"][f"{C}|{DOC}"] = {"model": C, "doc_id": DOC, "status": "failed",
                                     "output_path": None, "fact_count": 0,
                                     "guideline_version": "v1", "error": "RuntimeError: x"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pop = _build(tmp_path)
    assert pop["segments"][DOC] == []
    assert pop["excluded_by_reason"] == {"document_failed": 4}


def test_unfinished_pass_is_refused_unless_allowed(tmp_path, fact_factory):
    for ann in (A, B):                       # C has not run yet
        _write(tmp_path, "v1", ann, DOC, _facts(fact_factory, ann, [R0]))
    with pytest.raises(sp.PopulationError, match="not finished"):
        _build(tmp_path)
    pop = _build(tmp_path, allow_missing_records=True)
    assert pop["segments"][DOC] == [] and pop["complete_passes"] is False


@pytest.mark.parametrize("breakage", ["fact_count", "failed_with_triples", "label"])
def test_records_that_do_not_describe_the_files_are_refused(tmp_path, fact_factory, breakage):
    for ann in (A, B):
        _write(tmp_path, "v1", ann, DOC, _facts(fact_factory, ann, [R0, R1]))
    if breakage == "fact_count":        # e.g. an interactive re-extraction
        _write(tmp_path, "v1", C, DOC, _facts(fact_factory, C, [R0, R1]), fact_count=7)
    elif breakage == "failed_with_triples":
        _write(tmp_path, "v1", C, DOC, _facts(fact_factory, C, [R0, R1]), failed=[R1])
    else:
        _write(tmp_path, "v1", C, DOC, _facts(fact_factory, C, [R0, R1]))
    with pytest.raises(sp.PopulationError):
        if breakage == "label":         # a v1 root labelled v2
            sp.build_population([sp.VersionSource("v2", tmp_path)], SECTIONS, [A, B, C])
        else:
            _build(tmp_path)


def test_population_file_is_hash_checked(tmp_path, two_versions):
    path = sp.write_population(_build(*two_versions), tmp_path / "pop.json")
    assert sp.load_population(path)["n_segments"] == 2
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["segments"][DOC].append(R1)                          # hand edit
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(sp.PopulationError, match="hash"):
        sp.load_population(path)


# --- applying it -------------------------------------------------------------

def test_a_failed_section_is_removed_for_every_annotator_in_both_versions(two_versions):
    v1, v2 = two_versions
    pop = _build(v1, v2)
    for root in (v1, v2):
        facts = run_phase2.discover_annotators(root, DOC, annotators=[A, B, C])
        kept, summary = sp.restrict_to_population(facts, DOC, pop, facts_root=root)
        for ann, fl in kept.items():
            assert {sp.fact_segment(f) for f in fl} == {R0}, (root.name, ann)
        assert summary["doc"]["excluded_sections"] == sorted([R1, ART1])
    # v1: A and B lose their R1 and ART1 triples, C only ART1 (R1 it never had).
    _, s1 = sp.restrict_to_population(
        run_phase2.discover_annotators(v1, DOC, annotators=[A, B, C]), DOC, pop,
        facts_root=v1)
    assert s1["version"] == "v1"
    assert s1["doc"]["facts_removed"] == {A: 2, B: 2, C: 1}


def test_a_changed_facts_file_is_refused(two_versions, fact_factory):
    v1, v2 = two_versions
    pop = _build(v1, v2)
    _write(v1, "v1", A, DOC, _facts(fact_factory, A, [R0]), style="error")   # re-run
    facts = run_phase2.discover_annotators(v1, DOC, annotators=[A, B, C])
    with pytest.raises(sp.PopulationError, match="changed"):
        sp.restrict_to_population(facts, DOC, pop, facts_root=v1)


def _phase2(root, out_root, *, population=None, suffix):
    return json.loads(run_phase2.run(
        DOC, facts_root=root, parsed_root=root, out_root=out_root,
        align_threshold=0.78, redundancy_cosine=0.95, merge_threshold=0.78,
        layer2_model="stub", skip_layer2=True, layer2_url="",
        out_suffix=suffix, annotators=[A, B, C],
        segment_population=population).read_text(encoding="utf-8"))


def test_filtering_happens_before_alignment_so_the_orphans_disappear(
        two_versions, tmp_path, monkeypatch):
    """C did not return R1 under v1. Aligned over whole documents, A's and B's
    R1 triples have no partner in the pairs with C and come out as orphans:
    orphans that exist only because a segment is missing. Restricted, R1 and
    ART1 are removed before alignment and the orphans never exist."""
    monkeypatch.delenv("NEO4J_AUTOSYNC", raising=False)
    v1, v2 = two_versions
    out = tmp_path / "conflicts"

    free = _phase2(v1, out, suffix="free")
    orphans = [g for g in free["groups"] if g["shape"] == "orphan"]
    assert orphans, "the missing segment must produce orphans without the rule"
    assert free["params"]["segment_population"] is None

    pop = _build(v1, v2)
    kept = _phase2(v1, out, population=pop, suffix="pop")
    assert [g for g in kept["groups"] if g["shape"] == "orphan"] == []
    assert kept["group_shapes"] == {"one_to_one": 6}      # R0, 3 pairs x 2 sides
    assert {sp.fact_segment(f) for fl in kept["facts_per_annotator"].values()
            for f in fl} == {R0}
    rec = kept["params"]["segment_population"]
    assert rec["hash"] == pop["hash"] and rec["version"] == "v1"
    assert rec["excluded_counts"]["v1"][C] == 1


def test_phase2_cli_requires_a_suffix_in_restricted_mode(two_versions, tmp_path, monkeypatch):
    v1, v2 = two_versions
    path = sp.write_population(_build(v1, v2), tmp_path / "pop.json")
    monkeypatch.setattr(sys, "argv", [
        "run_phase2", "--doc", DOC, "--facts-root", str(v1), "--skip-layer2",
        "--annotators", ",".join([A, B, C]), "--segment-population", str(path),
        "--out-root", str(tmp_path / "out")])
    with pytest.raises(SystemExit):
        run_phase2.main()
    assert not (tmp_path / "out").exists()


# --- comparing ---------------------------------------------------------------

def _conflicts(population=None, granularity=3):
    return {"params": {"segment_population": population},
            "group_label_counts": {"granularity": granularity}, "groups": [],
            "aligned_pairs": []}


def _summary(h, excluded=None):
    return {"hash": h, "version": "v1", "versions": ["v1", "v2"],
            "n_segments": 2, "n_segments_universe": 4,
            "excluded_counts": excluded or {"v1": {C: 1}, "v2": {A: 1}},
            "doc": {"n_segments": 4, "n_kept": 2, "excluded_sections": [R1, ART1],
                    "excluded": [], "facts_removed": {A: 2}}}


@pytest.mark.parametrize("pa,pb", [
    (_summary("aaa"), None),              # restricted vs all segments
    (None, _summary("aaa")),
    (_summary("aaa"), _summary("bbb")),   # two different populations
])
def test_distribution_shift_refuses_mismatched_populations(pa, pb):
    res = compare_conflict_files(_conflicts(pa), _conflicts(pb))
    assert res["available"] is False
    assert "population" in res["error"]
    assert res["note"] == res["error"]    # what the UI panels print
    assert "labels" not in res            # no distribution is produced


def test_same_population_compares_and_exposes_the_excluded_counts():
    res = compare_conflict_files(_conflicts(_summary("aaa")),
                                 _conflicts(_summary("aaa"), granularity=1))
    assert res["available"] is True
    pop = res["population"]
    assert pop["restricted"] and pop["match"]
    assert pop["excluded_counts"] == {"v1": {C: 1}, "v2": {A: 1}}
    assert pop["doc_excluded_sections"] == [R1, ART1]
    assert pop["n_segments"] == 2 and pop["n_segments_universe"] == 4


def test_two_unrestricted_files_are_the_sensitivity_check_not_refused():
    res = compare_conflict_files(_conflicts(), _conflicts())
    assert res["available"] is True
    assert res["population"]["restricted"] is False
    assert "sensitivity" in res["population"]["note"]


# --- recording failed sections going forward ------------------------------------

def test_run_extraction_records_which_sections_failed(tmp_path, monkeypatch):
    """`incomplete_sections` was a count; the paths are what the rule needs.
    Recorded in the facts file and the manifest record, and read back by the
    population builder as a failed section, not as a legitimately empty one."""
    root_dir = tmp_path.resolve()     # a short (8.3) temp path must not break relative_to
    monkeypatch.setattr(run_extraction, "ROOT", root_dir)
    monkeypatch.setattr(sp, "ROOT", root_dir)
    monkeypatch.setattr(run_extraction, "load_parsed_doc", lambda p: object())

    def fake_extract(*, incomplete_out, **kw):
        incomplete_out.append(f"{ART1} (unparsed)")
        incomplete_out.append(f"{R1} (network)")
        return []
    monkeypatch.setattr(run_extraction, "extract", fake_extract)

    root = root_dir / "facts"
    rec = run_extraction._run_one(C, {"doc_id": DOC, "parsed_path": "x.json"},
                                  out_root=root, base_url="", guideline_version="v1",
                                  retries=0)
    want = [{"section_path": ART1, "cause": "unparsed"},
            {"section_path": R1, "cause": "network"}]
    assert rec["status"] == "failed" and rec["incomplete_sections"] == 2
    assert rec["failed_sections"] == want
    payload = json.loads((root / sp.safe_dir(C) / f"{DOC}.json").read_text(encoding="utf-8"))
    assert payload["failed_sections"] == want
    assert payload["facts"] == []            # the fact records are untouched

    lost = sp.classify_returns(rec, payload, SECTIONS[DOC],
                               facts_file=root / sp.safe_dir(C) / f"{DOC}.json")
    assert lost == {ART1: ("failed_section", "unparsed"),
                    R1: ("failed_section", "network")}   # R0, ART2: empty, returned


def test_write_facts_omits_the_key_when_the_caller_does_not_know(tmp_path):
    from src.extractor import write_facts
    fp = write_facts([], model_name=A, doc_id=DOC, guideline_version="v1", out_root=tmp_path)
    assert "failed_sections" not in json.loads(fp.read_text(encoding="utf-8"))
