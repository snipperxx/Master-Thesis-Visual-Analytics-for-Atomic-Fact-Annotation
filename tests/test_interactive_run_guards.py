"""What the interface may run, and where it may write.

Invariant 1: the locked eval set is touched once per guideline version, by a
recorded batch run. Invariant 3: only g_v varies between runs, and the runs of
record are the batch runs of `scripts/run_extraction.py` / `run_phase2.py`.

Until 2026-09-23 the interface honoured neither outside one endpoint:
  * `/api/reextract_span` wrote into `data/facts/` whatever version was chosen,
    looked for the existing file under a directory name that does not exist for
    any annotator of record (and so replaced a whole document's facts with one
    clause's), and filed every probed clause as `preamble.recitals[0]`
    (BUG-2026-09-23-D);
  * Phase-1 / Phase-2 jobs, "Build KG" and Delete could overwrite the v1
    baseline's facts and conflicts (BUG-2026-09-23-E);
  * only the probe refused an eval document (BUG-2026-09-23-F).

Hermetic: Flask's test client over a tmp data root; the extractor and the job
queue are replaced, so nothing calls a model and no worker thread starts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QWEN = "cloud:qwen3.6:35b"
GEMMA = "cloud:hf.co/unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL"
DEV, EVAL = "train-000004", "train-000000"

PREAMBLE = "Whereas: (1) The first recital says one thing. (2) The second recital says another."
ENACTING = "Article 1 The refunds are fixed. Article 2 This Regulation enters into force."


def _parsed(doc_id: str) -> dict:
    r1 = PREAMBLE.index("(1)"), PREAMBLE.index("(2)") - 1
    r2 = PREAMBLE.index("(2)"), len(PREAMBLE)
    a1 = 0, ENACTING.index("Article 2") - 1
    a2 = ENACTING.index("Article 2"), len(ENACTING)
    return {"document": {
        "celex": doc_id, "title": f"act {doc_id}", "source": "eurlex",
        "citations": [], "annexes": [], "concluding_formulas": "",
        "preamble_text": PREAMBLE, "enacting_text": ENACTING,
        "recitals": [
            {"number": "(1)", "text": PREAMBLE[r1[0]:r1[1]], "char_start": r1[0], "char_end": r1[1]},
            {"number": "(2)", "text": PREAMBLE[r2[0]:r2[1]], "char_start": r2[0], "char_end": r2[1]},
        ],
        "articles": [
            {"number": "1", "title": None, "paragraphs": [ENACTING[a1[0]:a1[1]]],
             "char_start": a1[0], "char_end": a1[1]},
            {"number": "2", "title": None, "paragraphs": [ENACTING[a2[0]:a2[1]]],
             "char_start": a2[0], "char_end": a2[1]},
        ]}}


def _fact(fid: str, section: str, gid: str | None = None) -> dict:
    return {"fact_id": fid, "group_id": gid or fid, "annotator": QWEN,
            "subject": "s", "predicate": "p", "object": "o",
            "natural_language": f"fact {fid}",
            "source_locator": {"section_path": section, "char_start": 0, "char_end": 1}}


class _FakeFact:
    def __init__(self, d):
        self.d = d

    def model_dump(self, mode="json"):
        return dict(self.d)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    ui_app = pytest.importorskip("ui.app")
    from src import extractor
    from src import reextract_worker as rw

    data = tmp_path / "data"
    parsed = data / "parsed"
    parsed.mkdir(parents=True)
    for d in (DEV, EVAL):
        (parsed / f"{d}.json").write_text(json.dumps(_parsed(d)), encoding="utf-8")
    lists = data / "doc_lists"
    lists.mkdir()
    (lists / "eurlex_eval.json").write_text(json.dumps([EVAL]), encoding="utf-8")
    (lists / "phase1_dev_only.json").write_text(json.dumps([DEV]), encoding="utf-8")

    # the v1 baseline of record: a manifest-listed cell, and a conflicts file
    # computed from that batch root
    facts = data / "facts"
    base = facts / "cloud_qwen3.6_35b" / f"{DEV}.json"
    base.parent.mkdir(parents=True)
    base.write_text(json.dumps({
        "doc_id": DEV, "annotator": QWEN, "guideline_version": "v1", "fact_count": 2,
        "facts": [_fact("a1", "enacting.article_1"), _fact("r1", "preamble.recitals[0]")]}),
        encoding="utf-8")
    (facts / "manifest.json").write_text(json.dumps({"runs": {
        f"{QWEN}|{DEV}": {"status": "ok", "fact_count": 2},
        f"{GEMMA}|{DEV}": {"status": "ok", "fact_count": 0}}}), encoding="utf-8")
    conflicts = data / "conflicts"
    conflicts.mkdir()
    (conflicts / f"{DEV}.json").write_text(json.dumps(
        {"params": {"facts_root": str(facts), "layer2_model": "qwen3.5:4b"}}),
        encoding="utf-8")

    prompts = tmp_path / "prompts"
    prompts.mkdir()
    for v in ("v1", "v2", "v3"):
        (prompts / f"extract_{v}.md").write_text(f"R1 {v}\n", encoding="utf-8")
    (prompts / "extract_v2.lock.json").write_text('{"sha256": "x"}', encoding="utf-8")

    for name, val in (("DATA", data), ("PARSED", parsed), ("FACTS", facts),
                      ("CONFLICTS", conflicts), ("PROMPTS", prompts),
                      ("_SPLIT_CACHE", {})):
        monkeypatch.setattr(ui_app, name, val)
    monkeypatch.setattr(rw, "FACTS_ROOT", facts)
    monkeypatch.setattr(rw, "PROMPTS_DIR", prompts)

    calls = {"extract": [], "enqueue": []}

    def fake_extract(model_name, parsed_doc, **kw):
        calls["extract"].append((model_name, parsed_doc, kw))
        secs = list(extractor.enumerate_sections(parsed_doc))
        return [_FakeFact(_fact("new-" + s.section_path, s.section_path)) for s in secs]

    monkeypatch.setattr(extractor, "extract", fake_extract)

    def fake_enqueue(kind):
        def f(*args, **kw):
            calls["enqueue"].append((kind, args, kw))
            return rw.Job(job_id="test", kind=kind)
        return f

    monkeypatch.setattr(rw, "enqueue", fake_enqueue("reextract"))
    monkeypatch.setattr(rw, "enqueue_reextract_with_version", fake_enqueue("reextract"))
    monkeypatch.setattr(rw, "enqueue_phase2", fake_enqueue("phase2"))
    monkeypatch.setattr(rw, "enqueue_pipeline", fake_enqueue("pipeline"))

    ui_app.app.config["TESTING"] = True
    with ui_app.app.test_client() as c:
        yield {"client": c, "data": data, "base": base, "calls": calls,
               "conflicts": conflicts, "facts": facts}


def _msg(r) -> str:
    return (r.get_json() or {}).get("message", "")


# --------------------------------------------------------------------------
# 1. the probe writes into the selected version's root, never the baseline's
# --------------------------------------------------------------------------

def test_a_v2_probe_writes_into_the_v2_root_and_leaves_the_baseline_alone(env):
    before = env["base"].read_bytes()
    r = env["client"].post("/api/reextract_span", json={
        "doc_id": DEV, "model": QWEN, "guideline_version": "v2",
        "section_path": "enacting.article_1"})
    assert r.status_code == 200, _msg(r)
    out = env["data"] / "facts__v2" / "cloud_qwen3.6_35b" / f"{DEV}.json"
    assert out.exists()
    written = json.loads(out.read_text(encoding="utf-8"))
    assert [f["source_locator"]["section_path"] for f in written["facts"]] \
        == ["enacting.article_1"]
    assert written["probes"][0]["section_path"] == "enacting.article_1"
    assert r.get_json()["facts_root"].endswith("facts__v2")
    assert env["base"].read_bytes() == before, "the v1 baseline was modified"


def test_a_v1_probe_of_a_cell_of_the_run_of_record_is_refused_before_the_model_runs(env):
    before = env["base"].read_bytes()
    r = env["client"].post("/api/reextract_span", json={
        "doc_id": DEV, "model": QWEN, "guideline_version": "v1",
        "section_path": "enacting.article_1"})
    assert r.status_code == 409
    assert "run of record" in _msg(r) and "overwrite_record" in _msg(r)
    assert env["calls"]["extract"] == []
    assert env["base"].read_bytes() == before


def test_an_explicit_overwrite_replaces_only_the_probed_clause(env):
    """`overwrite_record: true` is the deliberate route; even then the file keeps
    every other clause and says a probe touched it."""
    r = env["client"].post("/api/reextract_span", json={
        "doc_id": DEV, "model": QWEN, "guideline_version": "v1",
        "section_path": "enacting.article_1", "overwrite_record": True})
    assert r.status_code == 200, _msg(r)
    facts = json.loads(env["base"].read_text(encoding="utf-8"))
    ids = [f["fact_id"] for f in facts["facts"]]
    assert "a1" not in ids and "r1" in ids and "new-enacting.article_1" in ids
    assert facts["fact_count"] == 2
    assert facts["probes"][0]["n_replaced"] == 1


def test_the_probe_merges_into_the_existing_file_of_a_dotted_model_name(env):
    """The old path helper turned `cloud_qwen3.6_35b` into `cloud_qwen3_6_35b`,
    found nothing, and wrote a fresh file: the document's other clauses were
    gone. Every annotator of record has a dot in its name."""
    draft = env["data"] / "facts__v3" / "cloud_qwen3.6_35b" / f"{DEV}.json"
    draft.parent.mkdir(parents=True)
    draft.write_text(json.dumps({"doc_id": DEV, "annotator": QWEN,
                                 "guideline_version": "v3", "fact_count": 1,
                                 "facts": [_fact("keep", "enacting.article_2")]}),
                     encoding="utf-8")
    r = env["client"].post("/api/reextract_span", json={
        "doc_id": DEV, "model": QWEN, "guideline_version": "v3",
        "section_path": "enacting.article_1"})
    assert r.status_code == 200, _msg(r)
    ids = [f["fact_id"] for f in json.loads(draft.read_text(encoding="utf-8"))["facts"]]
    assert ids == ["keep", "new-enacting.article_1"]
    assert not (env["data"] / "facts__v3" / "cloud_qwen3_6_35b").exists()


@pytest.mark.parametrize("section, stream", [
    ("preamble.recitals[1]", "preamble"), ("enacting.article_2", "enacting")])
def test_the_probed_clause_is_extracted_at_its_own_path_and_text(env, section, stream):
    """The mini-document fed an article's offsets into the preamble and called
    every clause `preamble.recitals[0]`, which also gave it Recital 1's ids."""
    from src.extractor import enumerate_sections
    r = env["client"].post("/api/reextract_span", json={
        "doc_id": DEV, "model": QWEN, "guideline_version": "v3", "section_path": section})
    assert r.status_code == 200, _msg(r)
    doc = env["calls"]["extract"][-1][1]
    secs = list(enumerate_sections(doc))
    assert [s.section_path for s in secs] == [section]
    text = PREAMBLE if stream == "preamble" else ENACTING
    full = {s.section_path: s for s in enumerate_sections(
        type(doc)(**_parsed(DEV)["document"]))}
    assert secs[0].text == full[section].text and secs[0].text in text
    assert secs[0].base_offset == full[section].base_offset


def test_facts_file_is_where_write_facts_writes(tmp_path):
    from src.extractor import write_facts
    from src.record_guard import facts_file
    for model in (QWEN, GEMMA):
        assert write_facts([], model_name=model, doc_id=DEV, guideline_version="v1",
                           out_root=tmp_path) == facts_file(tmp_path, model, DEV)


# --------------------------------------------------------------------------
# 2. interactive runs do not touch the locked eval set
# --------------------------------------------------------------------------

@pytest.mark.parametrize("url, body", [
    ("/api/reextract_span", {"doc_id": EVAL, "model": QWEN, "guideline_version": "v3",
                             "section_path": "enacting.article_1"}),
    ("/api/run_phase1", {"doc_ids": [EVAL], "guideline_version": "v3"}),
    ("/api/run_phase2", {"doc_ids": [EVAL], "version": "v3"}),
    ("/api/run_matrix", {"cells": [{"doc_id": DEV, "model": QWEN, "guideline_version": "v3"},
                                   {"doc_id": EVAL, "model": QWEN, "guideline_version": "v3"}]}),
    ("/api/run_pipeline", {"doc_ids": [EVAL], "guideline_version": "v3"}),
    ("/api/guideline", {"guideline_text": "R1 x", "doc_paths": [f"data/parsed/{EVAL}.json"]}),
])
def test_every_interactive_run_refuses_an_eval_document(env, url, body):
    r = env["client"].post(url, json=body)
    assert r.status_code == 403, (url, r.status_code, _msg(r))
    assert "eurlex_eval.json" in _msg(r)
    assert env["calls"]["enqueue"] == [] and env["calls"]["extract"] == []


def test_overwrite_record_does_not_open_the_eval_set(env):
    r = env["client"].post("/api/run_phase1", json={
        "doc_ids": [EVAL], "guideline_version": "v3", "overwrite_record": True})
    assert r.status_code == 403


def test_a_default_document_set_leaves_the_eval_documents_out_and_says_so(env):
    r = env["client"].post("/api/run_phase1", json={"guideline_version": "v3",
                                                    "models": [QWEN]})
    assert r.status_code == 202, _msg(r)
    assert r.get_json()["excluded_eval"] == [EVAL]
    (_, _, kw), = env["calls"]["enqueue"]
    assert [Path(p).stem for p in kw["doc_paths"]] == [DEV]


def test_the_batch_run_of_record_can_still_reach_the_eval_set(tmp_path, monkeypatch):
    """The refusal is the interface's. `run_phase2.py --include-eval` is how
    the eval pass is run, and without the flag the script refuses on its own."""
    from scripts import run_phase2
    (tmp_path / "data" / "doc_lists").mkdir(parents=True)
    (tmp_path / "data" / "doc_lists" / "eurlex_eval.json").write_text(
        json.dumps([EVAL]), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    ran = []

    def fake_run(doc_id, **kw):
        ran.append(doc_id)
        out = tmp_path / f"{doc_id}.json"
        out.write_text('{"label_counts": {}}', encoding="utf-8")
        return out

    monkeypatch.setattr(run_phase2, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_phase2", "--doc", EVAL, "--include-eval",
                                      "--skip-layer2"])
    run_phase2.main()
    assert ran == [EVAL]

    monkeypatch.setattr(sys, "argv", ["run_phase2", "--doc", EVAL, "--skip-layer2"])
    with pytest.raises(SystemExit):
        run_phase2.main()
    assert ran == [EVAL]


# --------------------------------------------------------------------------
# interactive jobs do not overwrite a batch run of record (BUG-2026-09-23-E)
# --------------------------------------------------------------------------

def test_phase1_from_the_interface_refuses_the_v1_run_of_record_but_not_a_draft(env):
    c = env["client"]
    r = c.post("/api/run_phase1", json={"doc_ids": [DEV], "models": [QWEN],
                                        "guideline_version": "v1"})
    assert r.status_code == 409 and "cloud_qwen3.6_35b" in _msg(r)
    # a cell the manifest lists is of record even when its file is empty or
    # missing: the batch will retry it, and Phase-2 would read ours meanwhile
    r = c.post("/api/run_phase1", json={"doc_ids": [DEV], "models": [GEMMA],
                                        "guideline_version": "v1"})
    assert r.status_code == 409
    assert env["calls"]["enqueue"] == []
    r = c.post("/api/run_phase1", json={"doc_ids": [DEV], "models": [QWEN],
                                        "guideline_version": "v3"})
    assert r.status_code == 202


def test_build_kg_cannot_overwrite_the_baseline_conflicts(env):
    """"Build KG" posts exactly this: one document, the version on screen."""
    before = (env["conflicts"] / f"{DEV}.json").read_bytes()
    r = env["client"].post("/api/run_phase2", json={"doc_ids": [DEV], "version": "v1",
                                                    "skip_layer2": True})
    assert r.status_code == 409 and f"{DEV}.json" in _msg(r)
    assert env["calls"]["enqueue"] == []
    assert (env["conflicts"] / f"{DEV}.json").read_bytes() == before


def test_a_frozen_arms_conflicts_are_not_produced_from_the_interface(env):
    r = env["client"].post("/api/run_phase2", json={"doc_ids": [DEV], "version": "v2"})
    assert r.status_code == 409 and "frozen" in _msg(r)
    r = env["client"].post("/api/run_pipeline", json={"doc_ids": [DEV],
                                                      "guideline_version": "v2"})
    assert r.status_code == 409


def test_iterating_on_a_draft_version_stays_free(env):
    """A draft's conflicts were computed from the interface's own root, which
    has no manifest: re-running them is the point of the interface."""
    (env["conflicts"] / f"{DEV}__v3.json").write_text(json.dumps(
        {"params": {"facts_root": str(env["data"] / "facts__v3")}}), encoding="utf-8")
    r = env["client"].post("/api/run_phase2", json={"doc_ids": [DEV], "version": "v3"})
    assert r.status_code == 202, _msg(r)
    (kind, _, kw), = env["calls"]["enqueue"]
    assert kind == "phase2" and kw["params"]["out_suffix"] == "v3"


def test_a_corpus_document_cannot_be_deleted_from_the_interface(env):
    r = env["client"].delete(f"/api/doc/{DEV}")
    assert r.status_code == 403
    assert env["base"].exists() and (env["conflicts"] / f"{DEV}.json").exists()


def test_what_counts_as_of_record_is_read_off_the_data(tmp_path):
    from src import record_guard as rg
    root = tmp_path / "data" / "facts"
    root.mkdir(parents=True)
    assert not rg.facts_cell_of_record(root, QWEN, DEV)          # no manifest
    (root / "manifest.json").write_text(json.dumps({"runs": {f"{QWEN}|{DEV}": {}}}),
                                        encoding="utf-8")
    assert rg.facts_cell_of_record(root, QWEN, DEV)
    assert not rg.facts_cell_of_record(root, QWEN, "user-000001")
    # an unreadable manifest protects every file that exists
    (root / "manifest.json").write_text("{not json", encoding="utf-8")
    f = rg.facts_file(root, QWEN, DEV)
    f.parent.mkdir(parents=True)
    f.write_text("{}", encoding="utf-8")
    assert rg.facts_cell_of_record(root, QWEN, DEV)
    # run_phase2 records the facts root as typed on this machine: `data\facts`
    confl = tmp_path / "c.json"
    confl.write_text(json.dumps({"params": {"facts_root": "data\\facts"}}), encoding="utf-8")
    assert rg.conflicts_file_of_record(confl, tmp_path)
    confl.write_text(json.dumps({"params": {"facts_root": "data\\facts__v3"}}),
                     encoding="utf-8")
    assert not rg.conflicts_file_of_record(confl, tmp_path)
    confl.write_text(json.dumps({"params": {}}), encoding="utf-8")
    assert rg.conflicts_file_of_record(confl, tmp_path)          # provenance unknown


# --------------------------------------------------------------------------
# the corpus-level shift refuses mixed segment populations
# --------------------------------------------------------------------------

def _confl(pop=None):
    params = {"facts_root": "x"}
    if pop is not None:
        params["segment_population"] = {"hash": pop}
    return json.dumps({"params": params, "annotators": [QWEN, GEMMA],
                       "label_counts": {"granularity": 2}, "aligned_pairs": []})


@pytest.mark.parametrize("v1_pop, v2_pop, refused", [
    (None, None, False), ("h1", "h1", False), (None, "h1", True), ("h1", "h2", True)])
def test_the_aggregate_shift_is_over_one_segment_population(env, v1_pop, v2_pop, refused):
    confl = env["conflicts"]
    (confl / f"{DEV}.json").write_text(_confl(v1_pop), encoding="utf-8")
    (confl / f"{DEV}__v3.json").write_text(_confl(v2_pop), encoding="utf-8")
    d = env["client"].get("/api/distribution_shift_agg?v2=v3&unit=pair").get_json()
    if refused:
        assert d["available"] is False and "segment populations differ" in d["note"]
        assert d["population_mismatch"] == [DEV]
    else:
        assert d["docs"] == [DEV] and d["population"]["restricted"] == (v1_pop is not None)
