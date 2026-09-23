"""
Flask backend for the Visual Analytics prototype.

All endpoints:
    GET  /                              single-page UI
    GET  /api/docs                      docs on disk (with phase-2 flag)
    GET  /api/doc/<id>                  parsed text + section offsets
    GET  /api/facts/<id>                facts with filters
    GET  /api/pairs/<id>                aligned-pair table
    GET  /api/graph/<id>                Cytoscape elements; ?merge_threshold= recomputes
    GET  /api/coverage/<id>             per-annotator coverage metrics
    GET  /api/distribution_shift/<id>   v1 vs v2 conflict-label diff
    POST /api/verify/<doc>/<fact_id>    persist verified/rejected
    POST /api/upload_text               ingest user-pasted text -> ParsedDocument
    GET  /api/guidelines                list available prompts/extract_*.md
    GET  /api/guidelines/<version>      get one
    PUT  /api/guidelines/<version>      save (creates new version file)
    POST /api/reextract_span            re-extract a section/span with chosen model+guideline
    GET/POST /api/guideline_log/<v>     the change log: why each edit, citing what
    GET/POST /api/guideline_history/<v> checkpoints of a version, and their diffs
    GET/POST /api/guideline_freeze/<v>  freeze a version (lock + change-log digest)
    POST /api/guideline_shot/<v>        attach a screenshot (multipart or data URL)
    GET  /api/guideline_shot/<v>/<name> serve one stored screenshot

    Background jobs:
    POST /api/guideline                 enqueue Phase-4 re-extraction job (new guideline body)
    POST /api/run_phase1                enqueue Phase-1 batch (existing guideline_version)
    POST /api/run_phase2                enqueue Phase-2 batch (alignment + conflict detect)
    POST /api/run_matrix                multi-cell scheduling
    GET  /api/run_matrix/status         existing fact counts per (doc, model)
    GET  /api/jobs[/<job_id>]           list + poll jobs (all kinds)
    POST /api/jobs/<job_id>/cancel      cancel a queued job
"""

from __future__ import annotations
import os

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, render_template, request

# Importing providers also loads .env (zero-dep) so OLLAMA_CLOUD_API_KEY /
# OLLAMA_CLOUD_URL are available to every endpoint and worker thread.
import sys as _sys
_REPO_ROOT_FOR_SRC = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT_FOR_SRC) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT_FOR_SRC))
from src import providers  # noqa: E402
from src import record_guard  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"
PARSED = DATA / "parsed"
FACTS = DATA / "facts"
CONFLICTS = DATA / "conflicts"
GRAPHS = DATA / "graphs"
VERIFICATIONS = DATA / "verifications"
PROMPTS = REPO_ROOT / "prompts"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ui")

app = Flask(__name__,
            template_folder=str(REPO_ROOT / "ui" / "templates"),
            static_folder=str(REPO_ROOT / "ui" / "static"))
# Dev ergonomics on this machine: the Jinja template cache missed on-disk
# changes (mount mtime semantics + non-debug runs), serving stale HTML while
# static JS was already fresh. Force per-request template mtime checks and
# disable static-file caching regardless of --debug.
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.jinja_env.auto_reload = True


@app.errorhandler(Exception)
def _json_errors_for_api(exc):
    """Return JSON (not Flask's HTML page) for any /api/* path failure.

    Prevents the front-end's `response.json()` from blowing up with
    'Unexpected token <' when the backend hits an abort() or unhandled
    exception. Non-/api paths fall back to Flask's default handler.
    """
    from werkzeug.exceptions import HTTPException
    if not request.path.startswith("/api/"):
        # Let Flask render its normal HTML error page for non-API routes.
        raise exc
    if isinstance(exc, HTTPException):
        return jsonify({"error": exc.name, "status": exc.code,
                        "message": exc.description or str(exc)}), exc.code
    log.exception("Unhandled /api exception on %s", request.path)
    return jsonify({"error": type(exc).__name__, "status": 500,
                    "message": str(exc)}), 500


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _doc_ids_available() -> list[str]:
    return sorted(p.stem for p in PARSED.glob("*.json")) if PARSED.exists() else []


def _docs_with_conflicts(variant: str = "") -> set[str]:
    """Docs that have a Phase-2 result **for this guideline version**.

    The baseline lives at `<doc>.json`, every other version at
    `<doc>__<version>.json`. This used to hard-code the baseline, so selecting
    a version in the top bar left the document list showing v1's availability
    while every panel below showed the selected version — the list said a
    document had conflicts when that version had never been run on it.
    """
    if not CONFLICTS.exists():
        return set()
    if variant:
        suffix = f"__{variant}"
        return {p.stem[: -len(suffix)] for p in CONFLICTS.glob(f"*{suffix}.json")
                if p.stem.endswith(suffix)}
    return {p.stem for p in CONFLICTS.glob("*.json") if "__" not in p.stem}


def _guideline_versions() -> list[dict]:
    """Every guideline version that has a Phase-2 result on disk, newest last.

    "v1" is always present and always means the unsuffixed baseline files, so
    that the dropdown has a stable home entry even before any v2 run exists.
    """
    out = [{"value": "", "label": "v1 (baseline)",
            "n_docs": len(_docs_with_conflicts())}]
    if not CONFLICTS.exists():
        return out
    seen: set[str] = set()
    for fp in CONFLICTS.glob("*__*.json"):
        v = fp.stem.split("__", 1)[1]
        if v and _GUIDELINE_VERSION_RE.match(v):
            seen.add(v)
    for v in sorted(seen):
        out.append({"value": v, "label": v,
                    "n_docs": len(_docs_with_conflicts(v))})
    return out


def _variant_arg() -> str:
    """?variant=<guideline_version> selects data/conflicts/<doc>__<v>.json.

    Empty / "v1" / "base" mean the baseline file. Validation mirrors
    _GUIDELINE_VERSION_RE (defined below; resolved at request time)."""
    v = (request.args.get("variant") or "").strip()
    if v in ("", "v1", "base"):
        return ""
    if not _GUIDELINE_VERSION_RE.match(v):
        abort(400, "invalid variant")
    return v


def _conflicts_path_for(doc_id: str, variant: str) -> Path:
    return CONFLICTS / (f"{doc_id}__{variant}.json" if variant else f"{doc_id}.json")


def _conflicts_path(doc_id: str) -> Path:
    return _conflicts_path_for(doc_id, _variant_arg())


def _graph_cache_path(doc_id: str) -> Path:
    v = _variant_arg()
    return GRAPHS / (f"{doc_id}__{v}.json" if v else f"{doc_id}.json")


def _doc_variants(doc_id: str) -> list[str]:
    if not CONFLICTS.exists():
        return []
    return sorted(p.stem.split("__", 1)[1] for p in CONFLICTS.glob(f"{doc_id}__*.json"))


def _of_record() -> dict:
    """The frozen annotator set and the Layer-2 arbitrator of record.

    Read from `scripts/run_phase2.py`, which owns both, so the interface's
    defaults cannot drift from the batch runs they are meant to reproduce. They
    used to be spelled out in the page and the JS as the retired local 4B
    models, with the stub arbitrator ticked (BUG-2026-09-23-G).
    """
    from scripts.run_phase2 import ANNOTATORS_OF_RECORD, LAYER2_OF_RECORD
    return {"annotators": list(ANNOTATORS_OF_RECORD), "layer2": LAYER2_OF_RECORD}


@app.route("/")
def index():
    return render_template("index.html", record=_of_record())


@app.route("/api/guideline_versions")
def api_guideline_versions():
    """Every guideline version with a Phase-2 result, for the top-bar selector.

    Corpus-wide, not per document: the selector is a mode the whole session is
    in, so it must not be rebuilt (or hidden) each time the document changes.
    """
    return jsonify({"versions": _guideline_versions()})


# ---- which split a document belongs to -------------------------------------
#
# The interface lists every parsed document, and the eval set is 12 of the 50.
# Those are empty by construction (invariant 1: the locked set is touched once
# per guideline version, to record a result), so opening on one makes the app
# look broken — and worse, it puts an "Extract here" button in front of a
# document that must not be extracted. The split travels with the doc list so
# the UI can say which it is and refuse to open on one.

_SPLIT_CACHE: dict[str, str] = {}


def _doc_split(doc_id: str) -> str:
    if not _SPLIT_CACHE:
        for name, split in (("eurlex_eval.json", "eval"),
                            ("phase1_dev_only.json", "dev")):
            path = DATA / "doc_lists" / name
            if not path.exists():
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            items = raw if isinstance(raw, list) else next(
                (raw[k] for k in ("doc_ids", "docs", "ids") if k in raw), [])
            for it in items:
                key = it if isinstance(it, str) else (it or {}).get("doc_id")
                if key:
                    _SPLIT_CACHE.setdefault(key, split)
    return _SPLIT_CACHE.get(doc_id, "")


# ---- what an interactive run may touch ------------------------------------
#
# Two refusals shared by every path that starts extraction or conflict
# detection from the interface: the probe, the job endpoints, the run matrix,
# "Run on current doc", "Build KG" and the pipeline.
#
# 1. The locked eval set (invariant 1). It used to be refused by the per-segment
#    probe only, so "Run on current doc" with an eval document open ran Phase-1
#    on it (BUG-2026-09-23-F). A document the caller NAMES is refused outright;
#    a default set the caller did not name ("all parsed documents") has the eval
#    documents dropped and says which, as `run_phase2.py` does without
#    --include-eval. There is no override: the eval pass of record is a batch
#    run of the scripts, which this does not touch.
# 2. Runs of record (invariant 3; `src/record_guard.py`). A write that would
#    replace a file of a batch run of record is refused with 409 unless the
#    request says `overwrite_record: true` — a deliberate API call, which the
#    interface never sends by itself.

def _eval_docs(doc_ids) -> list[str]:
    return sorted({d for d in doc_ids if _doc_split(d) == "eval"})


def _refuse_eval(doc_ids, what: str) -> None:
    locked = _eval_docs(doc_ids)
    if locked:
        abort(403, f"{what} refused: {', '.join(locked[:5])}"
                   f"{' …' if len(locked) > 5 else ''} "
                   f"{'is' if len(locked) == 1 else 'are'} in the locked eval set "
                   f"(data/doc_lists/eurlex_eval.json, frozen 2026-07-31). It is "
                   f"touched once per guideline version, by a recorded batch run "
                   f"of scripts/run_extraction.py and scripts/run_phase2.py "
                   f"--include-eval, never from the interface.")


def _without_eval(doc_ids) -> tuple[list[str], list[str]]:
    """A default document set with the eval documents taken out, and which."""
    locked = set(_eval_docs(doc_ids))
    return [d for d in doc_ids if d not in locked], sorted(locked)


def _rel(path) -> str:
    try:
        return Path(path).resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _refuse_record_overwrite(payload: dict, what: str, *, facts_cells=(),
                             conflicts=()) -> None:
    """409 when the run would replace a file of a batch run of record.

    `facts_cells` are (root, model, doc_id); `conflicts` are conflict-file paths.
    """
    hits: list[str] = []
    for root, model, doc_id in dict.fromkeys(facts_cells):
        if record_guard.facts_cell_of_record(root, model, doc_id):
            hits.append(_rel(record_guard.facts_file(root, model, doc_id)))
    for path in dict.fromkeys(conflicts):
        if record_guard.conflicts_file_of_record(path, REPO_ROOT):
            hits.append(_rel(path))
    if not hits:
        return
    if payload.get("overwrite_record") is True:
        log.warning("%s: overwriting %d file(s) of a run of record on explicit "
                    "request: %s", what, len(hits), ", ".join(hits))
        return
    abort(409, f"{what} refused: it would overwrite {len(hits)} file(s) of a "
               f"batch run of record ({', '.join(hits[:3])}"
               f"{' …' if len(hits) > 3 else ''}). Runs of record are written by "
               f"scripts/run_extraction.py and scripts/run_phase2.py only "
               f"(invariant 3). To try a rule, save it as a draft version: its "
               f"runs go to data/facts__<version>/ and "
               f"data/conflicts/<doc>__<version>.json. If overwriting is really "
               f"intended, resend with overwrite_record: true and log why.")


def _refuse_frozen_phase2(version: str, payload: dict, what: str) -> None:
    """A committed arm's conflicts are the batch run's, not the interface's.

    From here a frozen version's Phase-2 would read `data/facts__<v>/` and write
    `data/conflicts/<doc>__<v>.json` — the file name its batch run of record
    uses, filled from other facts (the v2 dev pass writes `data/facts_v2/`).
    """
    if version in ("", "v1") or _guideline_lock(version) is None:
        return
    if payload.get("overwrite_record") is True:
        log.warning("%s: Phase-2 for frozen %s on explicit request", what, version)
        return
    abort(409, f"{what} refused: extract_{version}.md is frozen, and its "
               f"conflicts of record come from a batch run of "
               f"scripts/run_phase2.py over the arm's own facts root "
               f"(--facts-root <root> --out-suffix {version}). From the interface "
               f"it would write data/conflicts/<doc>__{version}.json from "
               f"data/facts__{version}/ — the same file name, other facts.")


@app.route("/api/docs")
def api_docs():
    out = []
    variant = _variant_arg()
    with_confl = _docs_with_conflicts(variant)
    for doc_id in _doc_ids_available():
        ppath = PARSED / f"{doc_id}.json"
        try:
            parsed = _load_json(ppath)["document"]
        except Exception:
            continue
        source = parsed.get("source") or (      # parsed JSON is authoritative;
            "webnlg" if doc_id.startswith("webnlg-")   # prefixes only for legacy
            else "user" if doc_id.startswith("user-")
            else "eurlex")
        entry = {
            "doc_id": doc_id, "title": parsed.get("title", doc_id),
            "has_conflicts": doc_id in with_confl,
            "is_user_doc": doc_id.startswith("user-"),
            "source": source,
            "split": _doc_split(doc_id),
        }
        entry["variants"] = _doc_variants(doc_id)
        if entry["has_conflicts"]:
            try:
                confl = _load_json(_conflicts_path_for(doc_id, variant))
                entry["annotators"] = confl.get("annotators", [])
                entry["label_counts"] = confl.get("label_counts", {})
                # Surfaced so the UI can WARN when the conflict labels came
                # from the deterministic stub instead of a model. Without it a
                # stub run is indistinguishable from an arbitrated one on
                # screen, and the whole point of the interface is to diagnose
                # on real labels.
                entry["layer2_model"] = (confl.get("params") or {}).get("layer2_model")
            except Exception:
                pass
        out.append(entry)
    return jsonify({"docs": out})


@app.route("/api/doc/<doc_id>", methods=["GET", "DELETE"])
def api_doc(doc_id: str):
    path = PARSED / f"{doc_id}.json"
    if request.method == "DELETE":
        # Delete exists for uploaded text. On a corpus document it removed the
        # run of record's facts, the baseline conflicts, graphs and the human
        # reviews in one click (BUG-2026-09-23-E).
        split = _doc_split(doc_id)
        if split:
            abort(403, f"{doc_id} is a corpus document ({split} split, "
                       f"data/doc_lists/): its facts and conflicts are runs of "
                       f"record and are not deleted from the interface")
        # Remove every per-doc artefact across all data subdirs.
        removed: list[str] = []
        # parsed
        if path.exists():
            path.unlink(); removed.append(str(path.relative_to(REPO_ROOT)))
        # facts (one file per annotator)
        if FACTS.exists():
            for sub in FACTS.iterdir():
                if not sub.is_dir():
                    continue
                fp = sub / f"{doc_id}.json"
                if fp.exists():
                    fp.unlink(); removed.append(str(fp.relative_to(REPO_ROOT)))
        # conflicts (main + __v2_* variants)
        if CONFLICTS.exists():
            for fp in [CONFLICTS / f"{doc_id}.json", *CONFLICTS.glob(f"{doc_id}__*.json")]:
                if fp.exists():
                    fp.unlink(); removed.append(str(fp.relative_to(REPO_ROOT)))
        # graphs (incl. variants) + verifications + reviews + versioned facts roots
        for root in (GRAPHS, VERIFICATIONS, DATA / "reviews"):
            for fp in [root / f"{doc_id}.json", *root.glob(f"{doc_id}__*.json")] if root.exists() else []:
                if fp.exists():
                    fp.unlink(); removed.append(str(fp.relative_to(REPO_ROOT)))
        for froot in DATA.glob("facts__*"):
            for sub in froot.iterdir():
                if sub.is_dir():
                    fp = sub / f"{doc_id}.json"
                    if fp.exists():
                        fp.unlink(); removed.append(str(fp.relative_to(REPO_ROOT)))
        if not removed:
            abort(404, f"doc {doc_id!r} has no files on disk")
        return jsonify({"ok": True, "doc_id": doc_id,
                        "removed": removed, "n_removed": len(removed)})
    # GET
    if not path.exists():
        abort(404, f"parsed doc {doc_id!r} not found")
    parsed = _load_json(path)["document"]
    return jsonify({
        "doc_id": doc_id, "title": parsed.get("title"),
        "citations": parsed.get("citations", []),
        "preamble_text": parsed.get("preamble_text", ""),
        "enacting_text": parsed.get("enacting_text", ""),
        "recitals": parsed.get("recitals", []),
        "articles": parsed.get("articles", []),
        "concluding_formulas": parsed.get("concluding_formulas", ""),
        # WebNLG extensions (empty for EUR-Lex / user docs)
        "source": parsed.get(
            "source",
            "webnlg" if doc_id.startswith("webnlg-")
            else "user" if doc_id.startswith("user-") else "eurlex",
        ),
        "gold_triples": parsed.get("gold_triples", []),
        "webnlg_meta": parsed.get("webnlg_meta", {}),
        "redocred_meta": parsed.get("redocred_meta", {}),
    })


@app.route("/api/facts/<doc_id>")
def api_facts(doc_id: str):
    path = _conflicts_path(doc_id)
    if not path.exists():
        return _facts_from_disk_only(doc_id)
    confl = _load_json(path)
    annotators = request.args.getlist("annotator")
    conflict = request.args.get("conflict")

    # Always resolve per-fact edge labels (not only when filtering) so the
    # top-bar label counts and the facts-table Conflict column are truthful.
    edge_label_by_fact: dict[str, str] = {}
    if True:
        gp = _graph_cache_path(doc_id)
        if gp.exists():
            for edge in _load_json(gp).get("edges", []):
                label = edge["data"]["conflict_label"]
                for fid in edge["data"]["fact_ids"]:
                    cur = edge_label_by_fact.get(fid)
                    if cur is None or label > cur:
                        edge_label_by_fact[fid] = label

    vpath = VERIFICATIONS / f"{doc_id}.json"
    verifications = _load_json(vpath) if vpath.exists() else {}

    out: list[dict] = []
    for annot, facts in confl["facts_per_annotator"].items():
        if annotators and annot not in annotators:
            continue
        for f in facts:
            f2 = dict(f)
            f2["_annotator"] = annot
            f2["_edge_conflict_label"] = edge_label_by_fact.get(f["fact_id"], "unlabeled")
            f2["_verification"] = verifications.get(f["fact_id"])
            if conflict and f2["_edge_conflict_label"] != conflict:
                continue
            out.append(f2)

    # Merge in annotators present on disk but NOT in this Phase-2 conflicts file
    # (e.g. a model extracted after the last run, like cloud:gemma4:latest).
    # They show as unlabeled until Phase-2 re-runs, so newly-added models —
    # cloud especially — are visible in all three views immediately.
    from scripts.run_phase2 import discover_annotators
    from src import reextract_worker as rw
    disk_fpa = discover_annotators(rw.facts_root_for_version(_variant_arg() or "v1"), doc_id)
    for annot, facts in disk_fpa.items():
        if annot in confl["facts_per_annotator"]:
            continue
        if annotators and annot not in annotators:
            continue
        for f in facts:
            lbl = edge_label_by_fact.get(f["fact_id"], "unlabeled")
            if conflict and lbl != conflict:
                continue
            f2 = dict(f)
            f2["_annotator"] = annot
            f2["_edge_conflict_label"] = lbl
            f2["_verification"] = verifications.get(f["fact_id"])
            out.append(f2)
    return jsonify({"doc_id": doc_id, "facts": out, "count": len(out)})


def _facts_from_disk_only(doc_id: str):
    if not PARSED.joinpath(f"{doc_id}.json").exists():
        abort(404, f"doc {doc_id!r} not found")
    # The selected version's root, not always v1's: with no conflicts file yet
    # for a draft version, this used to show the v1 facts under the draft's
    # name — and hide what a probe had just written for it.
    from src import reextract_worker as rw
    root = rw.facts_root_for_version(_variant_arg() or "v1")
    out: list[dict] = []
    for sub in sorted(root.iterdir()) if root.exists() else []:
        if not sub.is_dir():
            continue
        fp = sub / f"{doc_id}.json"
        if not fp.exists():
            continue
        try:
            doc = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        for f in doc.get("facts", []):
            f2 = dict(f)
            f2["_annotator"] = doc.get("annotator") or sub.name
            f2["_edge_conflict_label"] = "unlabeled"
            f2["_verification"] = None
            out.append(f2)
    return jsonify({"doc_id": doc_id, "facts": out, "count": len(out)})


@app.route("/api/pairs/<doc_id>")
def api_pairs(doc_id: str):
    path = _conflicts_path(doc_id)
    return jsonify(_load_json(path)["aligned_pairs"]) if path.exists() else jsonify([])


@app.route("/api/groups/<doc_id>")
def api_groups(doc_id: str):
    """The GROUP layer — one anchor fact plus the partners it matched.

    Groups, not pairs, are the unit of analysis (method.tex), and a one-to-many
    group is the only place a granularity split is representable at all. The
    pair endpoint above cannot show it: a pair is 1:1 by construction.

    Fact ids are resolved to the facts themselves so the client can render the
    "1 vs N" comparison without a second round trip. `?shape=` and `?label=`
    filter; both are optional.
    """
    path = _conflicts_path(doc_id)
    if not path.exists():
        return jsonify({"doc_id": doc_id, "groups": [], "shapes": {},
                        "label_counts": {}, "available": False})
    confl = _load_json(path)
    groups = confl.get("groups") or []

    by_id: dict[str, dict] = {}
    for annot, facts in (confl.get("facts_per_annotator") or {}).items():
        for f in facts:
            f2 = dict(f)
            f2["_annotator"] = annot
            by_id[f.get("fact_id")] = f2

    def slim(fid):
        f = by_id.get(fid)
        if not f:
            return {"fact_id": fid, "missing": True}
        return {
            "fact_id": fid,
            "annotator": f.get("_annotator"),
            "subject": f.get("subject"), "predicate": f.get("predicate"),
            "object": f.get("object"),
            "natural_language": f.get("natural_language"),
            "group_id": f.get("group_id"),
            "section_path": (f.get("source_locator") or {}).get("section_path"),
            "quote": (f.get("source_locator") or {}).get("quote"),
        }

    want_shape = (request.args.get("shape") or "").strip()
    want_label = (request.args.get("label") or "").strip()

    out = []
    for g in groups:
        if want_shape and g.get("shape") != want_shape:
            continue
        if want_label and g.get("conflict_label") != want_label:
            continue
        g2 = dict(g)
        g2["anchor"] = slim(g.get("anchor_id"))
        g2["partners"] = [slim(fid) for fid in (g.get("partner_ids") or [])]
        # Computed here, not just read from the file, so conflicts files written
        # before this diagnostic existed still get it (see
        # alignment._partner_annotation_groups for why it matters: a fan-out
        # inside ONE annotation is the reification schema copying one
        # natural_language onto N triples, not a granularity disagreement).
        ann_groups = sorted({(p.get("group_id") or "").strip()
                             for p in g2["partners"]} - {""})
        g2.setdefault("partner_annotation_groups", ann_groups)
        g2["partners_one_annotation"] = bool(
            len(g2["partners"]) > 1 and len(ann_groups) == 1)
        out.append(g2)

    params = confl.get("params") or {}
    return jsonify({
        "doc_id": doc_id,
        "available": bool(groups),
        "groups": out,
        "n_total": len(groups),
        "shapes": confl.get("group_shapes") or {},
        "label_counts": confl.get("group_label_counts") or {},
        "label_counts_layer1": confl.get("group_label_counts_layer1") or {},
        "layer2_calls": confl.get("group_layer2_calls", 0),
        "params": {
            "alignment": params.get("alignment"),
            "align_threshold": params.get("align_threshold"),
            "layer2_model": params.get("layer2_model"),
            "arbitrate_shapes": params.get("arbitrate_shapes"),
            "slot_gates": params.get("slot_gates"),
        },
    })


@app.route("/api/gold_compare/<doc_id>")
def api_gold_compare(doc_id: str):
    """Score extracted facts vs the doc's gold_triples (WebNLG / Re-DocRED) and
    return a gold KG. Strict + soft matching; 400 for docs without gold."""
    ppath = PARSED / f"{doc_id}.json"
    if not ppath.exists():
        abort(404, f"parsed doc {doc_id!r} not found")
    parsed = _load_json(ppath)["document"]
    gold = parsed.get("gold_triples") or []
    if not gold:
        abort(400, "no gold triples on this doc (WebNLG / Re-DocRED corpora only)")
    from scripts.run_phase2 import discover_annotators
    from src import reextract_worker as rw
    from src import gold_compare as gc
    facts_root = rw.facts_root_for_version(_variant_arg() or "v1")
    fpa = discover_annotators(facts_root, doc_id)
    extracted = []
    for ann, facts in fpa.items():
        for f in facts:
            f2 = dict(f)
            f2["_annotator"] = ann
            extracted.append(f2)
    result = gc.compare(gold, extracted)
    try:
        soft = gc.compare_soft(gold, extracted)
        result["soft"] = soft
        result["backend"] = soft["backend"]
    except Exception:
        app.logger.exception("soft gold matching failed for %s", doc_id)
        result["soft"], result["backend"] = None, "error"
    result["gold_graph"] = gc.gold_graph(gold)
    result["doc_id"] = doc_id
    result["annotators"] = sorted(fpa)
    result["gold_triples"] = gold
    return jsonify(result)


# `/api/gold_eval` was removed 2026-09-10 with `src/gold_eval.py`. It scored
# against `parsed["gold_triples"]`, a field only WebNLG and Re-DocRED carry,
# and both corpora were dropped from the thesis scope on 2026-07-27. EUR-Lex
# scoring lives in `src/gold_eurlex_eval.py`, which is segment-scoped against
# `data/gold/facts_hires/` and has no UI route.

@app.route("/api/graph/<doc_id>")
def api_graph(doc_id: str):
    """Cytoscape graph. Built from the conflicts doc when conflict detection has
    run (edges carry conflict colours); otherwise built from facts alone — entity
    clustering + S-P-O edges, all `unlabeled` — so the KG is viewable before (or
    without) Phase-2. Conflict labels are an overlay added once Phase-2 runs."""
    from scripts.run_phase2 import cluster_entities, discover_annotators
    from src.kg_build import build_graph, build_reified_graph
    from src import reextract_worker as rw

    cpath = _conflicts_path(doc_id)
    labeled = cpath.exists()
    if labeled:
        confl = _load_json(cpath)
        # Merge annotators present on disk but missing from the Phase-2 file
        # (e.g. cloud:gemma4:latest extracted after the run) so they appear in
        # the graph too — unlabeled until Phase-2 re-runs.
        _disk = discover_annotators(rw.facts_root_for_version(_variant_arg() or "v1"), doc_id)
        _fpa = dict(confl.get("facts_per_annotator", {}))
        for _a, _f in _disk.items():
            _fpa.setdefault(_a, _f)
        confl["facts_per_annotator"] = _fpa
    else:
        # No conflict detection yet — assemble a facts-only doc from disk.
        version = _variant_arg()
        facts_root = rw.facts_root_for_version(version or "v1")
        fpa = discover_annotators(facts_root, doc_id)
        if not fpa:
            return jsonify({"doc_id": doc_id, "nodes": [], "edges": [],
                            "summary": {"n_nodes": 0, "n_edges": 0, "edge_label_counts": {}}})
        confl = {"doc_id": doc_id, "params": {}, "annotators": list(fpa),
                 "facts_per_annotator": fpa, "aligned_pairs": [],
                 "entity_clusters": {}, "label_counts": {}}

    params = confl.get("params", {})
    want_thr = float(request.args.get("merge_threshold",
                                      params.get("merge_threshold", 0.78)))
    core_flag = request.args.get("core", "1") != "0"
    # reify=1 -> statement-node graph (mirrors the Neo4j reification); cached in
    # a SEPARATE file so the flat graph (still used by the Multiples tab) and the
    # reified graph for the same doc never overwrite each other.
    reify = request.args.get("reify", "0") != "0"
    gpath = _graph_cache_path(doc_id)
    if reify:
        gpath = gpath.with_name(gpath.stem + "__reified.json")

    # Serve cache only when knobs match AND the labeled/unlabeled mode matches
    # (so a Phase-2 run's labelled graph supersedes an earlier facts-only one).
    annot_key = "|".join(sorted(confl.get("facts_per_annotator", {}).keys()))
    # Invalidate the cache when the conflicts file was rewritten AFTER the graph
    # was cached (e.g. Phase-2 re-ran and changed conflict labels) even if the
    # annotator set is unchanged -- otherwise a fresh run's colours never show.
    cmtime = cpath.stat().st_mtime if (labeled and cpath.exists()) else 0.0
    if gpath.exists() and gpath.stat().st_mtime >= cmtime:
        g = _load_json(gpath)
        gp = g.get("params", {})
        try:
            if (abs(float(gp.get("merge_threshold", -1)) - want_thr) < 1e-9
                    and bool(gp.get("core_entities", False)) == core_flag
                    and bool(gp.get("labeled", True)) == labeled
                    and bool(g.get("reified", False)) == reify
                    and gp.get("annot_key") == annot_key):
                return jsonify(g)
        except (TypeError, ValueError):
            pass

    cfb = dict(confl)
    cfb["entity_clusters"] = cluster_entities(
        confl["facts_per_annotator"], merge_threshold=want_thr,
        core_entities=core_flag)
    cfb["params"] = dict(params)
    cfb["params"]["merge_threshold"] = want_thr
    cfb["params"]["core_entities"] = core_flag
    graph = build_reified_graph(cfb) if reify else build_graph(cfb)
    graph.setdefault("params", {})["labeled"] = labeled
    graph["params"]["annot_key"] = annot_key
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonify(graph)


@app.route("/api/verify/<doc_id>/<fact_id>", methods=["POST"])
def api_verify(doc_id: str, fact_id: str):
    payload = request.get_json(silent=True) or {}
    status = payload.get("status")
    if status not in ("verified", "rejected", "unset"):
        abort(400, "status must be one of: verified, rejected, unset")
    VERIFICATIONS.mkdir(parents=True, exist_ok=True)
    vpath = VERIFICATIONS / f"{doc_id}.json"
    cur = _load_json(vpath) if vpath.exists() else {}
    if status == "unset":
        cur.pop(fact_id, None)
    else:
        cur[fact_id] = {"status": status, "note": payload.get("note", "")}
    vpath.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonify({"ok": True, "fact_id": fact_id, "status": status, "n_verified": len(cur)})


@app.route("/api/coverage/<doc_id>")
def api_coverage(doc_id: str):
    cpath, ppath = _conflicts_path(doc_id), PARSED / f"{doc_id}.json"
    if not cpath.exists() or not ppath.exists():
        return jsonify({})
    from src.coverage import compute_coverage
    return jsonify(compute_coverage(_load_json(cpath)["facts_per_annotator"], _load_json(ppath)))


@app.route("/api/distribution_shift/<doc_id>")
def api_distribution_shift(doc_id: str):
    from src.distribution_shift import UNITS, compare_conflict_files
    v1_label = request.args.get("v1", "v1")
    v2_label = request.args.get("v2")
    # `group` is the unit of analysis (method.tex); `?unit=pair` keeps the old
    # pair-level view, which is what the KG edges are shaped like.
    unit = (request.args.get("unit") or "group").strip()
    if unit not in UNITS:
        abort(400, f"unit must be one of {UNITS}")
    def _path(version: str) -> Path:
        return CONFLICTS / (f"{doc_id}.json" if version == "v1" else f"{doc_id}__{version}.json")
    def _empty(note: str) -> dict:
        return {"v1_label": v1_label, "v2_label": v2_label,
                "note": note,
                "labels": [], "v1": [], "v2": [], "delta": [], "delta_pct": [],
                "totals": {"v1": 0, "v2": 0, "delta": 0, "delta_pct": None},
                "layer1_rate": {"v1": {}, "v2": {}}}
    p1 = _path(v1_label)
    if not p1.exists():
        return jsonify(_empty(
            f"No Phase-2 output for '{v1_label}' on this doc yet — "
            f"run Phase-2 first (Background runs tab)."))
    if v2_label is None:
        cands = sorted(CONFLICTS.glob(f"{doc_id}__*.json"), key=lambda p: p.stat().st_mtime)
        if not cands:
            return jsonify(_empty(
                "No v2 conflicts file found yet — "
                "save a new guideline version then re-run Phase-1 + Phase-2."))
        p2 = cands[-1]
        v2_label = p2.stem.split("__", 1)[1]
    else:
        p2 = _path(v2_label)
        if not p2.exists():
            return jsonify(_empty(f"conflicts for '{v2_label}' not found on disk"))
    try:
        out = compare_conflict_files(p1, p2, v1_label=v1_label, v2_label=v2_label,
                                     unit=unit)
        # Annotator sets of both sides: when they differ, pair counts are not
        # comparable (fewer annotator pairs => mechanically fewer aligned
        # pairs) and the UI must say so before anyone reads the deltas.
        try:
            out["v1_annotators"] = sorted(_load_json(p1).get("annotators", []))
            out["v2_annotators"] = sorted(_load_json(p2).get("annotators", []))
            out["annotators_match"] = out["v1_annotators"] == out["v2_annotators"]
        except Exception:
            pass
        return jsonify(out)
    except Exception as exc:
        log.exception("distribution_shift failed for %s", doc_id)
        return jsonify(_empty(f"comparison failed: {type(exc).__name__}: {exc}"))


# ===================== uploads & guidelines ============================


@app.route("/api/upload_text", methods=["POST"])
def api_upload_text():
    payload = request.get_json(silent=True) or {}
    title = (payload.get("title") or "").strip()
    text = (payload.get("text") or "")
    split_mode = payload.get("split_mode", "paragraph")
    if not text.strip():
        abort(400, "text is required")

    from src.text_ingest import build_parsed_doc, write_parsed_doc
    parsed = build_parsed_doc(title=title, text=text, split_mode=split_mode)
    out = write_parsed_doc(PARSED, parsed)
    doc_id = parsed["document"]["celex"]
    response: dict = {"ok": True, "doc_id": doc_id,
                      "title": parsed["document"]["title"],
                      "n_sections": len(parsed["document"]["recitals"]),
                      "parsed_path": str(out)}

    if payload.get("extract"):
        from src import reextract_worker as rw
        model = payload.get("model") or _of_record()["annotators"][0]
        guideline_version = payload.get("guideline_version") or "v1"
        existing = PROMPTS / f"extract_{guideline_version}.md"
        if not existing.exists():
            abort(400, f"guideline {guideline_version!r} not found")
        job = rw.enqueue_reextract_with_version(
            guideline_version, models=[model], doc_paths=[response["parsed_path"]],
            label=f"upload extract ({model} × {doc_id})",
        )
        response["job_id"] = job.job_id
    return jsonify(response), 201


_GUIDELINE_VERSION_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")

def _guideline_path(version: str) -> Path:
    if not _GUIDELINE_VERSION_RE.match(version or ""):
        abort(400, "version must be alphanumeric/underscore/dot/dash")
    return PROMPTS / f"extract_{version}.md"


def _guideline_lock(version: str) -> Path | None:
    """The `.lock.json` beside a frozen guideline, if there is one."""
    p = PROMPTS / f"extract_{version}.lock.json"
    return p if p.exists() else None


@app.route("/api/arbitrate_prompts")
def api_arbitrate_prompts_list():
    """List all prompts/arbitrate_*.md versions."""
    out = []
    for p in sorted(PROMPTS.glob("arbitrate_*.md")):
        version = p.stem.removeprefix("arbitrate_")
        out.append({"version": version, "filename": p.name, "size": p.stat().st_size})
    return jsonify({"arbitrate_prompts": out})


@app.route("/api/schema_prompts")
def api_schema_prompts_list():
    """List all prompts/schema_*.md versions."""
    out = []
    for p in sorted(PROMPTS.glob("schema_*.md")):
        version = p.stem.removeprefix("schema_")
        out.append({"version": version, "filename": p.name, "size": p.stat().st_size})
    return jsonify({"schema_prompts": out})


@app.route("/api/schema_prompt", methods=["GET", "PUT", "DELETE"])
def api_schema_prompt():
    """Read/write/delete prompts/schema_<version>.md."""
    version = (request.args.get("version") or "v1").strip()
    if not _GUIDELINE_VERSION_RE.match(version):
        abort(400, "version must be alphanumeric/underscore/dot/dash")
    path = PROMPTS / f"schema_{version}.md"
    if request.method == "DELETE":
        if version == "v1":
            abort(403, "v1 schema is protected and cannot be deleted")
        if not path.exists():
            abort(404, f"schema {version!r} not found")
        path.unlink()
        return jsonify({"ok": True, "version": version, "deleted": True})
    if request.method == "GET":
        if not path.exists():
            abort(404, f"schema {version!r} not found")
        return jsonify({"version": version, "text": path.read_text(encoding="utf-8"),
                        "filename": path.name})
    payload = request.get_json(silent=True) or {}
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        abort(400, "text required")
    path.write_text(text, encoding="utf-8")
    return jsonify({"version": version, "saved": True, "size": path.stat().st_size})


@app.route("/api/arbitrate_prompt", methods=["GET", "PUT", "DELETE"])
def api_arbitrate_prompt():
    """Read/write/delete prompts/arbitrate_<version>.md (the conflict-detection prompt)."""
    version = (request.args.get("version") or "v1").strip()
    if not _GUIDELINE_VERSION_RE.match(version):
        abort(400, "version must be alphanumeric/underscore/dot/dash")
    path = PROMPTS / f"arbitrate_{version}.md"
    if request.method == "DELETE":
        if version == "v1":
            abort(403, "v1 is the baseline arbitrate prompt and cannot be deleted")
        if not path.exists():
            abort(404, f"arbitrate prompt {version!r} not found")
        path.unlink()
        return jsonify({"version": version, "deleted": True})
    if request.method == "GET":
        if not path.exists():
            abort(404, f"arbitrate prompt {version!r} not found")
        return jsonify({"version": version, "text": path.read_text(encoding="utf-8"),
                        "filename": path.name})
    payload = request.get_json(silent=True) or {}
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        abort(400, "text required")
    path.write_text(text, encoding="utf-8")
    return jsonify({"version": version, "saved": True, "size": path.stat().st_size})


@app.route("/api/arbitrate_test", methods=["POST"])
def api_arbitrate_test():
    """Run one conflict-detection (Layer-2) arbitration on a pair and return the
    raw label/reason. Lenient — accepts any label the prompt defines."""
    payload = request.get_json(silent=True) or {}
    from src.conflict_layer2 import arbitrate_once
    try:
        res = arbitrate_once(
            doc_title=payload.get("doc_title", ""),
            source_quote=payload.get("source_quote", ""),
            fact_a=payload.get("fact_a", ""),
            fact_b=payload.get("fact_b", ""),
            model_name=payload.get("model") or _of_record()["layer2"],
            base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
            template=payload.get("prompt"),
            version=payload.get("version", "v1"),
        )
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 200
    return jsonify(res)


@app.route("/api/models")
def api_models():
    """Available models grouped by provider, for the Phase-1/2 pickers.

    {"local": [{id, tag, provider}], "cloud": [...]} — cloud entries carry the
    ``cloud:`` prefix in their id and are omitted when no API key is set.
    Live /api/tags is attempted per endpoint with a short timeout; on failure
    the static fallback lists seed the dropdowns so the UI is never empty.
    """
    return jsonify(providers.list_available_models())


@app.route("/api/guidelines")
def api_guidelines_list():
    if not PROMPTS.exists():
        return jsonify({"guidelines": []})
    out = []
    for p in sorted(PROMPTS.glob("extract_*.md")):
        ver = p.stem.removeprefix("extract_")
        try:
            body = p.read_text(encoding="utf-8")
            m = re.search(r"#+\s*Guideline\s*\(([^)]+)\)", body)
            summary = m.group(0) if m else body.strip().split("\n", 1)[0][:120]
            out.append({"version": ver, "filename": p.name,
                        "size": p.stat().st_size, "mtime": p.stat().st_mtime,
                        "summary": summary,
                        # a frozen arm stays runnable but is not editing
                        # material — see api_guideline_one
                        "locked": _guideline_lock(ver) is not None})
        except Exception as exc:
            log.warning("guideline %s unreadable: %s", p, exc)
    return jsonify({"guidelines": out})


def _guard_guideline_write(version: str, text: str) -> None:
    """Everything that must be true before a guideline version is written.

    Shared by the editor's Save and by "Save & log", which writes the same file:
    a second copy of these rules is a second place for one of them to go missing.
    """
    if version == "v1":
        abort(403, "v1 is the baseline guideline and cannot be overwritten; "
                   "put a new version in 'Save as'")
    if _guideline_lock(version) is not None:
        abort(423, f"extract_{version}.md is frozen by "
                   f"extract_{version}.lock.json; a committed arm is not "
                   f"edited, it is superseded by a new version")
    # The sentinels belong to the SCHEMA FRAME (prompts/schema_*.md), not to a
    # guideline — that is the whole point of the 2026-06-19 split, and it is what
    # stops a guideline revision from changing the output contract (invariant 3).
    # This check used to REQUIRE them, so no guideline written after the split
    # could be saved at all, and one that satisfied it would have been taken as a
    # legacy self-contained prompt by `extractor.compose_prompt` and replaced the
    # frozen frame in silence (BUG-2026-09-18-B).
    frame_only = [x for x in ("<<DOC_TITLE>>", "<<SECTION_PATH>>",
                              "<<SECTION_TEXT>>") if x in text]
    if frame_only:
        abort(400, f"a guideline must not contain the schema frame's sentinels "
                   f"{frame_only}: they live in prompts/schema_*.md, and a "
                   f"guideline carrying them replaces the frozen frame")


@app.route("/api/guidelines/<version>", methods=["GET", "PUT", "DELETE"])
def api_guideline_one(version: str):
    path = _guideline_path(version)
    if request.method == "DELETE":
        if version == "v1":
            abort(403, "v1 is the baseline guideline and cannot be deleted")
        if _guideline_lock(version) is not None:
            abort(423, f"extract_{version}.md is a committed arm of the "
                       f"experiment and cannot be deleted from here")
        if not path.exists():
            abort(404, f"guideline {version!r} not found")
        path.unlink()
        return jsonify({"ok": True, "version": version, "deleted": True})
    if request.method == "GET":
        if not path.exists():
            abort(404, f"guideline {version!r} not found")
        # A frozen guideline is not editing material *in this interface*. The
        # no-feedback control g_v2' is the case that matters: one author writes
        # both arms, so the only carryover the tooling can actually prevent is
        # re-reading the control while authoring g_v2 at the same keyboard. The
        # version stays selectable for Phase-1/Phase-2 runs — it has to be, its
        # facts are the point — and remains readable on disk and in the control
        # console. This closes the one-click path, not the door.
        if _guideline_lock(version) is not None:
            abort(423, f"extract_{version}.md is frozen "
                       f"(extract_{version}.lock.json) and is deliberately not "
                       f"loaded into the editor: it is a committed arm of the "
                       f"experiment, not a draft. It stays available for runs.")
        return jsonify({"version": version, "text": path.read_text(encoding="utf-8"),
                        "filename": path.name, "size": path.stat().st_size})
    from src.guideline_log import normalize_newlines
    payload = request.get_json(silent=True) or {}
    text = normalize_newlines(payload.get("text", ""))
    if not text.strip():
        abort(400, "text body is required")
    # The sentinels belong to the SCHEMA FRAME (prompts/schema_*.md), not to a
    # guideline — that is the whole point of the 2026-06-19 split, and it is
    # what stops a guideline revision from changing the output contract
    # (invariant 3). This check used to REQUIRE them, so no guideline written
    # after the split could be saved at all, and one that satisfied it would
    # have been taken as a legacy self-contained prompt by
    # `extractor.compose_prompt` and replaced the frozen frame in silence
    # (BUG-2026-09-18-B).
    _guard_guideline_write(version, text)
    PROMPTS.mkdir(parents=True, exist_ok=True)
    # `newline=""` is half of BUG-2026-09-18-C. Normalising the textarea's
    # CRLF on the way in is useless on Windows, where text-mode writing
    # translates every LF back to CRLF on the way out: extract_v2.md came back
    # 257 bytes larger than extract_v1.md and git showed every line as changed
    # — exactly the unreadable v1->v2 diff the normalisation exists to prevent,
    # and that diff IS the intervention.
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    # Same checkpointing as the control arm: a version you can only see the
    # latest state of is not a revision you can report (see _draft_log).
    _draft_log(version).checkpoint(
        text, datetime.now(timezone.utc), bool(payload.get("explicit", True)))
    return jsonify({"ok": True, "version": version, "size": path.stat().st_size}), 200


# ---------------- the revision record: why each edit, citing what -----------
#
# The diagnosed arm's half of `src/guideline_log.py`. The control console
# (scripts/serve_control_console.py) writes the same artefact from raw baseline
# output; here the citations are what the interface showed — a pair, a group, a
# conflict label, a review. RQ2 compares the two revisions, so the two records
# have to be the same kind of object, which is why neither side rolls its own.

def _draft_log(version: str):
    from src.guideline_log import DraftLog
    draft = _guideline_path(version)
    return DraftLog(draft=draft, seed=PROMPTS / "extract_v1.md",
                    snaps_dir=PROMPTS / "_drafts" / f"extract_{version}",
                    prefix="d_", artifact="g_v2_change_log",
                    # Screenshots sit beside the checkpoints, under the version
                    # they explain: the change log, its history and its pictures
                    # are one artefact and are moved, archived or cited as one.
                    shots_dir=PROMPTS / "_shots" / f"extract_{version}")


def _draft_text(version: str) -> str:
    path = _guideline_path(version)
    return path.read_text(encoding="utf-8") if path.exists() else ""


@app.route("/api/guideline_log/<version>", methods=["GET", "POST"])
def api_guideline_log(version: str):
    log = _draft_log(version)
    if request.method == "GET":
        return jsonify(log.read_log())
    payload = request.get_json(silent=True) or {}
    # The lock covers the reasons, not only the text — its `change_log` digest is
    # what makes a later edit detectable, and a digest you are still allowed to
    # invalidate is not a freeze. This must be checked for EVERY write to the
    # log, not only the ones that carry a draft: an entry appended without text
    # would slip past the write guard below (which only runs when `text` is
    # present) and silently break the lock.
    if _guideline_lock(version) is not None:
        abort(423, f"extract_{version}.md is frozen by extract_{version}.lock.json: "
                   f"its change log is part of the committed artefact and cannot "
                   f"be added to, edited or withdrawn")
    if "remove" in payload:
        if not log.remove_entry(payload["remove"]):
            abort(404, "no such entry")
        return jsonify({"entries": log.entries()})
    entry = payload.get("add")
    if not isinstance(entry, dict):
        abort(400, "add or remove required")
    now = datetime.now(timezone.utc)
    # One press: save the draft, checkpoint it, record the reason AND the lines
    # it changed. Splitting them is what produced a logged reason against a file
    # that had never been saved (BUG-2026-09-20-D). The control console gets the
    # same effect from autosaving; here the button is the save.
    text = payload.get("text")
    if text is not None:
        from src.guideline_log import normalize_newlines
        text = normalize_newlines(text)
        _guard_guideline_write(version, text)
        path = _guideline_path(version)
        PROMPTS.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as fh:
            fh.write(text)
    try:
        log.add_entry(entry, now, text=text)
    except ValueError as exc:
        abort(400, str(exc))
    # A reason logged against a draft that is still byte-identical to its seed
    # is the commonest way to lose an edit: the text was typed into the editor
    # and Save was never pressed, so the record claims a revision the file does
    # not carry. Not a refusal — a reason may legitimately be written before or
    # after the edit (see `written_late`) — but it must be said out loud.
    seed = PROMPTS / "extract_v1.md"
    draft = _guideline_path(version)
    unchanged = (draft.exists() and seed.exists()
                 and draft.read_text(encoding="utf-8")
                 == seed.read_text(encoding="utf-8"))
    return jsonify({"entries": log.entries(),
                    "saved_at": now.strftime("%H:%M:%SZ"),
                    "draft_matches_seed": bool(unchanged)})


@app.route("/api/guideline_shot/<version>", methods=["POST"])
def api_guideline_shot(version: str):
    """Attach a screenshot to the change log of `version`.

    Takes either a multipart `file` or a JSON `data:image/...;base64,...` URL,
    because the two ways a screenshot arrives are a file picker and Ctrl-V from
    the clipboard, and the second one has no file. The bytes are what is stored;
    whatever the browser called the file is discarded (`add_shot` decides the
    extension by sniffing, see the note there).
    """
    if version == "v1":
        abort(403, "v1 is the baseline guideline, not a revision under "
                   "diagnosis; it has no change log to illustrate")
    data = b""
    up = request.files.get("file")
    if up is not None:
        data = up.read()
    else:
        payload = request.get_json(silent=True) or {}
        raw = payload.get("data_url") or ""
        if "," in raw:
            import base64
            try:
                data = base64.b64decode(raw.split(",", 1)[1], validate=True)
            except Exception:
                abort(400, "data_url is not valid base64")
    if not data:
        abort(400, "no image: send multipart 'file' or JSON 'data_url'")
    try:
        shot = _draft_log(version).add_shot(
            data, datetime.now(timezone.utc),
            caption=(request.form.get("caption")
                     or (request.get_json(silent=True) or {}).get("caption") or ""))
    except ValueError as exc:
        abort(400, str(exc))
    shot["url"] = f"/api/guideline_shot/{version}/{shot['file']}"
    return jsonify(shot), 201


@app.route("/api/guideline_shot/<version>/<name>", methods=["GET"])
def api_guideline_shot_file(version: str, name: str):
    from flask import send_file
    from src.guideline_log import sniff_shot
    path = _draft_log(version).shot_path(name)
    if path is None or not path.exists():
        abort(404, "no such screenshot")
    kind = sniff_shot(path.read_bytes()[:16])
    # Serve the type the bytes are, never one derived from the request, and
    # never inline-guessed by the browser: nosniff + an explicit image type is
    # what keeps a stored file from being run as script on our own origin.
    resp = send_file(path, mimetype=(kind[1] if kind else "application/octet-stream"))
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp


@app.route("/api/guideline_history/<version>", methods=["GET", "POST"])
def api_guideline_history(version: str):
    log = _draft_log(version)
    if request.method == "POST":
        if not (request.get_json(silent=True) or {}).get("prune"):
            abort(400, "prune required")
        removed = log.prune()
        return jsonify({"removed": removed, "kept": len(log.snaps())})
    name = request.args.get("name")
    if name:
        detail = log.detail(name, _draft_text(version))
        if detail is None:
            abort(404, "no such checkpoint")
        return jsonify(detail)
    return jsonify(log.history(_draft_text(version) or None))


@app.route("/api/guideline_freeze/<version>", methods=["GET", "POST"])
def api_guideline_freeze(version: str):
    """Freeze a guideline version: the diagnosed arm's half of the commit.

    Same lock shape as the control console's (`extract_v2_control.lock.json`), so
    the two arms are frozen the same way and the write-up can compare them
    without comparing two formats: timestamp, sha256 of the text, size, and the
    change log's digest. The digest is what makes a later edit to the *reasons*
    detectable — freezing the guideline and leaving its record editable would
    freeze the artefact and not the account of it.

    What this refuses, and why:
      * v1 — the baseline is not an arm, it is what every arm is measured against
      * an empty change log — a revision with no recorded reason is not
        reportable as a *diagnosed* revision; that is the whole claim of the arm
      * a draft identical to the seed — nothing was revised
      * re-freezing — a committed arm is superseded by a new version, not edited
        (the control console broke its own freeze once, on 2026-09-18, and the
        recovery was to keep the superseded lock and record the break; the same
        route is available here by hand and is deliberately not a button)
    """
    import hashlib
    lock_path = PROMPTS / f"extract_{version}.lock.json"
    if request.method == "GET":
        if not lock_path.exists():
            return jsonify({"frozen": False})
        return jsonify({"frozen": True, "lock": _load_json(lock_path)})

    if version == "v1":
        abort(403, "v1 is the baseline, not an arm of the experiment")
    if lock_path.exists():
        abort(409, f"extract_{version}.md is already frozen "
                   f"({lock_path.name}); a committed arm is superseded by a new "
                   f"version, not re-frozen")
    draft = _guideline_path(version)
    if not draft.exists():
        abort(404, f"guideline {version!r} does not exist")
    text = draft.read_text(encoding="utf-8")
    seed = PROMPTS / "extract_v1.md"
    if seed.exists() and text == seed.read_text(encoding="utf-8"):
        abort(400, "this draft is byte-identical to extract_v1.md: there is "
                   "nothing to freeze")
    log = _draft_log(version)
    entries = log.entries()
    if not entries:
        abort(400, "the change log is empty. A revision with no recorded reason "
                   "cannot be reported as a diagnosed one — that is the claim "
                   "this arm exists to support")
    now = datetime.now(timezone.utc)
    lock = {
        "artifact": "g_v2_conflict_informed_revision",
        "path": f"prompts/{draft.name}",
        "committed_at": now.strftime("%Y-%m-%d %H:%M:%SZ"),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "chars": len(text),
        "words": len(text.split()),
        "seeded_from": "prompts/extract_v1.md",
        "written_from": "prompts/extract_v1.md + the VA interface "
                        "(alignment, conflict labels, groups, reviews)",
        "entries": len(entries),
        "cites": sum(len(e.get("cites") or []) for e in entries),
        "shots": sum(len(e.get("shots") or []) for e in entries),
        "change_log": log.digest(),
    }
    lock_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return jsonify({"frozen": True, "lock": lock}), 201


def _single_section_doc(doc, section_path: str):
    """A copy of `doc` whose `enumerate_sections` yields exactly `section_path`,
    at the same path and offsets as in a full run.

    The probe used to wrap the clause as a one-recital "mini" document. That
    sliced the preamble at an article's offsets (the wrong text for every
    article) and labelled every probed clause `preamble.recitals[0]`, so its
    facts were filed under Recital 1 and given Recital 1's ids — `group_id` and
    `fact_id` hash (model, doc, section_path, index) (BUG-2026-09-23-D). Here
    the lists are cut down instead: blank placeholders keep a recital's index
    (`enumerate_sections` skips a recital whose span is empty), and an article
    is addressed by its number, which is its path.
    """
    from src.extractor import enumerate_sections
    from src.schema import Recital
    m = re.fullmatch(r"preamble\.recitals\[(\d+)\]", section_path)
    if m:
        k = int(m.group(1))
        blank = Recital(number="", text="", char_start=0, char_end=0)
        update = {"recitals": [blank] * k + [doc.recitals[k]], "articles": []}
    else:
        update = {"recitals": [], "articles": [
            a for a in doc.articles if f"enacting.article_{a.number}" == section_path]}
    one = doc.model_copy(update=update)
    got = {s.section_path for s in enumerate_sections(one)}
    if got != {section_path}:   # never extract anything but what was asked for
        abort(500, f"could not isolate {section_path!r} for re-extraction "
                   f"(got {sorted(got)})")
    return one


@app.route("/api/reextract_span", methods=["POST"])
def api_reextract_span():
    """Re-extract one clause (or a preamble span) with one annotator under one
    guideline version: the probe behind each segment's annotate bar.

    It writes into the selected version's facts root — the batch jobs'
    convention, `reextract_worker.facts_root_for_version` — and never into a
    cell of a batch run of record. It used to write into `data/facts/` whatever
    version was chosen, so a v2 probe put v2 facts into the frozen v1 baseline
    (invariant 3; BUG-2026-09-23-D).
    """
    payload = request.get_json(silent=True) or {}
    doc_id = payload.get("doc_id")
    model = payload.get("model") or _of_record()["annotators"][0]
    guideline_version = (payload.get("guideline_version") or "v1").strip()
    section_path = payload.get("section_path")
    char_start = payload.get("char_start")
    char_end = payload.get("char_end")
    if not doc_id:
        abort(400, "doc_id is required")
    # Invariant 1: the locked eval set is touched once per guideline version, by
    # a recorded batch run, to produce a result. This endpoint is the opposite
    # kind of act — an interactive re-run of one clause while looking at it,
    # which is tuning against the eval set however it is labelled. The eval pass
    # goes through scripts/run_extraction.py, so refusing here costs nothing.
    _refuse_eval([doc_id], "Re-extraction from the interface")
    if not _GUIDELINE_VERSION_RE.match(guideline_version):
        abort(400, "invalid guideline_version")
    if not (PROMPTS / f"extract_{guideline_version}.md").exists():
        abort(400, f"guideline {guideline_version!r} not found")

    ppath = PARSED / f"{doc_id}.json"
    if not ppath.exists():
        abort(404, f"parsed doc {doc_id!r} not found")
    parsed = _load_json(ppath)["document"]
    from src.schema import ParsedDocument
    doc_pyd = ParsedDocument(**parsed)

    if section_path:
        from src.extractor import enumerate_sections
        if not any(s.section_path == section_path
                   for s in enumerate_sections(doc_pyd)):
            abort(404, f"section_path {section_path!r} not found in doc")
        probe_doc = _single_section_doc(doc_pyd, section_path)

        def _replaced(f: dict) -> bool:
            return (f.get("source_locator") or {}).get("section_path") == section_path
    elif char_start is not None and char_end is not None:
        cs, ce = int(char_start), int(char_end)
        if cs < 0 or ce <= cs:
            abort(400, "invalid char_start/char_end")
        section_text = parsed["preamble_text"][cs:ce]
        if not section_text.strip():
            abort(400, "selected span is empty")
        section_path = f"preamble.span[{cs}:{ce}]"
        probe_doc = ParsedDocument(**{
            "celex": doc_id, "title": parsed.get("title", doc_id),
            "citations": [], "annexes": [], "articles": [], "concluding_formulas": "",
            "preamble_text": parsed.get("preamble_text", ""),
            "enacting_text": parsed.get("enacting_text", ""),
            "recitals": [{"number": "(span)", "text": section_text,
                          "char_start": cs, "char_end": ce}],
        })

        def _replaced(f: dict) -> bool:
            loc = f.get("source_locator") or {}
            return (loc.get("section_path", "").startswith("preamble")
                    and loc.get("char_start") is not None
                    and cs <= loc.get("char_start", -1) < ce)
    else:
        abort(400, "either section_path or (char_start,char_end) is required")

    from src import reextract_worker as rw
    out_root = rw.facts_root_for_version(guideline_version)
    # Before the model is called: the same refusal after minutes of cloud time
    # would be the same answer, later.
    _refuse_record_overwrite(
        payload, f"Re-extracting {doc_id} with {model} under {guideline_version}",
        facts_cells=[(out_root, model, doc_id)])

    from src.extractor import extract
    try:
        new_facts = extract(model_name=model, parsed_doc=probe_doc,
                            guideline_version=guideline_version,
                            ensure_solo=True, unload_after=False)
    except Exception as exc:
        log.exception("reextract_span failed")
        abort(500, f"{type(exc).__name__}: {exc}")

    fp = record_guard.facts_file(out_root, model, doc_id)
    if fp.exists():
        out = json.loads(fp.read_text(encoding="utf-8"))
    else:   # the shape extractor.write_facts writes
        out = {"doc_id": doc_id, "annotator": model,
               "guideline_version": guideline_version, "fact_count": 0, "facts": []}
    before = out.get("facts", [])
    kept = [f for f in before if not _replaced(f)]
    out["facts"] = kept + [f.model_dump(mode="json") for f in new_facts]
    out["fact_count"] = len(out["facts"])
    # A file a probe has touched is no longer what a batch run wrote; it says so
    # itself, so nobody mistakes it for one later.
    out.setdefault("probes", []).append({
        "section_path": section_path,
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "n_new": len(new_facts), "n_replaced": len(before) - len(kept)})
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    return jsonify({"ok": True, "doc_id": doc_id, "model": model,
                    "guideline_version": guideline_version,
                    "section_path": section_path,
                    "facts_root": _rel(out_root), "out_path": _rel(fp),
                    "n_new_facts": len(new_facts),
                    "n_replaced": len(before) - len(kept)})


# ===================== Background jobs ============================


@app.route("/api/guideline", methods=["POST"])
def api_guideline_save_and_enqueue():
    payload = request.get_json(silent=True) or {}
    guideline_text = payload.get("guideline_text", "")
    if not guideline_text.strip():
        abort(400, "guideline_text is required")
    models = payload.get("models") or _of_record()["annotators"]
    doc_paths = payload.get("doc_paths")
    excluded: list[str] = []
    if doc_paths:
        _refuse_eval([Path(p).stem for p in doc_paths], "Re-extraction")
    else:
        kept, excluded = _without_eval([p.stem for p in sorted(PARSED.glob("*.json"))])
        doc_paths = [str(PARSED / f"{d}.json") for d in kept]
    # The job writes a fresh `v2_<token>` version to its own facts root, so it
    # cannot reach a run of record; only the eval refusal applies.
    from src import reextract_worker as rw
    job = rw.enqueue(guideline_text, models=models, doc_paths=doc_paths,
                     label=f"new-guideline re-extract ({len(models)}m × {len(doc_paths)}d)")
    return jsonify({**rw.to_dict(job), "excluded_eval": excluded}), 202


@app.route("/api/run_phase1", methods=["POST"])
def api_run_phase1():
    """Phase-1 batch from UI: extract on all (model × doc) cells with existing guideline."""
    payload = request.get_json(silent=True) or {}
    models = payload.get("models") or _of_record()["annotators"]
    guideline_version = (payload.get("guideline_version") or "v1").strip()
    if not _GUIDELINE_VERSION_RE.match(guideline_version):
        abort(400, "invalid guideline_version")
    doc_ids = payload.get("doc_ids")
    excluded: list[str] = []
    if doc_ids:
        _refuse_eval(doc_ids, "Phase-1 from the interface")
        doc_ids = [d for d in doc_ids if (PARSED / f"{d}.json").exists()]
    else:
        doc_ids, excluded = _without_eval(_doc_ids_available())
    if not doc_ids:
        abort(400, "no parsed docs available")
    from src import reextract_worker as rw
    root = rw.facts_root_for_version(guideline_version)
    _refuse_record_overwrite(
        payload, f"Phase-1 ({guideline_version})",
        facts_cells=[(root, m, d) for m in models for d in doc_ids])
    doc_paths = [str(PARSED / f"{d}.json") for d in doc_ids]
    try:
        job = rw.enqueue_reextract_with_version(
            guideline_version, models=models, doc_paths=doc_paths,
            label=f"phase-1 ({guideline_version}, {len(models)}m × {len(doc_paths)}d)")
    except FileNotFoundError as exc:
        abort(400, str(exc))
    return jsonify({**rw.to_dict(job), "excluded_eval": excluded}), 202


@app.route("/api/run_phase2", methods=["POST"])
def api_run_phase2():
    """Phase-2 batch from UI: alignment + Layer-1 + Layer-2 over N docs."""
    from scripts.run_phase2 import discover_all_doc_ids
    payload = request.get_json(silent=True) or {}
    from src import reextract_worker as rw
    version = (payload.get("version") or "v1").strip()
    if not _GUIDELINE_VERSION_RE.match(version):
        abort(400, "invalid version")
    facts_root = rw.facts_root_for_version(version)
    out_suffix = "" if version in ("", "v1") else version
    doc_ids = payload.get("doc_ids")
    excluded: list[str] = []
    if doc_ids:
        _refuse_eval(doc_ids, "Conflict detection from the interface")
    else:
        doc_ids, excluded = _without_eval(discover_all_doc_ids(facts_root))
    if not doc_ids:
        abort(400, f"no docs with >=2 annotators found under {facts_root}")
    _refuse_frozen_phase2(version, payload, f"Phase-2 ({version})")
    _refuse_record_overwrite(
        payload, f"Phase-2 ({version})",
        conflicts=[_conflicts_path_for(d, out_suffix) for d in doc_ids])
    arbitrate_version = (payload.get("arbitrate_version") or "v1").strip()
    if not _GUIDELINE_VERSION_RE.match(arbitrate_version):
        arbitrate_version = "v1"
    params = {
        # the arbitrator of record unless the caller asks for the stub; a
        # request that says nothing used to get the stub (BUG-2026-09-23-G)
        "skip_layer2": bool(payload.get("skip_layer2", False)),
        "align_threshold": float(payload.get("align_threshold", 0.78)),
        "redundancy_cosine": float(payload.get("redundancy_cosine", 0.95)),
        "merge_threshold": float(payload.get("merge_threshold", 0.78)),
        "layer2_model": payload.get("layer2_model") or _of_record()["layer2"],
        "layer2_url": payload.get("layer2_url", os.environ.get("OLLAMA_URL", "http://localhost:11434")),
        "facts_root": str(facts_root),
        "out_suffix": out_suffix,
        "arbitrate_version": arbitrate_version,
    }
    job = rw.enqueue_phase2(
        doc_ids=doc_ids, params=params,
        label=f"phase-2 ({version}, {len(doc_ids)} docs, {'stub' if params['skip_layer2'] else params['layer2_model']}, arb={arbitrate_version})")
    return jsonify({**rw.to_dict(job), "excluded_eval": excluded}), 202


@app.route("/api/run_matrix", methods=["POST"])
def api_run_matrix():
    payload = request.get_json(silent=True) or {}
    cells = payload.get("cells", [])
    if not isinstance(cells, list) or not cells:
        abort(400, "cells must be a non-empty list")
    from src import reextract_worker as rw
    # All or nothing, like the other entry points: a submission that names an
    # eval document, or a cell of a run of record, is refused as a whole rather
    # than queued minus the offending cells.
    cells = [c for c in cells if isinstance(c, dict)]
    _refuse_eval([c.get("doc_id") for c in cells if c.get("doc_id")],
                 "The run matrix")
    _refuse_record_overwrite(payload, "The run matrix", facts_cells=[
        (rw.facts_root_for_version(c.get("guideline_version", "v1")),
         c["model"], c["doc_id"])
        for c in cells if c.get("doc_id") and c.get("model")
        and _GUIDELINE_VERSION_RE.match(c.get("guideline_version", "v1") or "")])
    out: list[dict] = []
    for cell in cells:
        doc_id = cell.get("doc_id"); model = cell.get("model")
        version = cell.get("guideline_version", "v1")
        if not (doc_id and model):
            out.append({"cell": cell, "error": "missing doc_id or model"}); continue
        ppath = PARSED / f"{doc_id}.json"
        if not ppath.exists():
            out.append({"cell": cell, "error": f"doc {doc_id} not found"}); continue
        gpath = PROMPTS / f"extract_{version}.md"
        if not gpath.exists():
            out.append({"cell": cell, "error": f"guideline {version} not found"}); continue
        try:
            job = rw.enqueue_reextract_with_version(
                version, models=[model], doc_paths=[str(ppath)],
                label=f"matrix cell ({model} × {doc_id} × {version})")
            out.append({"cell": cell, "job_id": job.job_id})
        except Exception as exc:
            out.append({"cell": cell, "error": f"{type(exc).__name__}: {exc}"})
    return jsonify({"jobs": out}), 202


@app.route("/api/run_matrix/status")
def api_run_matrix_status():
    out: dict[str, dict] = {}
    for sub in sorted(FACTS.iterdir()) if FACTS.exists() else []:
        if not sub.is_dir(): continue
        for fp in sub.glob("*.json"):
            try: d = json.loads(fp.read_text(encoding="utf-8"))
            except Exception: continue
            key = f"{d.get('doc_id', fp.stem)}|{d.get('annotator', sub.name)}"
            out[key] = {"doc_id": d.get("doc_id", fp.stem),
                        "model": d.get("annotator", sub.name),
                        "guideline_version": d.get("guideline_version", "v1"),
                        "fact_count": int(d.get("fact_count", 0))}
    return jsonify(out)


@app.route("/api/jobs/<job_id>")
def api_job_status(job_id: str):
    from src import reextract_worker as rw
    job = rw.get_job(job_id)
    if job is None:
        abort(404, f"job {job_id!r} not found")
    return jsonify(rw.to_dict(job))


@app.route("/api/jobs")
def api_jobs_list():
    from src import reextract_worker as rw
    return jsonify({"jobs": [rw.to_dict(j) for j in rw.list_jobs(limit=30)]})


@app.route("/api/jobs/<job_id>/cancel", methods=["POST"])
def api_job_cancel(job_id: str):
    from src import reextract_worker as rw
    if not rw.cancel(job_id):
        abort(400, "job not cancellable (already running or not found)")
    return jsonify({"ok": True, "job_id": job_id})


# ===================== analysis & review & experiment ============================


def _facts_per_annotator_from_disk(doc_id: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    if not FACTS.exists():
        return out
    for sub in sorted(FACTS.iterdir()):
        if not sub.is_dir():
            continue
        fp = sub / f"{doc_id}.json"
        if not fp.exists():
            continue
        try:
            doc = _load_json(fp)
        except Exception:
            continue
        out[doc.get("annotator") or sub.name] = doc.get("facts", [])
    return out


@app.route("/api/similarity_matrix/<doc_id>")
def api_similarity_matrix(doc_id: str):
    """Poster Fig. 2 (left): full cosine matrix between two annotators.

    Prefers the persisted Phase-2 conflicts file (so matched cells agree
    with the pipeline); falls back to raw fact files + on-the-fly Hungarian.
    """
    from src.annotator_compare import similarity_matrix
    cpath = _conflicts_path(doc_id)
    aligned_pairs = None
    if cpath.exists():
        confl = _load_json(cpath)
        fpa = confl.get("facts_per_annotator", {})
        aligned_pairs = confl.get("aligned_pairs", [])
    else:
        fpa = _facts_per_annotator_from_disk(doc_id)
    annotators = sorted(fpa.keys())
    if len(annotators) < 2:
        return jsonify({"doc_id": doc_id, "annotators": annotators,
                        "note": "need at least 2 annotators with facts on disk"})
    a = request.args.get("a") or annotators[0]
    b = request.args.get("b") or next(x for x in annotators if x != a)
    if a not in fpa or b not in fpa:
        abort(404, f"annotator not found; available: {annotators}")
    threshold = float(request.args.get("threshold", 0.78))
    out = similarity_matrix(fpa[a], fpa[b], annotator_a=a, annotator_b=b,
                            threshold=threshold, aligned_pairs=aligned_pairs)
    out["doc_id"] = doc_id
    out["annotators"] = annotators
    return jsonify(out)


@app.route("/api/iaa/<doc_id>")
def api_iaa(doc_id: str):
    """Poster-style IAA summary per annotator pair (from the conflicts file)."""
    from src.annotator_compare import iaa_summary
    cpath = _conflicts_path(doc_id)
    if not cpath.exists():
        return jsonify({"doc_id": doc_id, "annotators": [], "pairs": [],
                        "note": "no Phase-2 output for this doc yet"})
    out = iaa_summary(_load_json(cpath))
    out["doc_id"] = doc_id
    return jsonify(out)


# ---------------------- conflict review queue ----------------------

REVIEWS = DATA / "reviews"


@app.route("/api/guideline_rules")
def api_guideline_rules():
    from src.review_store import parse_guideline_rules
    version = request.args.get("version", "v1")
    if not _GUIDELINE_VERSION_RE.match(version):
        abort(400, "invalid version")
    return jsonify({"version": version,
                    "rules": parse_guideline_rules(version, PROMPTS)})


@app.route("/api/review/<doc_id>", methods=["GET", "POST"])
def api_review(doc_id: str):
    from src.review_store import load_reviews, save_review
    if request.method == "GET":
        return jsonify({"doc_id": doc_id, "reviews": load_reviews(REVIEWS, doc_id)})
    payload = request.get_json(silent=True) or {}
    pair_key = (payload.get("pair_key") or "").strip()
    if not pair_key:
        abort(400, "pair_key is required")
    try:
        out = save_review(REVIEWS, doc_id, pair_key, payload)
    except ValueError as exc:
        abort(400, str(exc))
    return jsonify({"ok": True, "doc_id": doc_id, **out})


@app.route("/api/review_summary")
def api_review_summary():
    from src.review_store import summarize_reviews
    version = request.args.get("version", "v1")
    if not _GUIDELINE_VERSION_RE.match(version):
        abort(400, "invalid version")
    return jsonify(summarize_reviews(REVIEWS, prompts_dir=PROMPTS,
                                     guideline_version=version))


# ---------------------- one-click experiment pipeline ----------------------


@app.route("/api/run_pipeline", methods=["POST"])
def api_run_pipeline():
    """Closed loop in one job: Phase-1 (versioned facts root) -> Phase-2
    (versioned conflicts file). Defaults to the docs that already have a
    baseline Phase-2 result so v1 vs v2 stays apples-to-apples."""
    from scripts.run_phase2 import discover_all_doc_ids
    payload = request.get_json(silent=True) or {}
    version = payload.get("guideline_version") or "v1"
    if not _GUIDELINE_VERSION_RE.match(version):
        abort(400, "invalid guideline_version")
    models = payload.get("models") or _of_record()["annotators"]
    doc_ids = payload.get("doc_ids")
    excluded: list[str] = []
    if doc_ids:
        _refuse_eval(doc_ids, "The experiment pipeline")
    else:
        doc_ids, excluded = _without_eval(
            sorted(_docs_with_conflicts()) or discover_all_doc_ids(FACTS))
    doc_ids = [d for d in doc_ids if (PARSED / f"{d}.json").exists()]
    doc_paths = [str(PARSED / f"{d}.json") for d in doc_ids]
    if not doc_paths:
        abort(400, "no parsed docs for the requested doc_ids")
    skip_extract = bool(payload.get("skip_extract", False))
    from src import reextract_worker as rw
    root = rw.facts_root_for_version(version)
    out_suffix = "" if version in ("", "v1") else version
    _refuse_frozen_phase2(version, payload, f"The experiment pipeline ({version})")
    _refuse_record_overwrite(
        payload, f"The experiment pipeline ({version})",
        facts_cells=[] if skip_extract else
                    [(root, m, d) for m in models for d in doc_ids],
        conflicts=[_conflicts_path_for(d, out_suffix) for d in doc_ids])
    phase2_params = {
        "skip_layer2": bool(payload.get("skip_layer2", False)),
        "align_threshold": float(payload.get("align_threshold", 0.78)),
        "redundancy_cosine": float(payload.get("redundancy_cosine", 0.95)),
        "merge_threshold": float(payload.get("merge_threshold", 0.78)),
        "layer2_model": payload.get("layer2_model") or _of_record()["layer2"],
        "layer2_url": payload.get("layer2_url", os.environ.get("OLLAMA_URL", "http://localhost:11434")),
        "skip_extract": skip_extract,
    }
    try:
        job = rw.enqueue_pipeline(
            version, models=models, doc_paths=doc_paths,
            phase2_params=phase2_params,
            label=f"experiment {version} ({len(models)}m × {len(doc_paths)}d → phase2)")
    except FileNotFoundError as exc:
        abort(400, str(exc))
    return jsonify({**rw.to_dict(job), "excluded_eval": excluded}), 202


@app.route("/api/distribution_shift_agg")
def api_distribution_shift_agg():
    """Corpus-level v1 vs v2 distribution shift: sums label counts and
    Layer-1 rates over every doc that has BOTH conflicts files."""
    from src.distribution_shift import (CONFLICT_LABELS, UNITS, UNRESOLVED_LABEL,
                                        _label_counts, _raw_counts,
                                        check_populations)
    v2 = request.args.get("v2")
    if not v2 or not _GUIDELINE_VERSION_RE.match(v2):
        abort(400, "?v2=<version> is required")
    unit = (request.args.get("unit") or "group").strip()
    if unit not in UNITS:
        abort(400, f"unit must be one of {UNITS}")
    totals = {"v1": {l: 0 for l in CONFLICT_LABELS},
              "v2": {l: 0 for l in CONFLICT_LABELS}}
    unresolved = {"v1": 0, "v2": 0}
    skipped_no_unit: list[str] = []
    layer = {"v1": {"matched": 0, "layer2_calls": 0},
             "v2": {"matched": 0, "layer2_calls": 0}}
    ann_sets = {"v1": set(), "v2": set()}
    docs = []
    # RQ1's segment population (method.tex sec:missing-segments): both sides of
    # a comparison are computed over ONE set of segments, recorded by Phase-2 in
    # `params.segment_population` (None = all segments). The per-document diff
    # refuses mixed populations; this sum used to add them up regardless, so an
    # unrestricted v1 file could be set against a restricted v2 one and the
    # difference in which segments were counted would read as a shift
    # (BUG-2026-09-23-J). The per-pair rule is
    # `distribution_shift.check_populations` (one owner): unrestricted pairs
    # compare (the sensitivity figure), a restricted pair only on an equal
    # hash. On top of it, one aggregate is over one population.
    population_mismatch: list[str] = []
    populations: dict = {}
    for p1 in sorted(CONFLICTS.glob("*.json")):
        if "__" in p1.stem:
            continue
        p2 = CONFLICTS / f"{p1.stem}__{v2}.json"
        if not p2.exists():
            continue
        try:
            d1, d2 = _load_json(p1), _load_json(p2)
        except Exception:
            continue
        pop = check_populations(d1, d2, v2_label=v2)
        if not pop["match"]:
            population_mismatch.append(p1.stem)
            continue
        populations.setdefault(pop["hash"]["v1"] if pop["restricted"] else None,
                               []).append(p1.stem)
        # A doc missing counts for the requested unit is SKIPPED and named, not
        # summed as zeros — silently averaging in a zero row is how a corpus
        # aggregate ends up reporting a shift that is really missing data.
        if not (_raw_counts(d1, unit)[1] and _raw_counts(d2, unit)[1]):
            skipped_no_unit.append(p1.stem)
            continue
        docs.append(p1.stem)
        ann_sets["v1"].update(d1.get("annotators", []))
        ann_sets["v2"].update(d2.get("annotators", []))
        for key, d in (("v1", d1), ("v2", d2)):
            counts = _label_counts(d, unit)
            for l in CONFLICT_LABELS:
                totals[key][l] += int(counts.get(l, 0))
            unresolved[key] += int(_raw_counts(d, unit)[0].get(UNRESOLVED_LABEL, 0))
            layer[key]["matched"] += sum(
                1 for pr in d.get("aligned_pairs", []) if pr.get("status") == "matched")
            layer[key]["layer2_calls"] += int(d.get("layer2_calls", 0))
    if population_mismatch or len(populations) > 1:
        # refused, not summed: `note` is what the comparison panel prints in
        # place of bars
        msg = ("segment populations differ, so these files do not add up to one "
               "RQ1 comparison (method.tex sec:missing-segments): "
               + (f"{len(population_mismatch)} document(s) have a v1 and a "
                  f"'{v2}' file computed over different segments "
                  f"({', '.join(population_mismatch[:5])}"
                  f"{' …' if len(population_mismatch) > 5 else ''}); "
                  if population_mismatch else "")
               + (f"the documents span {len(populations)} populations; "
                  if len(populations) > 1 else "")
               + "re-run Phase-2 on both versions with the same "
                 "--segment-population, or on neither for the sensitivity check")
        return jsonify({"unit": unit, "v1_label": "v1", "v2_label": v2,
                        "available": False, "error": msg, "note": msg,
                        "population_mismatch": population_mismatch,
                        "populations": {str(k): len(v) for k, v in populations.items()}})
    only_pop = next(iter(populations), None)
    labels = list(CONFLICT_LABELS)
    v1_counts = [totals["v1"][l] for l in labels]
    v2_counts = [totals["v2"][l] for l in labels]
    def _rate(side):
        m = layer[side]["matched"]
        return round(1 - layer[side]["layer2_calls"] / m, 3) if m else None
    out = {
        "unit": unit,
        "v1_label": "v1", "v2_label": v2, "docs": docs, "n_docs": len(docs),
        "labels": labels, "v1": v1_counts, "v2": v2_counts,
        "delta": [b - a for a, b in zip(v1_counts, v2_counts)],
        "delta_pct": [round((b - a) / a * 100, 1) if a else None
                      for a, b in zip(v1_counts, v2_counts)],
        "totals": {
            "v1": sum(v1_counts), "v2": sum(v2_counts),
            "delta": sum(v2_counts) - sum(v1_counts),
            "delta_pct": round((sum(v2_counts) - sum(v1_counts)) / sum(v1_counts) * 100, 1)
                         if sum(v1_counts) else None,
        },
        "layer1_rate": {"v1": _rate("v1"), "v2": _rate("v2")},
        "unresolved": {"label": UNRESOLVED_LABEL, **unresolved,
                       "warning": ("Layer-2 left items unresolved; the "
                                   "distribution is over the resolved remainder"
                                   if any(unresolved.values()) else None)},
        "skipped_no_unit_counts": skipped_no_unit,
        "v1_annotators": sorted(ann_sets["v1"]),
        "v2_annotators": sorted(ann_sets["v2"]),
        "annotators_match": ann_sets["v1"] == ann_sets["v2"],
        "population": {
            "restricted": only_pop is not None, "hash": only_pop,
            "note": (None if only_pop is not None else
                     "all segments (unrestricted): the sensitivity figure of "
                     "sec:missing-segments, not the primary RQ1 figure")},
    }
    if skipped_no_unit:
        out["note"] = (f"{len(skipped_no_unit)} doc(s) have no {unit}-level counts "
                       f"and were EXCLUDED, not zero-filled: "
                       f"{', '.join(skipped_no_unit[:5])}"
                       f"{'…' if len(skipped_no_unit) > 5 else ''}. Re-run Phase-2 "
                       f"on them, or pass ?unit=pair.")
    if not docs:
        out["note"] = (f"no doc has both a baseline and a '{v2}' conflicts file yet — "
                       f"run the experiment pipeline first")
    return jsonify(out)


# --- Neo4j custom conflict detection ("graph patching") --------------------


@app.route("/api/neo4j/status")
def api_neo4j_status():
    from src import neo4j_constraints as nc
    return jsonify(nc.status())


@app.route("/api/neo4j/constraints")
def api_neo4j_constraints():
    from src import neo4j_constraints as nc
    return jsonify({"constraints": nc.BUILTIN_CONSTRAINTS})


@app.route("/api/neo4j/export/<doc_id>", methods=["POST"])
def api_neo4j_export(doc_id: str):
    """Push the current doc's reified KG into Neo4j (wipe+reload for this doc)."""
    from src import neo4j_constraints as nc
    from src.neo4j_export import build_batches, export
    cpath = _conflicts_path(doc_id)
    if not cpath.exists():
        abort(404, f"no conflicts doc for {doc_id!r} (run Phase-2 first)")
    if not nc.driver_available():
        abort(400, "neo4j driver not installed (pip install neo4j)")
    doc = _load_json(cpath)
    batches = build_batches(doc, _variant_arg() or None)
    uri, user, pw = nc._conn()
    try:
        summary = export(batches, uri, user, pw, wipe=True)
    except Exception as exc:
        abort(502, f"neo4j export failed: {type(exc).__name__}: {exc}")
    return jsonify({"ok": True, "doc_id": doc_id, "summary": summary})


@app.route("/api/neo4j/constraint", methods=["POST"])
def api_neo4j_constraint():
    """Run a custom Cypher constraint; return violation rows (or materialize)."""
    from src import neo4j_constraints as nc
    payload = request.get_json(silent=True) or {}
    cypher = (payload.get("cypher") or "").strip()
    params = payload.get("params") or {}
    mode = payload.get("mode") or "report"
    if not cypher:
        abort(400, "cypher required")
    ok, reason = nc.validate_cypher(cypher, mode)
    if not ok:
        abort(400, reason)
    if not nc.driver_available():
        abort(400, "neo4j driver not installed (pip install neo4j)")
    try:
        result = nc.run_constraint(cypher, params, mode=mode)
    except ValueError as exc:
        abort(400, str(exc))
    except Exception as exc:
        abort(502, f"neo4j query failed: {type(exc).__name__}: {exc}")
    return jsonify(result)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
