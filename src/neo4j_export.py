"""
Phase-3 — Neo4j exporter (statement-reification / conflict-aware KG).

Exports triples (S,P,O) into Neo4j with provenance metadata; see
thesis/chapters/system.tex §Graph-Constraint Patching. Reads a Phase-2 conflicts document
(``data/conflicts/<doc_id>[__<variant>].json``) — the SAME input
``kg_build.py`` consumes — and loads it into Neo4j using statement
reification:

    (:Document)        one per doc_id
    (:Entity)          one per entity cluster (reuses entity_clusters mapping)
    (:Annotator)       one per annotator
    (:Span)            one per source_locator  (the "alignment flag" target)
    (:Fact)            one per atomic fact      (the reified statement node)

    (f:Fact)-[:SUBJECT]->(:Entity)
    (f:Fact)-[:OBJECT]->(:Entity)
    (f:Fact)-[:ASSERTED_BY]->(:Annotator)
    (f:Fact)-[:FROM_SPAN]->(:Span)
    (f:Fact)-[:IN_DOC]->(:Document)
    (fa:Fact)-[:ALIGNED_WITH {label, cosine, ...}]->(fb:Fact)   # Phase-2 pairs

Why reify (Fact as a NODE, not a bare (s)-[:PRED]->(o) edge)?
  * Multiple annotators, conflict labels, source spans and guideline versions
    all attach to the *statement* — an edge cannot carry that cleanly.
  * Complex logic a flat triple drops lives as PROPERTIES on the Fact node:
    qualifiers (temporal/locative tails), conditional, negated, modality.
    Genuine first-order logic (if/or/quantifiers) is FLAGGED (complex=true)
    rather than forced — that flag is itself a guideline-refinement signal.
  * Surface-form variants collapse to one :Entity via the precomputed
    entity_clusters mapping (Phase-2 SBERT/TF-IDF clustering), so custom
    Cypher rules operate over resolved entities, not raw strings.

:ALIGNED_WITH = the pipeline's own Phase-2 output (provenance). Conflict edges
materialized by *rules* live under :CONFLICTS_WITH (see ``cypher/``), so a
rule can be validated against the pipeline labels.

Connection via env vars (CLI flags override):
    NEO4J_URI       (default bolt://localhost:7687)
    NEO4J_USER      (default neo4j)
    NEO4J_PASSWORD  (default neo4j)

``--dry-run`` builds the batches and writes them to
``data/graphs/<doc>__<version>__neo4j.json`` without touching a database.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

try:
    from src.entity_norm import split_entity
except ImportError:  # running as a plain script from inside src/
    from entity_norm import split_entity  # type: ignore


# --- deterministic, no-model signals (kept consistent with kg_build) --------

_COND_RE = re.compile(
    r"\b(if|when|unless|provided that|subject to|in the event|conditional upon)\b",
    re.IGNORECASE)
_NEG_RE = re.compile(
    r"\b(not|no longer|never|without|fails? to|shall not|may not|cannot)\b",
    re.IGNORECASE)
_MODAL_RE = re.compile(
    r"\b(alleged(?:ly)?|reportedly|according to|claims? that|is said to)\b",
    re.IGNORECASE)
# Propositional / first-order logic that a single (s,p,o) triple cannot hold.
_COMPLEX_RE = re.compile(
    r"\b(either|or else|whether|all of|each of|none of|any of|"
    r"if and only if|implies)\b",
    re.IGNORECASE)


def _norm_predicate(pred: str) -> str:
    return " ".join((pred or "").lower().split())


def _fact_text(f: dict) -> str:
    loc = f.get("source_locator") or {}
    quote = loc.get("quote") or loc.get("linearized_sentence") or ""
    return f"{quote} {f.get('natural_language', '') or ''}"


def _span_id(doc_id: str, loc: dict) -> str:
    key = json.dumps(loc, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha1(f"{doc_id}|{key}".encode("utf-8")).hexdigest()[:16]
    return f"sp_{h}"


# --- core build -------------------------------------------------------------

def build_batches(conflicts_doc: dict, version: str | None = None) -> dict:
    """Translate a conflicts doc into flat row batches for UNWIND/MERGE."""
    doc_id = conflicts_doc["doc_id"]
    entity_clusters: dict = conflicts_doc.get("entity_clusters", {})
    facts_per_ann: dict = conflicts_doc.get("facts_per_annotator", {})
    aligned_pairs: list = conflicts_doc.get("aligned_pairs", [])

    if version is None:
        for facts in facts_per_ann.values():
            if facts:
                version = facts[0].get("guideline_version")
                break
        version = version or "v1"

    def ent_id(cluster: str) -> str:
        return f"{doc_id}__{version}__{cluster}"

    cluster_surfaces: dict = defaultdict(set)
    for surface, cid in entity_clusters.items():
        cluster_surfaces[cid].add(surface)

    def cluster_of(surface: str):
        s = (surface or "").strip()
        if not s or s.lower() == "null":
            return None
        return entity_clusters.get(s)

    annotators_set: set = set()
    referenced: set = set()
    spans: dict = {}
    facts: list = []
    subject_edges: list = []
    object_edges: list = []
    span_edges: list = []
    fact_index: dict = {}

    for annotator, ann_facts in facts_per_ann.items():
        annotators_set.add(annotator)
        for f in ann_facts:
            fid = f["fact_id"]
            fact_index[fid] = f
            subj = (f.get("subject") or "").strip()
            obj = (f.get("object") or "").strip()
            text = _fact_text(f)
            cond_field = (f.get("condition") or "").strip()
            temporal_field = (f.get("temporal_context") or "").strip()
            quals: set = set()
            for k in ("subject", "object"):
                for q in split_entity(f.get(k) or "")[1]:
                    if q:
                        quals.add(q[:60])
            facts.append({
                "id": fid,
                "doc_id": doc_id,
                "version": version,
                "annotator": annotator,
                "predicate": f.get("predicate", ""),
                "predicate_norm": _norm_predicate(f.get("predicate", "")),
                "subject_surface": subj,
                "object_surface": obj,
                "natural_language": f.get("natural_language"),
                "conflict_label": f.get("conflict_label", "unlabeled"),
                "qualifiers": sorted(quals),
                "condition": cond_field,
                "temporal_context": temporal_field,
                "conditional": bool(cond_field) or bool(_COND_RE.search(text)),
                "negated": bool(_NEG_RE.search(text)),
                "modality": "alleged" if _MODAL_RE.search(text) else "asserted",
                "complex": bool(_COMPLEX_RE.search(text)),
            })

            s_c = cluster_of(subj)
            if s_c:
                referenced.add(s_c)
                subject_edges.append({"fact_id": fid, "entity_id": ent_id(s_c)})
            o_c = cluster_of(obj)
            if o_c:
                referenced.add(o_c)
                object_edges.append({"fact_id": fid, "entity_id": ent_id(o_c)})

            loc = f.get("source_locator") or {}
            if loc:
                sid = _span_id(doc_id, loc)
                if sid not in spans:
                    spans[sid] = {
                        "id": sid,
                        "doc_id": doc_id,
                        "source_type": loc.get("source_type"),
                        "section_path": loc.get("section_path"),
                        "char_start": loc.get("char_start"),
                        "char_end": loc.get("char_end"),
                        "quote": loc.get("quote") or loc.get("linearized_sentence"),
                        "annex_id": loc.get("annex_id"),
                        "table_id": loc.get("table_id"),
                        "row_idx": loc.get("row_idx"),
                    }
                span_edges.append({"fact_id": fid, "span_id": sid})

    entities = []
    for cid in sorted(referenced):
        surfaces = sorted(cluster_surfaces.get(cid, {cid}))
        label = min(surfaces, key=lambda s: (len(s), s)) if surfaces else cid
        entities.append({
            "id": ent_id(cid),
            "cluster_id": cid,
            "doc_id": doc_id,
            "version": version,
            "label": label,
            "surface_forms": surfaces,
            "n_surface": len(surfaces),
        })

    aligned_edges = []
    for p in aligned_pairs:
        fa, fb = p.get("fact_a_id"), p.get("fact_b_id")
        if not fa or not fb or fa not in fact_index or fb not in fact_index:
            continue
        aligned_edges.append({
            "fact_a_id": fa,
            "fact_b_id": fb,
            "label": p.get("conflict_label", "unlabeled"),
            "cosine": p.get("cosine"),
            "status": p.get("status"),
            "annotator_a": p.get("annotator_a"),
            "annotator_b": p.get("annotator_b"),
            "layer1_reason": p.get("layer1_reason"),
            "layer2_reason": p.get("layer2_reason"),
        })

    return {
        "meta": {"doc_id": doc_id, "version": version},
        "documents": [{"id": doc_id, "version": version}],
        "annotators": [{"id": a} for a in sorted(annotators_set)],
        "entities": entities,
        "spans": list(spans.values()),
        "facts": facts,
        "subject_edges": subject_edges,
        "object_edges": object_edges,
        "span_edges": span_edges,
        "aligned_edges": aligned_edges,
    }


# --- Cypher (parameterized, batched via UNWIND) -----------------------------

CONSTRAINTS = [
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT fact_id IF NOT EXISTS FOR (f:Fact) REQUIRE f.id IS UNIQUE",
    "CREATE CONSTRAINT annotator_id IF NOT EXISTS FOR (a:Annotator) REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT span_id IF NOT EXISTS FOR (s:Span) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
]

LOADERS = [
    ("documents",
     "UNWIND $rows AS r MERGE (d:Document {id: r.id}) SET d.version = r.version"),
    ("annotators",
     "UNWIND $rows AS r MERGE (:Annotator {id: r.id})"),
    ("entities",
     "UNWIND $rows AS r MERGE (e:Entity {id: r.id}) SET e += r"),
    ("spans",
     "UNWIND $rows AS r MERGE (s:Span {id: r.id}) SET s += r"),
    ("facts",
     "UNWIND $rows AS r "
     "MERGE (f:Fact {id: r.id}) SET f += r "
     "WITH f, r MATCH (d:Document {id: r.doc_id}) MERGE (f)-[:IN_DOC]->(d) "
     "WITH f, r MATCH (a:Annotator {id: r.annotator}) MERGE (f)-[:ASSERTED_BY]->(a)"),
    ("subject_edges",
     "UNWIND $rows AS r MATCH (f:Fact {id: r.fact_id}), (e:Entity {id: r.entity_id}) "
     "MERGE (f)-[:SUBJECT]->(e)"),
    ("object_edges",
     "UNWIND $rows AS r MATCH (f:Fact {id: r.fact_id}), (e:Entity {id: r.entity_id}) "
     "MERGE (f)-[:OBJECT]->(e)"),
    ("span_edges",
     "UNWIND $rows AS r MATCH (f:Fact {id: r.fact_id}), (s:Span {id: r.span_id}) "
     "MERGE (f)-[:FROM_SPAN]->(s)"),
    ("aligned_edges",
     "UNWIND $rows AS r MATCH (a:Fact {id: r.fact_a_id}), (b:Fact {id: r.fact_b_id}) "
     "MERGE (a)-[w:ALIGNED_WITH]->(b) "
     "SET w.label = r.label, w.cosine = r.cosine, w.status = r.status, "
     "w.annotator_a = r.annotator_a, w.annotator_b = r.annotator_b, "
     "w.layer1_reason = r.layer1_reason, w.layer2_reason = r.layer2_reason"),
]


def export(batches: dict, uri: str, user: str, password: str,
           wipe: bool = False, batch_size: int = 1000) -> dict:
    from neo4j import GraphDatabase  # lazy: not needed for --dry-run
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            if wipe:
                session.run("MATCH (n) WHERE n.doc_id = $d DETACH DELETE n",
                            d=batches["meta"]["doc_id"])
            for c in CONSTRAINTS:
                session.run(c)
            for key, cypher in LOADERS:
                rows = batches.get(key, [])
                for i in range(0, len(rows), batch_size):
                    session.run(cypher, rows=rows[i:i + batch_size])
    finally:
        driver.close()
    return {k: len(v) for k, v in batches.items() if isinstance(v, list)}


# --- CLI --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path,
                    default=Path("data/conflicts/train-000000.json"))
    ap.add_argument("--version", default=None,
                    help="guideline version tag; default: inferred from facts")
    ap.add_argument("--uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    ap.add_argument("--user", default=os.environ.get("NEO4J_USER", "neo4j"))
    ap.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD", "neo4j"))
    ap.add_argument("--wipe", action="store_true",
                    help="DETACH DELETE existing nodes for this doc_id first")
    ap.add_argument("--dry-run", action="store_true",
                    help="build batches and write JSON; no DB connection")
    args = ap.parse_args()

    doc = json.loads(args.inp.read_text(encoding="utf-8"))
    batches = build_batches(doc, args.version)
    counts = {k: len(v) for k, v in batches.items() if isinstance(v, list)}

    if args.dry_run:
        meta = batches["meta"]
        out = Path("data/graphs") / f"{meta['doc_id']}__{meta['version']}__neo4j.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(batches, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"[neo4j_export][dry-run] {counts}")
        print(f"[neo4j_export][dry-run] wrote {out}")
        return

    summary = export(batches, args.uri, args.user, args.password, wipe=args.wipe)
    print(f"[neo4j_export] loaded into {args.uri}: {summary}")


logger = logging.getLogger(__name__)


def autosync(conflicts_doc: dict, version: str | None = None) -> bool:
    """Best-effort push to Neo4j when NEO4J_AUTOSYNC is truthy.

    Hooked into the Phase-2 / pipeline write path so the graph stays in sync
    without a manual export. Never raises — a Neo4j outage must not break the
    pipeline. Returns True iff a sync was attempted and succeeded.
    """
    if os.environ.get("NEO4J_AUTOSYNC", "").strip().lower() not in ("1", "true", "yes", "on"):
        return False
    try:
        batches = build_batches(conflicts_doc, version)
        summary = export(
            batches,
            os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            os.environ.get("NEO4J_USER", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "neo4j"),
            wipe=True,
        )
        print(f"[neo4j autosync] synced {batches['meta']} {summary}")
        return True
    except Exception as exc:  # best effort: a Neo4j outage must not break the pipeline
        logger.warning("[neo4j autosync] skipped (%s): %s", type(exc).__name__, exc)
        print(f"[neo4j autosync] skipped: {exc}")
        return False


if __name__ == "__main__":
    main()
