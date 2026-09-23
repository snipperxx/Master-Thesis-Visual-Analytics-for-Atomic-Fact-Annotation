"""
Phase-3 — custom conflict-detection ("graph patching") over the reified Neo4j KG.

The analyst supplies a Cypher pattern; matching rows come back as the
constraint's *violations*, and (in `materialize` mode) the rule MERGEs a
`:CONFLICTS_WITH {rule}` edge that the KG surfaces as CONTRADICTION — the
proposal's "Interactive Graph Constraints" item. This is what lets a domain
expert add a NEW conflict type without touching the Python pipeline: a
guideline rule becomes a graph pattern.

Safety: read-only by default. `mode="materialize"` permits MERGE/SET so the
R1–R3 rules can write `:CONFLICTS_WITH`; DELETE / DETACH / REMOVE / DROP /
`CALL db.*` / LOAD CSV are *always* rejected so a stray query can't wipe the
graph. Connection comes from NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD.
"""
from __future__ import annotations

import os
import re

# Built-in, guideline-relevant constraint templates. `$name` placeholders are
# bound as Neo4j parameters from the request's `params` (never string-interpolated).
BUILTIN_CONSTRAINTS: dict = {
    "conditional_scope": {
        "label": "Conditional-scope conflict (same S-P-O, different condition)",
        "description": "Two annotators assert the SAME subject/predicate/object but "
                       "attach DIFFERENT conditions. The flat graph collapses these to "
                       "one edge and calls it redundancy; reified, they are a genuine "
                       "scope disagreement — a direct signal the guideline must treat "
                       "the condition field as load-bearing.",
        "mode": "report",
        "params": {},
        "cypher": (
            "MATCH (fa:Fact)-[:SUBJECT]->(s:Entity)<-[:SUBJECT]-(fb:Fact),\n"
            "      (fa)-[:OBJECT]->(o:Entity)<-[:OBJECT]-(fb)\n"
            "WHERE fa.id < fb.id AND fa.annotator <> fb.annotator\n"
            "  AND fa.version = fb.version\n"
            "  AND fa.predicate_norm = fb.predicate_norm\n"
            "  AND coalesce(fa.condition,'') <> coalesce(fb.condition,'')\n"
            "RETURN s.label AS subject, fa.predicate AS predicate, o.label AS object,\n"
            "       fa.condition AS condition_a, fb.condition AS condition_b,\n"
            "       fa.annotator AS annotator_a, fb.annotator AS annotator_b\n"
            "ORDER BY subject LIMIT 200"
        ),
    },
    "cardinality": {
        "label": "Cardinality — a functional predicate with >1 distinct object",
        "description": "A predicate you declare functional (subject -> at most one "
                       "object) but which the KG shows pointing at several distinct "
                       "object-entities. Edit the $pred parameter.",
        "mode": "report",
        "params": {"pred": "decided"},
        "cypher": (
            "MATCH (f:Fact {predicate_norm: $pred})-[:SUBJECT]->(s:Entity)\n"
            "MATCH (f)-[:OBJECT]->(o:Entity)\n"
            "WITH s, collect(DISTINCT o.label) AS objects, collect(DISTINCT f.id) AS facts\n"
            "WHERE size(objects) > 1\n"
            "RETURN s.label AS subject, size(objects) AS distinct_objects, objects, facts\n"
            "ORDER BY distinct_objects DESC LIMIT 200"
        ),
    },
    "predicate_mismatch": {
        "label": "Predicate mismatch (same S and O, different predicate)",
        "description": "Two annotators link the same subject- and object-entity with "
                       "different predicates — a contradiction candidate; signals the "
                       "predicate vocabulary is under-constrained in the guideline.",
        "mode": "report",
        "params": {},
        "cypher": (
            "MATCH (fa:Fact)-[:SUBJECT]->(s:Entity)<-[:SUBJECT]-(fb:Fact),\n"
            "      (fa)-[:OBJECT]->(o:Entity)<-[:OBJECT]-(fb)\n"
            "WHERE fa.id < fb.id AND fa.annotator <> fb.annotator\n"
            "  AND fa.version = fb.version AND fa.predicate_norm <> fb.predicate_norm\n"
            "RETURN s.label AS subject, o.label AS object,\n"
            "       fa.predicate AS predicate_a, fb.predicate AS predicate_b,\n"
            "       fa.annotator AS annotator_a, fb.annotator AS annotator_b\n"
            "ORDER BY subject LIMIT 200"
        ),
    },
    "materialize_predicate_mismatch": {
        "label": "MATERIALIZE predicate-mismatch as :CONFLICTS_WITH",
        "description": "Write mode: MERGE a :CONFLICTS_WITH{rule:'predicate_mismatch'} "
                       "edge for each predicate mismatch so it shows in the KG. "
                       "Requires mode=materialize.",
        "mode": "materialize",
        "params": {},
        "cypher": (
            "MATCH (fa:Fact)-[:SUBJECT]->(s:Entity)<-[:SUBJECT]-(fb:Fact),\n"
            "      (fa)-[:OBJECT]->(o:Entity)<-[:OBJECT]-(fb)\n"
            "WHERE fa.id < fb.id AND fa.annotator <> fb.annotator\n"
            "  AND fa.version = fb.version AND fa.predicate_norm <> fb.predicate_norm\n"
            "MERGE (fa)-[c:CONFLICTS_WITH {rule:'predicate_mismatch'}]->(fb)\n"
            "SET c.type = 'predicate_mismatch'\n"
            "RETURN count(c) AS materialized"
        ),
    },
    "custom": {
        "label": "Custom Cypher (read-only)",
        "description": "Write your own MATCH ... RETURN pattern over (:Fact),(:Entity),"
                       "(:Annotator),(:Span). Read-only.",
        "mode": "report",
        "params": {},
        "cypher": "MATCH (f:Fact)\nRETURN f.annotator, f.predicate, f.natural_language\nLIMIT 50",
    },
}

# Always-forbidden, regardless of mode.
_DESTRUCTIVE_RE = re.compile(
    r"\b(DELETE|DETACH|REMOVE|DROP|FOREACH|LOAD\s+CSV)\b|\bCALL\s+(db|dbms|apoc\.periodic)\.",
    re.IGNORECASE)
# Write clauses — only allowed in materialize mode.
_WRITE_RE = re.compile(r"\b(MERGE|CREATE|SET)\b", re.IGNORECASE)


def validate_cypher(cypher: str, mode: str = "report") -> tuple[bool, str]:
    """Gatekeeper. Returns (ok, reason). Pure — unit-testable without a DB."""
    if not (cypher or "").strip():
        return False, "empty cypher"
    if _DESTRUCTIVE_RE.search(cypher):
        return False, "destructive clause (DELETE/DETACH/REMOVE/DROP/CALL db.*/LOAD CSV) is never allowed"
    if mode == "report" and _WRITE_RE.search(cypher):
        return False, "report mode is read-only; use mode=materialize to MERGE :CONFLICTS_WITH edges"
    if mode not in ("report", "materialize"):
        return False, f"unknown mode {mode!r}"
    return True, ""


def driver_available() -> bool:
    try:
        import neo4j  # noqa: F401
        return True
    except Exception:
        return False


def _conn() -> tuple[str, str, str]:
    return (os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            os.environ.get("NEO4J_USER", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "neo4j"))


# --- Driver factory -------------------------------------------------------
# Bounded timeouts so a missing / unreachable Neo4j FAILS FAST instead of
# hanging the request thread on the neo4j-driver defaults (connection 30 s,
# acquisition 60 s, retry 30 s). status() runs when the Neo4j tab opens; with
# no cap a blackholed Bolt host froze that tab (and, on the dev server, starved
# the /api/jobs poll behind it). Values sit above the ~10 s cold-connect we
# measured on a healthy DB, so a live database is never mis-reported as down.
# Keep EVERY driver creation in this module going through here.
# See doc/bugfix_log.md BUG-2026-07-06-A.
_DRIVER_TIMEOUTS = dict(
    connection_timeout=20.0,
    connection_acquisition_timeout=20.0,
    max_transaction_retry_time=15.0,
)


def _driver():
    """Create a neo4j driver with bounded timeouts (lazy import kept).

    Falls back to driver defaults if a driver build rejects a timeout
    config key, so hardening can never break a working connection.
    """
    from neo4j import GraphDatabase
    uri, user, pw = _conn()
    try:
        return GraphDatabase.driver(uri, auth=(user, pw), **_DRIVER_TIMEOUTS)
    except Exception:  # unknown/renamed config key on this driver version
        return GraphDatabase.driver(uri, auth=(user, pw))


def status() -> dict:
    uri = _conn()[0]
    out = {"driver_installed": driver_available(), "uri": uri, "connected": False}
    if not out["driver_installed"]:
        out["message"] = "neo4j driver not installed — pip install neo4j"
        return out
    try:
        drv = _driver()
        drv.verify_connectivity()
        with drv.session() as s:
            out["n_facts"] = s.run("MATCH (f:Fact) RETURN count(f) AS n").single()["n"]
            out["n_conflicts_with"] = s.run(
                "MATCH ()-[c:CONFLICTS_WITH]->() RETURN count(c) AS n").single()["n"]
        drv.close()
        out["connected"] = True
    except Exception as exc:
        out["message"] = f"{type(exc).__name__}: {exc}"
    return out


def run_constraint(cypher: str, params: dict | None = None, *,
                   mode: str = "report", limit: int = 200) -> dict:
    ok, reason = validate_cypher(cypher, mode)
    if not ok:
        raise ValueError(reason)
    drv = _driver()
    try:
        with drv.session() as s:
            result = s.run(cypher, **(params or {}))
            records = list(result)
            columns = list(result.keys())
            rows = [dict(r) for r in records][:limit]
    finally:
        drv.close()
    return {"columns": columns, "rows": rows, "n": len(rows), "mode": mode}
