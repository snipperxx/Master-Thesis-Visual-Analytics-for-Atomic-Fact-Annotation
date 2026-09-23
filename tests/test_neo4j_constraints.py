"""Tests for src.neo4j_constraints — Cypher gatekeeper (no DB needed)."""
from src.neo4j_constraints import validate_cypher, BUILTIN_CONSTRAINTS


def test_report_mode_blocks_writes():
    ok, _ = validate_cypher("MATCH (f:Fact) MERGE (f)-[:X]->(f) RETURN f", "report")
    assert not ok


def test_report_mode_allows_readonly():
    ok, reason = validate_cypher("MATCH (f:Fact) RETURN f LIMIT 5", "report")
    assert ok, reason


def test_destructive_always_blocked_even_in_materialize():
    for q in ["MATCH (n) DETACH DELETE n",
              "MATCH (f:Fact) REMOVE f.x RETURN f",
              "DROP CONSTRAINT fact_id",
              "CALL db.labels()"]:
        ok, _ = validate_cypher(q, "materialize")
        assert not ok, q


def test_materialize_allows_merge():
    ok, reason = validate_cypher(
        "MATCH (a:Fact),(b:Fact) MERGE (a)-[:CONFLICTS_WITH]->(b) RETURN count(*)",
        "materialize")
    assert ok, reason


def test_builtins_are_self_consistent():
    for name, c in BUILTIN_CONSTRAINTS.items():
        ok, reason = validate_cypher(c["cypher"], c["mode"])
        assert ok, f"{name}: {reason}"
