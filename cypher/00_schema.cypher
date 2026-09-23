// ── Constraints + indexes for the conflict-aware KG ──────────────────────────
// Also created automatically by src/neo4j_export.py; kept here so the schema
// can be (re)applied independently in Neo4j Browser / cypher-shell.
CREATE CONSTRAINT entity_id    IF NOT EXISTS FOR (e:Entity)    REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT fact_id      IF NOT EXISTS FOR (f:Fact)      REQUIRE f.id IS UNIQUE;
CREATE CONSTRAINT annotator_id IF NOT EXISTS FOR (a:Annotator) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT span_id      IF NOT EXISTS FOR (s:Span)      REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT document_id  IF NOT EXISTS FOR (d:Document)  REQUIRE d.id IS UNIQUE;
CREATE INDEX fact_predicate    IF NOT EXISTS FOR (f:Fact) ON (f.predicate_norm);
CREATE INDEX fact_doc_version  IF NOT EXISTS FOR (f:Fact) ON (f.doc_id, f.version);
