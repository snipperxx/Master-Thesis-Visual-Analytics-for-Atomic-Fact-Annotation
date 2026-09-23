// ── Conflict-detection rules (a guideline rule == a graph pattern) ───────────
// Each rule MERGEs a :CONFLICTS_WITH edge tagged with `rule`. These are
// materialized independently of the pipeline's :ALIGNED_WITH labels so a rule
// can be validated against them (see R5). Run after src/neo4j_export.py.

// R1 — PREDICATE MISMATCH (contradiction candidate):
// two annotators link the SAME subject- and object-entity with DIFFERENT
// predicates. Signal: predicate vocabulary under-constrained in the guideline.
MATCH (fa:Fact)-[:SUBJECT]->(s:Entity)<-[:SUBJECT]-(fb:Fact),
      (fa)-[:OBJECT]->(o:Entity)<-[:OBJECT]-(fb)
WHERE fa.id < fb.id AND fa.annotator <> fb.annotator
  AND fa.version = fb.version
  AND fa.predicate_norm <> fb.predicate_norm
MERGE (fa)-[c:CONFLICTS_WITH {rule:'predicate_mismatch'}]->(fb)
SET c.type = 'predicate_mismatch';

// R2 — REDUNDANCY / CONSENSUS:
// identical resolved triple (subject-entity, predicate_norm, object-entity)
// asserted by 2+ annotators. Signal: agreement (or duplicated annotation).
MATCH (fa:Fact)-[:SUBJECT]->(s:Entity)<-[:SUBJECT]-(fb:Fact),
      (fa)-[:OBJECT]->(o:Entity)<-[:OBJECT]-(fb)
WHERE fa.id < fb.id AND fa.annotator <> fb.annotator
  AND fa.version = fb.version
  AND fa.predicate_norm = fb.predicate_norm
MERGE (fa)-[c:CONFLICTS_WITH {rule:'redundancy'}]->(fb)
SET c.type = 'redundancy';

// R3 — GRANULARITY:
// two annotators' facts cover overlapping source text (same section_path,
// overlapping char ranges) but resolve to different triples → one split where
// the other didn't. Signal: decomposition granularity under-specified.
MATCH (fa:Fact)-[:FROM_SPAN]->(sa:Span),
      (fb:Fact)-[:FROM_SPAN]->(sb:Span)
WHERE fa.id < fb.id AND fa.annotator <> fb.annotator
  AND fa.version = fb.version
  AND sa.section_path = sb.section_path
  AND sa.char_start <= sb.char_end AND sb.char_start <= sa.char_end
  AND (fa.predicate_norm <> fb.predicate_norm
       OR fa.subject_surface <> fb.subject_surface)
MERGE (fa)-[c:CONFLICTS_WITH {rule:'granularity'}]->(fb)
SET c.type = 'granularity';

// R4 — CARDINALITY CONSTRAINT (the "cardinality editor" roadmap item):
// a predicate declared functional (subject should have AT MOST ONE object)
// but found with several distinct object-entities. Read-only report.
//   :param pred => 'decided'
MATCH (f:Fact {predicate_norm: $pred})-[:SUBJECT]->(s:Entity)
MATCH (f)-[:OBJECT]->(o:Entity)
WITH s, collect(DISTINCT o.id) AS objs, collect(DISTINCT f.id) AS facts
WHERE size(objs) > 1
RETURN s.label AS subject, size(objs) AS distinct_objects, facts
ORDER BY distinct_objects DESC;

// R5 — VALIDATE a rule against the Phase-2 pipeline labels:
// how often does each rule's :CONFLICTS_WITH edge coincide with an
// :ALIGNED_WITH label, and which label?
MATCH (a:Fact)-[c:CONFLICTS_WITH]->(b:Fact)
OPTIONAL MATCH (a)-[w:ALIGNED_WITH]-(b)
RETURN c.rule AS rule, w.label AS pipeline_label, count(*) AS n
ORDER BY rule, n DESC;
