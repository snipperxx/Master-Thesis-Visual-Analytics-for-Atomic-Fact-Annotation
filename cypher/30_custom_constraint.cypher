// ── Custom conflict-detection templates (run from the UI "Neo4j" tab) ────────
// Read-only checks RETURN their violations; the UI also offers a write variant
// that MERGEs :CONFLICTS_WITH so the new conflict type shows up in the KG.
// These mirror src/neo4j_constraints.BUILTIN_CONSTRAINTS.

// Conditional-scope conflict: same resolved S-P-O, different condition.
MATCH (fa:Fact)-[:SUBJECT]->(s:Entity)<-[:SUBJECT]-(fb:Fact),
      (fa)-[:OBJECT]->(o:Entity)<-[:OBJECT]-(fb)
WHERE fa.id < fb.id AND fa.annotator <> fb.annotator
  AND fa.version = fb.version
  AND fa.predicate_norm = fb.predicate_norm
  AND coalesce(fa.condition,'') <> coalesce(fb.condition,'')
RETURN s.label AS subject, fa.predicate AS predicate, o.label AS object,
       fa.condition AS condition_a, fb.condition AS condition_b,
       fa.annotator AS annotator_a, fb.annotator AS annotator_b
ORDER BY subject LIMIT 200;
