// ── Guideline-refinement exploration queries (read-only) ─────────────────────

// Conditional facts (poster's If/Or concept) — flagged at import.
MATCH (f:Fact {conditional: true})
RETURN f.doc_id, f.version, count(*) AS n_conditional;

// FOL / "complex" facts a single (s,p,o) triple cannot represent — a direct
// guideline gap: these are candidates for a decomposition rule in v2.
MATCH (f:Fact {complex: true})
RETURN f.annotator, f.subject_surface, f.predicate, f.object_surface,
       f.natural_language
ORDER BY f.annotator;

// Negated facts — does the guideline tell annotators how to encode negation?
MATCH (f:Fact {negated: true})
RETURN f.annotator, f.predicate, f.natural_language;

// Per-version conflict-label distribution (v1 vs v2 distribution shift).
MATCH (f:Fact)
RETURN f.version, f.conflict_label, count(*) AS n
ORDER BY f.version, n DESC;

// Entities with the most surface-form variants — entity-resolution hotspots
// the guideline may need to address (canonical naming rule).
MATCH (e:Entity)
RETURN e.label, e.n_surface, e.surface_forms
ORDER BY e.n_surface DESC LIMIT 20;
