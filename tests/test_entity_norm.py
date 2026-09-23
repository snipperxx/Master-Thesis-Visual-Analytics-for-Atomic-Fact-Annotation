"""Tests for core-entity normalization (poster-style merge keys)."""

from scripts.run_phase2 import cluster_entities, normalize_entity


def test_strips_determiners_and_date_tails():
    assert normalize_entity("the Council") == "Council"
    assert normalize_entity("Council Decision 2005/136/EC of 7 June 2005") == \
        "Council Decision 2005/136/EC"
    # The country qualifier survives (semantically load-bearing in a
    # multi-country corpus); only the year tail is stripped.
    assert normalize_entity("general government deficit of the Netherlands in 2004") == \
        "general government deficit of the Netherlands"


def test_truncates_long_clause_objects_at_first_preposition():
    out = normalize_entity(
        "a further decline in the general government deficit to 2,0 % of GDP")
    assert out == "further decline"          # determiner stripped, clause cut


def test_short_or_empty_inputs_survive():
    assert normalize_entity("deficit") == "deficit"
    assert normalize_entity("") == ""
    assert normalize_entity("the") == "the"          # never collapse to empty


def _facts(*pairs):
    return [{"fact_id": f"f{i}", "subject": s, "predicate": "p", "object": o,
             "natural_language": f"{s} p {o}"}
            for i, (s, o) in enumerate(pairs)]


def test_core_mode_merges_modified_variants():
    fpa = {"m1": _facts(("the deficit of the Netherlands in 2004", "x")),
           "m2": _facts(("the deficit of the Netherlands in 2005", "y"))}
    plain = cluster_entities(fpa, merge_threshold=0.95, core_entities=False)
    core = cluster_entities(fpa, merge_threshold=0.95, core_entities=True)
    s1 = "the deficit of the Netherlands in 2004"
    s2 = "the deficit of the Netherlands in 2005"
    # Core mode must merge the two year-variants; plain high-threshold may not.
    assert core[s1] == core[s2]
    assert len(set(core.values())) <= len(set(plain.values()))
