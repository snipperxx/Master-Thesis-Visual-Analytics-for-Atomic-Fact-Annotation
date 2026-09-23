"""
The shared substrate under all three similarity measures.

Three questions are asked of the same corpus, and they legitimately need
different UNITS and different OUTPUTS:

    RQ1   model vs model   fact unit     -> conflict-type distribution
          (src/alignment.py + src/conflict_layer1.py, tau = 0.78)
    RQ2   model vs gold    triple unit   -> precision/recall/F1 + failed slot
          (src/gold_compare.py, slot gates 0.70/0.45/0.62)
    S4.5  gold A vs gold B fact unit     -> coverage + span containment
          (scripts/compare_annotators.py, threshold 0.60)

The unit has to differ. Correctness must be *packaging-invariant* — three facts
of one triple and one fact of three triples are the same triple set — or one
fixed gold could not score two guideline versions (CLAUDE.md non-negotiable 5).
Agreement must be *packaging-sensitive* — the packaging is the thing being
measured; granularity is read straight off the shape of a fact group
(`conflict_layer1.classify_group`). One unit cannot satisfy both.

What must NOT differ is everything *below* the unit: how a string is normalised
before it is embedded, and what counts as a number. Those had drifted into three
private copies, which is what this module ends.

Deliberately NOT unified
------------------------
`alignment` has no numeric short-circuit and must not grow one. gold_compare
asks "is this the same triple as the reference?", where a different quantity is
a different triple, so a numeric mismatch may veto the match. RQ1 has to LINK a
numerically divergent pair first so that `conflict_layer1` can then LABEL it a
contradiction; a similarity veto in `alignment` would turn every real
contradiction into two orphans and delete the category RQ1 exists to count.

Frozen-threshold discipline
---------------------------
CLAUDE.md non-negotiable 4: tau = 0.78 was swept on dev under a specific
representation, and "the text a fact is embedded as is part of this". Every
function here that can move a similarity value is therefore either
(a) byte-identical to the code it replaced, or (b) documented in
`doc/bugfix_log.md` with the measured delta.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# String normalisation
#
# Moved verbatim from gold_compare (2026-08-27) so alignment and gold scoring
# can share one definition. Behaviour is unchanged for every existing caller.
# ---------------------------------------------------------------------------

# Verbatim from gold_compare — leading predicate words stripped so modal /
# auxiliary / determiner variants align. Copied character for character on
# purpose: any addition or removal moves every RQ2 predicate similarity.
_PRED_STOP_LEAD = {
    "shall", "must", "may", "should", "will", "would", "can", "could",
    "is", "are", "be", "been", "being", "was", "were", "am",
    "has", "have", "had", "does", "do", "did", "hereby", "to", "the", "a", "an",
}
_ARTICLES = {"a", "an", "the"}
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def normalize_term(s: str) -> str:
    """Canonical form of a subject/object slot: lowercase, depunctuated, no
    leading article."""
    s = (s or "").strip().replace("_", " ").lower()
    s = re.sub(r"[\"'`.,;:()\[\]/]", " ", s)
    toks = [t for t in s.split() if t]
    while toks and toks[0] in _ARTICLES:
        toks.pop(0)
    return " ".join(toks)


def normalize_predicate(p: str) -> str:
    """Canonical form of a predicate slot: camelCase split, no leading
    auxiliary ('shall be applied' -> 'applied')."""
    p = _CAMEL_RE.sub(" ", p or "")            # cityServed -> city Served
    p = p.replace("_", " ").lower()
    p = re.sub(r"[^a-z0-9 ]", " ", p)
    toks = [t for t in p.split() if t]
    while len(toks) > 1 and toks[0] in _PRED_STOP_LEAD:
        toks.pop(0)
    return " ".join(toks)


def triple_key(s: str, p: str, o: str) -> tuple[str, str, str]:
    return (normalize_term(s), normalize_predicate(p), normalize_term(o))


def normalize_slot(slot: str, value: str) -> str:
    """Normalise by slot name. `predicate` takes the predicate rule, the rest
    take the term rule."""
    return normalize_predicate(value) if slot == "predicate" else normalize_term(value)


# ---------------------------------------------------------------------------
# Numbers
#
# Legal prose is saturated with reference numbers — "Regulation (EC) No
# 1260/2001", "Article 15(5)", "Decision 2005/136/EC", "2 June 2004". Read as
# quantities they made conflict_layer1's numeric rule fire ~77 % false on the
# EUR-Lex set, which is why the scrubber below was written for it. gold_compare
# never had it, and applied a raw numeric rule to 184 of the 1112 gold
# subject/object slots whose only digits are a citation or a date
# (BUG-2026-08-27-D).
#
# Trade-off, inherited unchanged from conflict_layer1: bare 4-digit years are
# scrubbed too, so a rare 4-digit *quantity* ("2000 tonnes") is not compared
# numerically. Accepted in this corpus, where 4-digit numbers are overwhelmingly
# years and regulation numbers.
# ---------------------------------------------------------------------------

_MONTHS = (r"january|february|march|april|may|june|july|august|"
           r"september|october|november|december")

_REFERENCE_PATTERNS = (
    # The bare slash citation runs FIRST. Behind the keyword pattern its lazy
    # \d+ stops at the first group, so "Decision 2005/136/EC" was scrubbed to
    # "/136/EC" and kept 136 as though it were a quantity — which left two
    # different Decisions numerically comparable. Ordered first, the whole
    # citation goes (BUG-2026-08-27-D).
    re.compile(r"\b\d{1,4}\s*/\s*\d{1,4}(?:\s*/\s*[a-z]{1,4})?\b", re.I),         # 584/75, 2005/136/EC
    re.compile(r"\b(?:no|nr)\.?\s*\d+(?:\s*/\s*\d+)*", re.I),                     # No 584/75
    re.compile(r"\(\s*e{1,2}[ceu]{1,2}\s*\)\s*(?:no\.?\s*)?\d+(?:/\d+)*", re.I),  # (EEC) No 404/93
    re.compile(r"\b(?:article|articles|paragraph|paragraphs|point|points|"
               r"annex|indent|subparagraph|regulation|directive|decision|"
               r"recommendation|chapter|title)\b[^.,;:]*?\d+"
               r"(?:\s*\(\s*\d+[a-z]?\s*\))?", re.I),                             # Article 15(5)
    re.compile(rf"\b\d{{1,2}}\s+(?:{_MONTHS})\s+\d{{4}}\b", re.I),                # 2 June 2004
    re.compile(rf"\b(?:{_MONTHS})\s+\d{{4}}\b", re.I),                            # June 2004
    re.compile(r"\b(?:19|20)\d{2}\b"),                                            # bare year
)

# EUR-Lex groups thousands with a space or a non-breaking space: "EUR 1 000",
# "1 000 000 tonnes". A tokeniser that reads digit runs separately turns
# "1 000 000" into {1, 0} and "2 000 000" into {2, 0} — which then share the
# member 0 and were scored IDENTICAL by gold_compare's any-shared-number rule
# (BUG-2026-08-27-C). Glue the groups back together before tokenising.
_GROUPED_THOUSANDS_RE = re.compile(r"(?<=\d)[   ,](?=\d{3}(?!\d))")

# EUR-Lex writes decimals with a comma ("37,5 %"). Applied AFTER the thousands
# rule above, so the comma in "1,000" is already gone and only a genuine decimal
# comma is left to convert.
_DECIMAL_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def scrub_reference_numbers(text: str) -> str:
    """Blank out citation, article and date numbers, leaving content quantities."""
    s = text or ""
    for pat in _REFERENCE_PATTERNS:
        s = pat.sub(" ", s)
    return s


def _tokenize_numbers(s: str) -> set[float]:
    """Numbers as floats, EUR-Lex's continental conventions handled.

    Order matters. A comma before exactly three digits groups thousands and is
    deleted; any comma left between digits is a decimal separator and becomes a
    point. So `1,000` -> 1000, `37,5` -> 37.5, `1 234,50` -> 1234.5.
    """
    t = _GROUPED_THOUSANDS_RE.sub("", s or "")
    t = _DECIMAL_COMMA_RE.sub(".", t)
    out: set[float] = set()
    for tok in _NUM_RE.findall(t):
        try:
            out.add(round(float(tok), 4))
        except ValueError:                                   # pragma: no cover
            continue
    return out


def all_numbers(s: str) -> set[float]:
    """Every number in the string, references and dates included."""
    return _tokenize_numbers(s)


def content_numbers(s: str) -> set[float]:
    """Genuine content quantities — amounts, percentages, prices.

    Citation, article and date numbers are scrubbed first, so
    `Decision 2005/136/EC` contributes nothing at all rather than {2005, 136}.
    """
    return _tokenize_numbers(scrub_reference_numbers(s or ""))


def is_bare_number(s: str) -> bool:
    """True when the slot is nothing but a number — `25.0`, `2 000`, `37,5`.

    The test is on the normalised token stream, so `25.0` (which normalises to
    `25 0`) counts and `25 metres` does not.
    """
    toks = normalize_term(s).split()
    return bool(toks) and all(_NUM_RE.fullmatch(t) for t in toks)


def numeric_override(a: str, b: str) -> float | None:
    """What the numbers alone settle about two slot strings. (RQ2.)

    Returns 0.0, 1.0, or None to mean "the embedding decides". The rule is that
    the override may only speak where the embedding is BLIND:

      * different numbers -> **0.0**. An embedding scores `610.0` against
        `4349.0` at ~0.9; magnitude is exactly what it cannot see, so the veto
        supplies it.
      * equal numbers, and at least one side is nothing but a number -> **1.0**.
        There is no other content for the embedding to judge, so the numeric
        equality is the whole truth. This is what lets a bare gold literal
        `25.0` match a predicted `25 metres`.
      * equal numbers, both sides carrying other content -> **None**. The
        embedding can see `per tonne` against `per hectare`, and `2 June`
        against `2 July`, so it should.

    Both halves of this were wrong before 2026-08-27 (BUG-2026-08-27-C). The
    test was set INTERSECTION, so any shared digit — a year, a magnitude, a `0`
    from a shattered thousands group — forced 1.0. Replacing it with equality
    fixed the citation cases but left the second fault standing: equal numbers
    still asserted identity from digits alone, so `ECU 20 per tonne` and
    `ECU 20 per hectare` were still a flat 1.0.
    """
    na, nb = all_numbers(a), all_numbers(b)
    if not na or not nb:
        return None
    if na != nb:
        return 0.0
    return 1.0 if (is_bare_number(a) or is_bare_number(b)) else None


def quantity_divergence(a: str, b: str) -> bool:
    """Do these two strings assert CONFLICTING quantities? (RQ1.)

    True only when both sides carry a content quantity, the sets differ, and
    neither contains the other — a one-sided extra number is a coverage
    difference, not a contradiction. Citations and dates are scrubbed first,
    because a regulation number is not a quantity and reading it as one made
    this rule fire ~77 % false on EUR-Lex.

    This is `conflict_layer1`'s rule, moved here unchanged. It asks the opposite
    question to `numeric_override` and the two must not be collapsed: RQ1 needs
    a numerically divergent pair to stay LINKED so it can be labelled a
    contradiction, so this returns a hint and never vetoes a similarity.
    """
    na, nb = content_numbers(a), content_numbers(b)
    return bool(na and nb and na != nb and not na <= nb and not nb <= na)


# ---------------------------------------------------------------------------
# What text represents a fact to the encoder
#
# DECISION 2026-08-09: the triple, `subject predicate object` — see the block
# comment in src/alignment.py for the dev evidence and the tau re-sweep.
#
# Two fact SHAPES reach this code and they are not interchangeable:
#
#   flat    {"subject": ..., "predicate": ..., "object": ...}   one triple
#           model output (src/extractor.py), everything in data/facts/
#   nested  {"triples": [{...}, {...}], "natural_language": ...}  N triples
#           gold: data/gold/facts*/, data/gold/annotator_b/
#
# Before 2026-08-27 only the flat shape was handled. A nested fact produced an
# empty triple string and fell through to `natural_language` in silence, so
# `fact_text_mode()` answered "triple" while the encoder was being fed
# sentences — every gold A/B similarity number was computed on a representation
# nobody had chosen (BUG-2026-08-27-B). The flattening below is now a declared
# policy, and `fact_text_report()` says what actually happened.
# ---------------------------------------------------------------------------

FACT_TEXT_MODES = ("triple", "natural_language")


def _flat_triple(fact: dict) -> str:
    return " ".join(str(fact.get(k) or "").strip()
                    for k in ("subject", "predicate", "object")).strip()


def _nested_triples(fact: dict) -> str:
    """A nested fact is its triples, in order, joined.

    Joining rather than picking one is what keeps the representation
    packaging-honest at the fact unit: a fact that externalises three relations
    is a longer object than one that externalises none, and that difference is
    exactly the granularity signal the fact-unit measures exist to see.
    """
    parts = []
    for t in fact.get("triples") or []:
        if not isinstance(t, dict):
            continue
        joined = " ".join(str(t.get(k) or "").strip()
                          for k in ("subject", "predicate", "object")).strip()
        if joined:
            parts.append(joined)
    return " ; ".join(parts)


def fact_shape(fact: dict) -> str:
    """"flat" | "nested" | "empty" — which triple carrier this fact uses."""
    if _flat_triple(fact):
        return "flat"
    if _nested_triples(fact):
        return "nested"
    return "empty"


def fact_text(fact: dict, mode: str = "triple") -> str:
    """The string this fact is embedded as, under `mode`.

    `triple` reads the flat slots, then the nested `triples` list. Only a fact
    with neither falls back to `natural_language`, and `fact_text_report`
    counts those so a caller can refuse to report a number computed on them.
    """
    if mode not in FACT_TEXT_MODES:
        raise ValueError(f"fact text mode must be one of {FACT_TEXT_MODES}, got {mode!r}")
    nl = (fact.get("natural_language") or "").strip()
    if mode == "natural_language":
        return nl or _flat_triple(fact) or _nested_triples(fact)
    return _flat_triple(fact) or _nested_triples(fact) or nl


def fact_text_report(facts: list[dict], mode: str = "triple") -> dict:
    """What `fact_text` actually did to a corpus — for provenance, not control.

    `fell_back` counts facts with no triple of either shape, which under `triple`
    mode were embedded as sentences. Any similarity number reported alongside a
    non-zero `fell_back` is a number computed on a mixed representation.
    """
    shapes = {"flat": 0, "nested": 0, "empty": 0}
    for f in facts or []:
        shapes[fact_shape(f)] += 1
    return {
        "mode": mode,
        "n": len(facts or []),
        "flat": shapes["flat"],
        "nested": shapes["nested"],
        "fell_back": shapes["empty"] if mode == "triple" else 0,
        "mixed": bool(shapes["flat"] and shapes["nested"]),
    }
