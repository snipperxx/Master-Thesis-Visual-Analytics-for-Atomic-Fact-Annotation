"""
Core-entity normalization shared by entity clustering (scripts/run_phase2)
and KG construction (src/kg_build).

split_entity() separates a (possibly heavily decontextualized) mention into
  * core  — the head noun phrase used as merge key and node identity
  * qualifiers — the load-bearing modifiers that were attached to it
    (date tails, clause tails cut at the first preposition)

Design rationale (Wikidata-style): a triple's temporal/locative/conditional
modifiers belong on the STATEMENT (KG: the edge), not inside the entity.
The annotation schema stays plain (S,P,O) — this is a display/identity-layer
decomposition, not a change to the annotation unit.
"""

from __future__ import annotations

import re

_DET_RE = re.compile(r"^(?:the|a|an|this|that|these|those|its|their)\s+", re.I)
_DATE_TAIL_RE = re.compile(
    r"\s*(?:of \d{1,2} \w+ \d{4}|in (?:19|20)\d{2}|on \d{1,2} \w+ \d{4}|\((?:19|20)\d{2}\))",
    re.I)
_PREPS = {"of", "in", "on", "to", "for", "by", "with", "under", "at", "from",
          "concerning", "regarding"}


def split_entity(s: str, max_tokens: int = 6) -> tuple[str, list[str]]:
    """Return (core, qualifiers). Deterministic, no model calls."""
    raw = (s or "").strip()
    if not raw:
        return "", []
    qualifiers: list[str] = []
    t = raw
    for m in _DATE_TAIL_RE.finditer(t):
        q = m.group(0).strip()
        if q:
            qualifiers.append(q)
    t = _DATE_TAIL_RE.sub("", t)
    t = _DET_RE.sub("", t)
    toks = t.split()
    if len(toks) > max_tokens:
        cut = None
        for i in range(2, len(toks)):
            if toks[i].lower() in _PREPS:
                cut = i
                break
        if cut is not None:
            tail = " ".join(toks[cut:]).strip(" ,;:.")
            if tail:
                qualifiers.append(tail)
            toks = toks[:cut]
        toks = toks[:max_tokens + 2]
    core = " ".join(toks).strip(" ,;:.")
    return (core or raw, qualifiers)


def normalize_entity(s: str, max_tokens: int = 6) -> str:
    """Core noun phrase only — the clustering merge key."""
    return split_entity(s, max_tokens)[0]
