"""Machine pre-review of the gold facts, to triage the human adjudication.

This is NOT adjudication and it deliberately does not write `status`. The gold
was pre-annotated by claude-opus-5; the same model marking its own output
`accepted` would make "machine pre-annotated, human-adjudicated" a false
description of the artefact and would turn RQ2 into a model scoring a model,
which is the circularity the design exists to avoid. The human decision stays
the human's.

What this does instead is everything that does not require being the annotator:
run the checks that are mechanically decidable, flag the patterns that are worth
a second look, and order the queue so the suspicious facts come first. The point
is to turn "read 307 facts cold" into "check ~N flagged facts, then confirm the
rest".

**A flag the decision log has already settled is not a flag.** Re-emitting one
costs the reviewer the same reading twice and teaches him to ignore the badge,
which is worse than not flagging at all. Every check removed for that reason is
listed in `SETTLED` below with the entry that settled it, and it is written into
the output file — so a later session can see the check was retired on purpose
rather than lost.

**Chains.** Since the high-resolution pass (D-18) a fact holds N triples, not
one. Every check walks the whole chain; a check that looked at `triples[0]` only
would report the first link of a four-link chain and stay silent about the rest,
and — worse — call two facts duplicates because their first links agree.

Severities
    error  near-certainly wrong: a schema or protocol rule is broken
    warn   worth a look: a pattern that is usually but not always a defect
    info   context the reviewer should have, no action implied

Output
    data/gold/machine_review.json              for data/gold/facts
    data/gold/machine_review_<root>.json       for any other --facts-root

Usage
    python -m scripts.machine_review_gold
    python -m scripts.machine_review_gold --facts-root data/gold/facts_hires --write
    python -m scripts.machine_review_gold --write --report   # + a readable digest
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.gold_store import GOLD, fold, is_verbatim, sample_index, segment_text

#: Checks this module used to emit and no longer does, with the decision-log
#: entry that settled them. Kept in code, and copied into the output file, so
#: that "why is this not flagged any more?" has an answer that is not "someone
#: deleted it". Do not re-add one of these without overturning its entry first.
SETTLED = {
    "boilerplate_not_flagged": (
        "D-13 — entry-into-force, application-date and addressee clauses are NOT "
        "boilerplate; only the contentless closing formula of D-03 is. The check "
        "now looks for that formula alone (`closing_formula_not_boilerplate`)."),
    "condition_not_verbatim": (
        "D-14 — verbatimness is required of source_quote, not of the qualifier "
        "slots. Replaced by `qualifier_adds_content`, which is what D-14 actually "
        "asks: the slot may be reworded, it may not import absent content."),
    "temporal_context_not_verbatim": "D-14 — see condition_not_verbatim.",
    "quote_misses_object": (
        "D-09 — resolution inside the segment is reading, not inference, so a slot "
        "may name something the fact's own short quote does not repeat. Replaced by "
        "`slot_absent_from_segment`, which scopes the same question to the segment."),
    "object_over_nine_words": (
        "D-30 — a word count is not a test for an unsplit chain. On "
        "data/gold/facts_hires it fired on 7 of 556 triples, all legitimate: "
        "foreign institution names, CN-code lists, and the entry-into-force "
        "formula. The reduced-relative-clause test (CLAUSE in "
        "make_second_annotator_console.py) is what looks for the real defect. "
        "Retired from the second-annotator console 2026-09-07; never add it "
        "here."),
    "predicate_overloaded": (
        "R1 forbids an ENTITY in the predicate, not a long predicate: 'shall be "
        "binding in its entirety and directly applicable in' is 10 words and hides "
        "nothing. Replaced by `node_split_into_predicate` / "
        "`predicate_holds_a_noun_phrase`, which look for the entity."),
}

# Self-referential subjects violate the decontextualization requirement in
# method.tex "The fact schema": a fact whose subject is "this Regulation" cannot
# be checked without knowing which document it came from.
SELF_REF = re.compile(r"\bthis (regulation|decision|directive|convention|annex)\b", re.I)
BARE_PRONOUN = re.compile(r"^\s*(it|they|them|these|those|this|that|which|such)\s*$", re.I)

#: D-03's closing formula, and only that. See SETTLED["boilerplate_not_flagged"].
CLOSING_FORMULA = re.compile(
    r"binding in its entirety|directly applicable in all Member States", re.I)

#: An act citation in a slot. D-12 permits the act's own identity in the slots —
#: it is document metadata carried on every segment, not content imported from
#: another one — so its tokens are exempt from the segment-grounding check.
ACT_CITATION = re.compile(
    r"\b(Regulation|Directive|Decision|Convention|Protocol|Treaty)\b\s*"
    r"(\([A-Z]+\)\s*)?(No\s*)?[\d/]*", re.I)
ACT_HEADED = re.compile(
    r"^\s*(the\s+)?(Council|Commission|European Parliament and Council)?\s*"
    r"(Regulation|Directive|Decision|Convention|Protocol|Treaty)\b", re.I)

DATE = re.compile(
    r"\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{4}\b|\b(19|20)\d{2}\b")

STOP = set("""a an the of in on at to for by and or with from as is are was were be been
being that this these those it its their there which who whom whose not no such any all
each per shall may must will would can could should than then when where while under over
into upon within without pursuant respect regard case order both either neither also
following referred provided laid down concerned question relevant same other""".split())

#: Words of >=3 characters, plus numbers of any length. Numbers must survive
#: whole: an early version tokenised "3,2 %" into "3" and "2", dropped both as
#: too short, and then reported that the object "3,2 % of GDP" was absent from a
#: quote that contained it.
_TOKEN = re.compile(r"[a-z]+|\d+(?:[.,]\d+)*\s*%?")

#: Crudest possible stemmer, and deliberately so. D-09 permits inflecting a word
#: that is in the segment ("fixed" -> "the fixing of refunds"), so comparing
#: surface forms reports 30 defects that are not defects. Longest suffix first.
_SUFFIXES = ("ations", "ation", "ings", "ing", "ements", "ement", "ions", "ion",
             "ed", "es", "s")


def _stem(w: str) -> str:
    for suf in _SUFFIXES:
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def content_words(text: str) -> set[str]:
    out = set()
    for tok in _TOKEN.finditer(fold(text).lower()):
        w = re.sub(r"\s+", "", tok.group(0))
        if w and w[0].isdigit():
            out.add(w)
        elif len(w) >= 3 and w not in STOP:
            out.add(_stem(w))
    return out


def _norm(s: str) -> str:
    """Node identity for chain-connectivity and node-inside-predicate checks."""
    s = fold(s or "").lower().strip()
    for a in ("the ", "a ", "an "):
        if s.startswith(a):
            return s[len(a):]
    return s


def _chain(fact: dict) -> list[dict]:
    return fact.get("triples") or []


def _identity(fact: dict) -> tuple:
    """What makes two facts the same fact: the WHOLE chain plus the qualifiers.

    D-08: the SPO triple is not the identity of a fact — f36/f37 of
    train-000000 are character-identical in S/P/O and differ only in
    temporal_context, and merging them is the packaging decision the gold is
    forbidden to make. Since D-18 the chain matters too: two facts whose first
    triples agree can decompose completely differently after that.
    """
    return (tuple((t.get("subject", ""), t.get("predicate", ""), t.get("object", ""))
                  for t in _chain(fact)),
            (fact.get("condition") or "").strip(),
            (fact.get("temporal_context") or "").strip())


#: `<det> <noun phrase>` inside a predicate. R1 lets a relational noun stay as
#: glue ("the application of", "a decline in"), which in practice means the
#: det-noun sits at the very end followed by one preposition and nothing else.
#: Anything after that preposition is a second relation wearing a predicate.
_DET_NOUN = re.compile(r"\b(?:the|a|an)\s+[a-z][\w'-]*(?:[ -][a-z][\w'-]*)?", re.I)
#: ...and a compound preposition counts as glue too: "have the purpose of
#: protecting", "are incurred as a result of engaging" state a purpose and a
#: cause, not a hidden entity, so the gerund after the preposition is part of
#: the connective. "the quantities specified in" has a participle, not a gerund,
#: and stays flagged.
_GLUE_TAIL = re.compile(
    r"\s*(of|in|to|for|with|on|by|from|at|into|under|through|against|as)?"
    r"(\s+\w+ing)?\s*$", re.I)


def _predicate_noun_phrases(pred: str) -> list[tuple[str, bool, bool]]:
    """[(phrase, is_glue, is_longest)] for the determiner-headed phrases in a predicate.

    Both prefixes of a two-word phrase are returned, because the two readings need
    different ones. Matching against the segment's node inventory wants every
    prefix — "may extend the time-limits by" has to offer "the time-limits", or
    the entity it split stays invisible. The is-this-glue guess wants the longest,
    because "a further" is not a noun phrase and "a further decline in" is.
    """
    pred = " ".join(pred.split())
    out: list[tuple[str, bool, bool]] = []
    for m in _DET_NOUN.finditer(pred):
        words = m.group(0).split()
        # A trailing participle belongs to the relation, not to the noun phrase:
        # "the quantities specified in" is [the quantities] + [specified in], and
        # reading it as one phrase is what hid the entity.
        if len(words) > 2 and words[-1].endswith("ed"):
            words = words[:-1]
        for n in range(2, len(words) + 1):
            phrase = " ".join(words[:n])
            tail = pred[m.start() + len(phrase):]
            out.append((phrase, bool(_GLUE_TAIL.fullmatch(tail)), n == len(words)))
    return out


def check_fact(fact: dict, seg_text: str, seg_facts: list[dict],
               nodes: set[str]) -> list[tuple[str, str, str]]:
    """Return [(severity, code, detail)] for one fact.

    `nodes` is every subject/object used anywhere in the segment, normalised —
    the inventory a predicate must not be hiding one of.
    """
    out: list[tuple[str, str, str]] = []
    triples = _chain(fact)
    quote = fact.get("source_quote") or ""
    seg_words = content_words(seg_text)

    if not triples:
        out.append(("error", "no_triples", "the fact has no triple"))

    # --- mechanically decidable -------------------------------------------
    if not is_verbatim(quote, seg_text):
        out.append(("error", "quote_not_verbatim", quote[:60]))

    for i, tri in enumerate(triples, start=1):
        where = f"triple {i}/{len(triples)}"
        for name in ("subject", "predicate", "object"):
            val = (tri.get(name) or "").strip()
            if not val:
                out.append(("error", "empty_slot", f"{where}: {name} is empty"))
                continue
            if name == "predicate":
                continue
            if SELF_REF.search(val):
                # A formulaic closing clause ("This Regulation shall enter into
                # force ...") is self-referential by nature and D-03 already keeps
                # it out of scoring via the boilerplate flag, so it is noise here.
                # The same slot in a substantive fact is a real defect (D-12).
                sev = "info" if fact.get("boilerplate") else "error"
                out.append((sev, "self_referential_slot",
                            f"{where}: {name} = {val!r} — not decontextualized"
                            + (" (boilerplate; excluded from scoring)"
                               if fact.get("boilerplate") else "")))
            if BARE_PRONOUN.match(val):
                out.append(("error", "bare_pronoun_slot", f"{where}: {name} = {val!r}"))

    # R2: a chain is only a chain if it connects. `build_gold_hires` refuses to
    # write one that does not, but the gold can also be edited in review.html,
    # which has no such gate — so check it here too rather than trusting upstream.
    # `nolink` is the declaration that a span genuinely asserts two propositions
    # sharing no entity ("the information must be disregarded" + "the goods are
    # not of marketable quality"); an undeclared near-miss still fails.
    for i, tri in enumerate([] if fact.get("nolink") else triples[1:], start=2):
        earlier = {_norm(t.get("object", "")) for t in triples[:i - 1]}
        earlier |= {_norm(t.get("subject", "")) for t in triples[:i - 1]}
        sub = _norm(tri.get("subject", ""))
        if sub and not any(sub == e or sub in e for e in earlier):
            out.append(("error", "chain_does_not_connect",
                        f"triple {i}/{len(triples)}: subject {tri.get('subject')!r} "
                        f"repeats no earlier subject or object — the KG gets a "
                        f"second unconnected node"))

    group, op = fact.get("logic_group") or "", fact.get("logic_op") or ""
    if group and not op:
        out.append(("error", "group_without_op", group))
    if op and not group:
        out.append(("error", "op_without_group", op))
    if group:
        siblings = [f for f in seg_facts if (f.get("logic_group") or "") == group]
        if len(siblings) < 2:
            out.append(("error", "orphan_logic_group",
                        f"{group!r} has {len(siblings)} member — a group of one is "
                        f"meaningless"))

    mine = _identity(fact)
    for other in seg_facts:
        if other is fact:
            continue
        theirs = _identity(other)
        if mine == theirs:
            out.append(("error", "duplicate_fact",
                        f"identical chain and qualifiers to {other['gold_id']}"))
            break
        if mine[0] == theirs[0]:
            out.append(("info", "same_chain_different_qualifier",
                        f"shares its whole chain with {other['gold_id']}, differs in "
                        f"a qualifier (D-08); scoring must not silently merge them"))
            break

    # --- patterns worth a second look -------------------------------------
    # Slot content the SEGMENT does not carry. Scoped to the segment, not to the
    # fact's own quote: D-09 makes resolution inside the segment reading rather
    # than inference, so a chain link may name an act the quote does not repeat.
    # D-12 exempts the act's own citation, which is document metadata.
    for i, tri in enumerate(triples, start=1):
        for name in ("subject", "object"):
            val = (tri.get(name) or "").strip()
            if not val or ACT_HEADED.match(val):
                continue
            residue = ACT_CITATION.sub(" ", val)
            present = content_words(residue) & seg_words
            missing = sorted(content_words(residue) - seg_words)
            if missing:
                # An abstract node coined so the chain has something to hang on
                # ('the transposition obligation', 'the following provisions').
                # D-19 permits it; what the reviewer has to check is that the
                # label means one thing inside its segment, which no mechanical
                # check can decide — hence info, not warn.
                out.append(("info", "coined_node_label",
                            f"triple {i}/{len(triples)}: {name} = {val!r} — "
                            f"{', '.join(missing)} not in the segment"
                            + ("" if present else " (nothing in it is)")
                            + ". D-19: fine if the label names one thing here"))

    # R1: the predicate carries the relation only. Two readings, two severities.
    for i, tri in enumerate(triples, start=1):
        pred = (tri.get("predicate") or "").strip()
        phrases = _predicate_noun_phrases(pred)
        split = next((p for p, _, _ in phrases if _norm(p) in nodes), None)
        if split:
            out.append(("warn", "node_split_into_predicate",
                        f"triple {i}/{len(triples)}: predicate {pred!r} contains "
                        f"{split!r}, which this segment also uses as a node — the KG "
                        f"splits one entity in two (D-19 R1b)"))
        else:
            for phrase, is_glue, longest in phrases:
                if longest and not is_glue:
                    out.append(("info", "predicate_holds_a_noun_phrase",
                                f"triple {i}/{len(triples)}: predicate {pred!r} carries "
                                f"{phrase!r} in non-glue position — a relation may be "
                                f"hiding there (R1)"))

    # R4: a date that IS the object stays in the object and temporal_context stays
    # empty; a scoping date lives in temporal_context only. Never both.
    #
    # "Never both" is about ONE date in two slots, which is what the coarse f04 did
    # ("2,3 % of GDP in 2004" beside temporal_context "in 2004"). The check used to
    # fire whenever each slot merely held some date, without ever comparing them, and
    # said "carry the same date" about '22 November 1993' beside 'before 1 March
    # 2005'. It also fired on 'the 2004 budget', where the year is part of a name
    # rather than an asserted date. So: the dates have to match, and the object's
    # date has to be the value it asserts — the whole object, or a trailing
    # preposition phrase — not a modifier sitting in front of a head noun.
    temporal = (fact.get("temporal_context") or "").strip()
    temporal_dates = {m.group(0).lower() for m in DATE.finditer(temporal)}
    if temporal_dates:
        for i, tri in enumerate(triples, start=1):
            obj = (tri.get("object") or "").strip()
            if ACT_HEADED.match(obj):
                continue
            stripped = ACT_CITATION.sub(" ", obj)
            shared = {m.group(0).lower() for m in DATE.finditer(stripped)} & temporal_dates
            if not shared:
                continue
            if not any(re.search(r"(?:^|\b(?:of|on|in|from|to|by|until|before|after)\s+)"
                                 + re.escape(d) + r"\s*[.,;:]?\s*$", stripped, re.I)
                       for d in shared):
                continue                      # a year inside a name, not a date slot
            out.append(("warn", "date_in_object_and_temporal",
                        f"triple {i}/{len(triples)}: object {obj!r} and "
                        f"temporal_context {temporal!r} both carry "
                        f"{', '.join(sorted(shared))} (R4)"))

    # D-14: a qualifier slot may be reworded; it may not import absent content.
    for name in ("condition", "temporal_context"):
        val = (fact.get(name) or "").strip()
        if val:
            missing = sorted(content_words(val) - seg_words)
            if missing:
                out.append(("warn", "qualifier_adds_content",
                            f"{name} = {val!r} introduces {', '.join(missing)}, "
                            f"absent from the segment (D-14 permits rewording, not "
                            f"new content)"))

    # Quoting the whole segment only signals under-anchoring when there was
    # something to choose between. Many recitals are a single sentence, where
    # the whole segment IS the clause.
    n_sentences = len([s for s in re.split(r"(?<=[.;:])\s+", fold(seg_text)) if s.strip()])
    if (quote and n_sentences >= 2
            and len(fold(quote)) >= 0.9 * len(fold(seg_text))
            and len(fold(seg_text)) > 120):
        out.append(("warn", "quote_is_whole_segment",
                    f"{len(fold(quote))} of {len(fold(seg_text))} chars across "
                    f"{n_sentences} sentences — under-anchored"))

    # A short quote is not a defect: "Trifolium repens" is the whole of what the
    # fact asserts. Length alone carries no signal here, so it is not checked.

    if CLOSING_FORMULA.search(quote) and not fact.get("boilerplate"):
        out.append(("warn", "closing_formula_not_boilerplate", quote[:60]))
    if fact.get("boilerplate") and not CLOSING_FORMULA.search(quote):
        out.append(("info", "boilerplate_flag_unexpected", quote[:60]))

    first_subject = (triples[0].get("subject") or "").strip() if triples else ""
    for other in seg_facts:
        if other is fact:
            continue
        oq = other.get("source_quote") or ""
        ot = _chain(other)
        if (oq and quote and fold(quote) != fold(oq) and fold(quote) in fold(oq)
                and ot and (ot[0].get("subject") or "").strip() == first_subject):
            out.append(("info", "quote_nested_same_subject",
                        f"quote sits inside {other['gold_id']}'s quote, same subject"))
            break

    if fact.get("reviewer_note"):
        out.append(("info", "carries_annotator_note", fact["reviewer_note"][:70]))

    return out


def review_root(facts_root: Path) -> tuple[dict, Counter, Counter, dict, int]:
    idx = sample_index()
    review: dict[str, dict] = {}
    by_code: Counter = Counter()
    by_sev: Counter = Counter()
    per_doc: dict[str, Counter] = defaultdict(Counter)
    n_facts = 0

    for path in sorted(facts_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        doc_id = payload["doc_id"]
        for seg in payload["segments"]:
            meta = idx.get(seg["segment_id"])
            if meta is None:
                continue
            text = segment_text(doc_id, seg["section_path"],
                                meta["char_start"], meta["char_end"])
            # The node inventory is per SEGMENT and has to exist before any fact
            # is checked: "the time-limits" is only provably a split node because
            # a later fact predicates of it.
            nodes = {_norm(t.get(k, "")) for f in seg["facts"] for t in _chain(f)
                     for k in ("subject", "object")}
            nodes.discard("")
            for fact in seg["facts"]:
                n_facts += 1
                checks = check_fact(fact, text, seg["facts"], nodes)
                if not checks:
                    continue
                sev = ("error" if any(c[0] == "error" for c in checks)
                       else "warn" if any(c[0] == "warn" for c in checks) else "info")
                review[fact["gold_id"]] = {
                    "severity": sev,
                    "checks": [{"severity": s, "code": c, "detail": d} for s, c, d in checks],
                }
                by_sev[sev] += 1
                for s, c, _ in checks:
                    by_code[c] += 1
                    if s in ("error", "warn"):
                        per_doc[doc_id][s] += 1

    return review, by_code, by_sev, per_doc, n_facts


def out_path(facts_root: Path) -> Path:
    try:
        rel = facts_root.resolve().relative_to(GOLD.resolve()).as_posix()
    except ValueError:
        rel = facts_root.resolve().name
    return GOLD / ("machine_review.json" if rel == "facts"
                   else f"machine_review_{rel.replace('/', '_')}.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--facts-root", type=Path, default=GOLD / "facts",
                    help="directory of gold fact files to review "
                         "(default data/gold/facts)")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    review, by_code, by_sev, per_doc, n_facts = review_root(args.facts_root)

    print(f"{args.facts_root}: {n_facts} facts reviewed; {len(review)} carry at "
          f"least one flag")
    print(f"  errors: {by_sev['error']}   warnings: {by_sev['warn']}   "
          f"info only: {by_sev['info']}")
    print("\nby check:")
    for code, n in by_code.most_common():
        print(f"  {n:>4}  {code}")
    print("\nby document (error/warn only):")
    for doc_id in sorted(per_doc):
        c = per_doc[doc_id]
        print(f"  {doc_id:<14} {c['error']:>3} error  {c['warn']:>3} warn")
    print(f"\nnot checked, settled by the decision log: {', '.join(sorted(SETTLED))}")

    if args.report:
        print("\n" + "=" * 100)
        print("FLAGGED FACTS (errors first)")
        for want in ("error", "warn"):
            for gid, rec in review.items():
                if rec["severity"] != want:
                    continue
                print(f"\n[{want.upper()}] {gid}")
                for c in rec["checks"]:
                    if c["severity"] == "info":
                        continue
                    print(f"    {c['code']}: {c['detail']}")

    if not args.write:
        print(f"\nDry run. Re-run with --write to save {out_path(args.facts_root)}.")
        return 0

    dest = out_path(args.facts_root)
    dest.write_text(json.dumps({
        "schema_version": 2,
        "note": ("Machine pre-review, NOT adjudication. Produced by the same model that "
                 "produced the pre-annotation, so it carries no authority over "
                 "correctness — it exists to order the human review queue. `status` is "
                 "untouched."),
        "facts_root": args.facts_root.resolve().name,
        "n_facts": n_facts,
        "n_flagged": len(review),
        "counts": {"error": by_sev["error"], "warn": by_sev["warn"], "info": by_sev["info"]},
        "by_check": dict(by_code.most_common()),
        "settled_not_checked": SETTLED,
        "facts": review,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
