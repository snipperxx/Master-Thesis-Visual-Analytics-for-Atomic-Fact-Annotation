"""Build the console for the SECOND ANNOTATOR pass over the gold sample.

Replaces `make_gemini_console.py`, which served the first pass's facts and asked
a model to adjudicate them. That measures the wrong thing: agreement on a given
segmentation, not agreement on how many facts a segment holds — and granularity
is precisely what the two annotators must decide independently
(`doc/methodology_v2.tex` §4.5).

So this console is **blind by construction**: it embeds the segment text and
nothing else. No fact from `data/gold/` reaches the page, so nothing the first
pass decided can leak into the second annotation.

One segment at a time: copy the prompt, paste it into the model, paste the reply
back. The reply is validated here — JSON shape, every `source_quote` verbatim in
that segment, no empty slot, chains that connect — so a bad reply costs one
re-ask instead of being discovered after the whole queue is done.

Scope: **all 59 sampled segments**, not the 9 of the frozen IAA subsample
(changed 2026-08-18). The subsample existed to bound a human annotator's hours,
and the 2026-08-17 decision makes the second annotator a model, so that reason is
gone -- while what it cost was most of the measurement: those 9 hold 24 gold
facts, 15.3 % of the segments but 7.7 % of the annotation mass, because the draw
(`sample_gold_segments.py:155`) was uniform over segments and landed on the short
ones. The 9 stay marked -- `iaa: true` on the segment, a dotted chip in the queue,
`iaa` on every bundle entry -- so agreement can still be reported on the subsample
alone, for continuity with what the thesis text already says. `--iaa-only`
rebuilds the old 9-segment page.

The prompt states the high-resolution rules R1-R9 with worked before/after
examples, because the failure mode to avoid is the one the first machine pass fell
into: one triple per fact, propositions left inside a 17-word object or a 20-word
predicate.

Prompt v2 (2026-08-18) repairs four defects the recitals[4] reply exposed --
`PROMPT_VERSION` is stamped into the page, into every segment accepted from now
on, and into the bundle, so v1 and v2 annotations stay distinguishable:

* **R1 contradicted R1b.** R1 offered "projects a further decline in" as glue and
  illustrated it; R1b then forbids exactly that whenever the segment predicates of
  "the further decline", which the source of that very example does. R1 now says
  R1b wins, and the example teaches the interaction instead of walking into it.
* **R3 contradicted the coordination rule.** "Do not turn one span into several
  facts" against `logic_group`'s "one fact per branch": the two readings differ by
  a factor of three on a three-way coordination. R3 is now about a proposition and
  its chain, and coordination expansion is R7, a rule rather than a slot note.
* **Two rules the reviewer enforces were never stated.** `machine_review_gold`
  errors on a bare-pronoun slot and warns on an entity split across a predicate,
  yet nothing told the annotator to resolve "this threshold" or to stop
  alternating "the forecast" with the act's full name (R8), and nothing said what
  to do with "according to X" other than reify the proposition it attributes,
  which the KG cannot use (R9).
* **The examples were drawn from the sample.** R1's fat predicate and R2's
  of-chain were `train-000000::preamble.recitals[4]`'s own text and R5's was
  `train-000005::preamble.recitals[1]`'s -- 39 of 312 gold facts pre-solved in the
  instructions. Every example is now invented, and `screen_examples` fails the
  build if any 5-gram of the rule block reappears in a sampled segment.

v2.1 and v2.2 came out of the first reply written under v2, which the checks passed
at 0 errors while it turned two forecasts into statements of fact:

* **A hedge has to survive on every triple that carries the hedged content.**
  Reifying the projection ("the forecast projects the fall", "the fall is to 0,0 %
  of GDP") satisfies R1b and moves the hedge off the triple that holds the value;
  `(the government debt ratio | remains below | the reference value)` then asserts
  what the source only projects. Degree words go with it: drop "partly" from two
  branches of one coordination and each branch claims the whole thing (R5, v2.1).
* **The R2 label rule that came with v2.1 was too broad and is now measured.** It
  read "a label may not contain of / on / in", which condemns 24 labels the gold
  keeps -- "the finding of an excessive deficit", "the reporting by the
  Netherlands", "the sale of natural gas" each name one thing, and it makes no
  difference that the segment also predicates of the inner noun. What breaks the KG
  is a reduced relative clause left inside a label ("the Treaty establishing the
  European Community"), or an OJ footnote marker swallowed into a name. Against
  `facts_hires` those two tests flag nothing, the earlier one flagged 96 (v2.2).

The same measurement is why R8 is stated as a check and not an absolute: 12 gold
labels differ from a sibling by one modifier ("export refunds" beside "refunds"),
so the console warns and leaves the call to the annotator.

Provenance: whoever produces these replies is a **machine** annotator. The
agreement number it yields bounds how reproducibly the protocol can be applied;
it is not agreement between two independent human annotators, and the thesis has
to say which of the two it is reporting.

Opening it: `file://` works, but a browser may deny an opaque origin both
localStorage and `navigator.clipboard` (the latter needs a secure context). The
page degrades rather than breaking -- it keeps state in memory and copies through
`execCommand` -- and says in the header when progress is not being stored. To get
both back, serve it instead:

    python -m http.server 8765 --directory data/gold
    # then open http://localhost:8765/second_annotator.html

Usage:
    python scripts/make_second_annotator_console.py
    # open data/gold/second_annotator.html, work the queue, download the bundle
    python -m scripts.import_second_annotation data/gold/second_annotation_bundle.json \\
        --annotator gemini-3-pro
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import gold_store as gs  # noqa: E402

GOLD = ROOT / "data" / "gold"

#: Stamped into every prompt and into every accepted segment, so a fact
#: annotated before the guideline was repaired stays distinguishable from one
#: annotated after it (D-20, 2026-08-18).
#: v2.1 added R5's hedge-survival and degree-word rules, after a reply reified a
#: projection (which moved the hedge off every triple that carried the value) and
#: dropped "partly" from two branches of one coordination (which made them
#: contradict each other).
#: v2.2 narrowed the R2 label rule that came with v2.1. It had said a label may not
#: contain of / on / in at all, which condemned 24 gold labels — "the finding of an
#: excessive deficit", "the reporting by the Netherlands", "the sale of natural gas"
#: each name one thing. Measured against `facts_hires`, the surviving test (a reduced
#: relative clause or a trailing footnote marker inside a label) flags none of them.
PROMPT_VERSION = "second-annotator-prompt-v2.4 (2026-08-27)"

PROMPT = """\
You are annotating atomic facts in EU legislation as an INDEPENDENT SECOND
ANNOTATOR. You will not be shown anyone else's annotation. Work only from the
segment text at the bottom of this message. Output JSON only.

Guideline version: <<PROMPT_VERSION>>

# The task

List every atomic fact the segment states. A fact is one proposition a reader can
check against the text on its own, out of context.

Record WHAT THE TEXT SAYS at the finest granularity at which each fact stays
independently checkable, and stay silent on how it ought to be packaged. Never
bake a merging decision into the annotation.

Every example below is invented. None of it is drawn from the segment you are
given: do not look for these words in your segment, and do not carry these
entities into your answer.

# Resolution rules — this is the part that goes wrong

R1. THE PREDICATE CARRIES THE RELATION ONLY. It must contain no entity that
    could be predicated of in its own right, no quantity and no date. A
    relational noun may stay in the predicate as glue ("concerns the application
    of", "records a reduction of") — unless R1b applies, and R1b wins.

    WRONG  (the type-approval certificate | records a reduction of the emission
            limit to | 80 mg/km)
    RIGHT  (the type-approval certificate | records a reduction of |
            the emission limit)
           (the emission limit | is reduced to | 80 mg/km)

    R1b. IF A NOUN PHRASE INSIDE A PREDICATE IS SOMETHING YOU ALSO USE AS A
    SUBJECT OR AN OBJECT anywhere in this segment, take it out into its own
    triple. Glue that the segment predicates of is not glue. Otherwise one
    entity becomes two unconnected nodes.

    The same predicate as above, in a segment that goes on to say when the
    reduction takes effect:

    WRONG  (the type-approval certificate | records a reduction of |
            the emission limit)
           (the reduction | takes effect on | 1 July 2007)
    RIGHT  (the type-approval certificate | records | the reduction)
           (the reduction | concerns | the emission limit)
           (the reduction | takes effect on | 1 July 2007)

    R1 is about the entity, not about length: "shall be binding in its entirety
    and directly applicable in" is ten words and hides nothing.

R2. A RELATION STATED INSIDE A NOUN PHRASE BECOMES ITS OWN TRIPLE — of-chains,
    postposed prepositional phrases, reduced relative clauses ("annexed to ...",
    "drawn up by ...", "issued under ...", "provided for in ..."). Chain them:
    the object of one triple must be repeated VERBATIM as the subject of the
    next. A near-miss is a defect: it leaves two unconnected nodes.

    WRONG  (the aid application | is accompanied by | a copy of the inspection
            report drawn up by the veterinary service of the exporting country)
    RIGHT  (the aid application | is accompanied by | a copy)
           (a copy | is of | the inspection report)
           (the inspection report | was drawn up by | the veterinary service)
           (the veterinary service | is of | the exporting country)

    WHAT MUST COME OUT OF A LABEL IS A CLAUSE, NOT EVERY PREPOSITION. A reduced
    relative — "establishing ...", "annexed to ...", "issued under ...", "drawn up
    by ...", "setting out ..." — is a proposition and becomes its own triple. An
    of / for / by nominalisation that names ONE thing stays one node: "the register
    of authorised establishments", "the surrender of the licence", "the inspection
    by the veterinary service" are each a single entity, and it makes no difference
    that the segment also predicates of the inner noun.

    WRONG  (the register of authorised establishments | is accompanied by |
            the Annex setting out the model certificate (4))
    RIGHT  (the register of authorised establishments | is accompanied by |
            the Annex)
           (the Annex | sets out | the model certificate)

    A trailing "(4)" is an OJ footnote marker. It is never part of a name; leave it
    out of the label even though the quote keeps it.

R3. ONE FACT = ONE PROPOSITION AND ITS WHOLE CHAIN. Every triple that
    externalises a part of the same proposition belongs in that fact's `triples`
    array, never in a fact of its own. R7 is the one thing that does turn one
    span into several facts.

R4. QUANTITIES AND DATES GO WHERE THEY BELONG. A quantity is the object. A date
    that IS the object of the assertion ("is dated 3 March 1998") stays in the
    object and `temporal_context` is empty. A date that scopes the whole
    statement ("in 2004", "by 31 December 2006") goes in `temporal_context`
    ONLY. The SAME date must never sit in both. Two DIFFERENT dates in one fact
    are fine — an act's own date in an object, a scoping date in
    `temporal_context`.

R5. STOP HERE. Never split inside a named entity ("Commission Regulation (EEC)
    No 2454/93" is one node), a measurement ("4,5 % of turnover" is one value),
    or a code list. Never add a participant or an event the text does not state:
    if the text says "invites the Commission to set a reference price", do NOT
    assert that the price was set.

    A HEDGE HAS TO SURVIVE ON EVERY TRIPLE THAT CARRIES THE HEDGED CONTENT, not
    only on the one that names the projection. Naming the projection and then
    stating its value flatly turns a forecast into a fact:

    WRONG  (the spring report | projects | the reduction)
           (the reduction | is to | EUR 15 per tonne)
           (the levy | remains below | the ceiling)
    RIGHT  (the spring report | projects | the reduction)
           (the reduction | is projected to reach | EUR 15 per tonne)
           (the levy | is projected to remain below | the ceiling)

    A DEGREE WORD IS PART OF THE CLAIM. "partly", "largely", "mostly", "markedly"
    stay in the predicate. Drop "partly" from two branches of one coordination and
    each branch then claims the whole thing, so the two contradict each other.

R6. NAMING SOMETHING THE TEXT LEAVES AS A VERB. A chain sometimes needs a noun
    the segment does not contain: "the holder shall surrender the licence" gives
    you a node like "the surrender". That is allowed — a name for something the
    segment already asserts adds no new claim — under two conditions.

    (a) WITHIN A SEGMENT, SUCH A LABEL MUST NAME EXACTLY ONE THING. If two
        different requirements both end up called "the requirement" they merge
        in the KG and both are wrong; name them apart.

    (b) THE TRIPLE THAT INTRODUCES THE NOUN MUST USE THE NOUN. Coining it in a
        later subject, while the earlier triple left the verb as a participle,
        gives you two nodes and a chain that does not connect.

        WRONG  (an authorisation | is to be | withdrawn)
               (the withdrawal | is provided for in | Article 12(3))
        RIGHT  (Article 12(3) | provides for | the withdrawal)
               (the withdrawal | concerns | an authorisation)

    A PROPOSITION IS NOT A NODE. Never coin one: "the keeping of the stock below
    the ceiling" is a sentence wearing a noun's clothes and nothing in the KG can
    use it. R9 says what to do instead.

R7. A COORDINATION EXPANDS INTO FACTS — the only thing that turns one span into
    several. One fact per branch, each repeating the SAME `source_quote`, all
    sharing one `logic_group` string, with `logic_op` = AND | OR | XOR. This
    covers coordinated subjects ("A and B each reduced the arrears"), coordinated
    objects ("based on A, on B and on C"), coordinated obligations, and a
    coordination inside a qualifier ("in 2005 and 2006" — one fact per year).
    Expand ONE level only; never cross-multiply two coordinations. A
    `logic_group` with a single member is an error: either it has siblings or it
    does not belong.

R8. ONE ENTITY, ONE LABEL. Pick one string for an entity and use that string in
    every triple of the segment, character for character; where the segment
    offers a long and a short form, take the fullest one. Never alternate them:
    "the annual programme" and "the programme" are one entity and two KG nodes. If
    two labels really are two things, make the difference visible in both ("the
    2005 savings measures", "the 2006 savings measures"), never by a modifier that
    one of them silently drops. No slot may be a bare pronoun or an unresolved
    demonstrative either ("it", "this threshold", "that period", "such aid") —
    resolve it to the label you use elsewhere in the segment.

R9. WHO SAYS IT, AND UNDER WHAT AUTHORITY. "according to X", "in accordance with
    X", "pursuant to X", "as referred to in X" name a source or a legal basis.
    Do not reify the proposition to give them a subject (R6). Make the source or
    the basis the subject of its own triple, and chain it on the entity it speaks
    about.

    WRONG  (the keeping of the stock below the ceiling | is according to |
            the annual return)
    RIGHT  (the annual return | reports | the stock level)
           (the stock level | was kept below | the ceiling)

    WRONG  (the withdrawal | is in accordance with | Article 12(3))
    RIGHT  (Article 12(3) | requires the withdrawal of | an authorisation)
           (Article 12(3) | is part of | the basic Regulation)

R10. FLATTENED TABLE ROWS. A segment may contain a table that was turned into
    prose, one sentence per row. Such a sentence opens with a prefix naming the
    container and a row number, then gives that row's cells, separated by a
    vertical bar or written as `heading: value;` pairs. For example:

        In Annex II, Table 3, row 4: 'Ruritania | Seed Office, Vaduz | rye barley'

    The row is packaging, not content. Its number is an artefact of the
    flattening and names nothing in the law, so a row is never a node and its
    existence is never a fact. Annotate the CELLS: read them as the columns of
    one entry and state what that entry asserts, chained onto the entity the
    entry is about — normally whatever the first cell names.

    A cell may hold a coordination of its own ('rye barley'). Expand it under R7
    like any other, one fact per member.

    ONE ROW IS ONE ENTRY. When the flattening emits more than one row, those are
    different entries, not one entry mentioned again and again. R6(a) applies:
    label them apart, and never let a value from one row attach to the label of
    another. Two rows that each carry a code give you two distinct facts, not one
    fact asserted twice — if your answer states the same triple twice, you have
    merged rows that the source keeps apart.

    ANCHOR ON THE ROW. Take the whole flattened row as the `source_quote` of every
    fact you draw from that row. A lone cell value is not an anchor: a code or a
    single letter recurs down the column, so a quote that short no longer says
    which entry the fact came from.

    WRONG  (row 4 | is in | Table 3)
    WRONG  (row 4 | contains | the entry for Ruritania)
    RIGHT  (Ruritania | is listed in | Table 3 of Annex II)
           (the entry for Ruritania | names | the Seed Office, Vaduz)
           (the entry for Ruritania | covers | rye)
           (the entry for Ruritania | covers | barley)

    Where the flattening says outright that it could not recover the row
    geometry, do not invent one. Annotate only what each listed cell asserts on
    its own, and leave a relation the flattening does not state unstated.

# Slots

- `condition` — an if / when / where precondition, in the text's own words. Never
  duplicated into the object.
- `temporal_context` — ordering or deadline (after / before / within / by).
- `logic_group` + `logic_op` — see R7.
- `boilerplate` — true for a closing formula such as "This Regulation shall be
  binding in its entirety and directly applicable in all Member States". Still
  annotate it.

# source_quote

A VERBATIM contiguous substring of the segment text, copied character for
character — including typos, odd spacing and any replacement characters (�)
that appear. Do not normalise quotes or dashes, do not tidy anything. Every fact
needs one.

# Output

JSON only. No prose, no markdown fences, no commentary.

{"facts": [
  {"natural_language": "one self-contained sentence restating this fact",
   "source_quote": "verbatim substring of the segment text",
   "condition": "", "temporal_context": "",
   "logic_group": "", "logic_op": "", "boilerplate": false,
   "triples": [{"subject": "...", "predicate": "...", "object": "..."}]}
]}

Cover every clause of the segment. Return `{"facts": []}` only for a segment that
is purely a citation block or a bare header.

# Before you answer, check each fact against this list

- the `source_quote` is a character-for-character substring of the segment;
- after the first triple, every subject repeats an earlier subject or object of
  the same fact VERBATIM (R2, R6b);
- no subject or object still holds a reduced relative clause ("establishing ...",
  "annexed to ...") or a trailing footnote marker (R2);
- every projected value and state still says it is projected, and every degree
  word the text uses is still there (R5);
- no predicate hides a noun phrase you use as a node anywhere in this segment
  (R1b);
- one label per entity, no short forms, no bare pronouns or demonstratives (R8);
- no node is a whole proposition, and no source or legal basis was turned into
  one (R6, R9);
- every coordination is expanded and every `logic_group` has at least two
  members (R7);
- no coined label names two different things in this segment (R6a).
- no fact is about a row, a column or the table itself (R10).

# What you are not given

These rules and the segment text, and nothing else. You are not shown the first
annotator's facts, and you are not shown the project's log of ambiguous cases
already settled during the first pass. Both are withheld on purpose: where these
rules under-determine an answer, your divergence from the first annotation is a
finding about the rules, not a mistake to be avoided. Do not try to guess what
someone else decided.

# Segment to annotate

Document title: <<DOC_TITLE>>
Section path:   <<SECTION_PATH>>

Segment text (annotate exactly this, quote from exactly this):
\"\"\"
<<SEGMENT_TEXT>>
\"\"\"
"""


#: Phrases the rule block is allowed to share with a sampled segment. The closing
#: formula is the definition of `boilerplate` and the reviewer flags it
#: mechanically anyway (CLOSING_FORMULA in machine_review_gold), so quoting it
#: gives the second annotator no information it would not have had.
EXAMPLE_ALLOWLIST = (
    "this regulation shall be binding in its entirety and directly applicable in "
    "all member states",
)


def screen_examples(segment_texts: dict[str, str], n: int = 5) -> list[str]:
    """Report every n-gram the rules share verbatim with a sampled segment.

    The v1 prompt drew its R1 and R2 worked examples from
    `train-000000::preamble.recitals[4]` — the fat predicate and the 3605/93
    of-chain are that segment's own text (`data/gold/calibration/_RULES.md`, "The
    example this started from"). Every annotator therefore received two solved
    sub-cases of a segment still in the queue, worth 39 of the 312 gold facts, and
    agreement on it measured nothing. The build refuses to write a page that does
    that again.
    """
    rules = PROMPT.split("# Segment to annotate")[0]
    words = re.findall(r"[\w'’%,()/-]+", gs.fold(rules).lower())
    grams = {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}
    folded = {sid: gs.fold(t).lower() for sid, t in segment_texts.items()}
    hits = []
    for g in sorted(grams):
        if any(a in g or g in a for a in EXAMPLE_ALLOWLIST):
            continue
        where = [sid for sid, t in folded.items() if g in t]
        if where:
            hits.append(f"{g!r} appears in {', '.join(sorted(where)[:3])}"
                        + (f" (+{len(where) - 3} more)" if len(where) > 3 else ""))
    return hits


def build_payload(iaa_only: bool = False) -> dict:
    sample = gs.load_sample()
    idx = {s["segment_id"]: s for s in sample["segments"]}
    iaa = set(sample["iaa_subsample"]["segment_ids"])
    segs = []
    for sid in sorted(iaa if iaa_only else idx):
        m = idx[sid]
        text = gs.segment_text(m["doc_id"], m["section_path"],
                               m["char_start"], m["char_end"])
        segs.append({
            "segment_id": sid,
            "doc_id": m["doc_id"],
            "doc_title": m["doc_title"],
            "section_path": m["section_path"],
            "kind": m["kind"],
            "iaa": sid in iaa,
            "text": text,
            "prompt": (PROMPT.replace("<<PROMPT_VERSION>>", PROMPT_VERSION)
                             .replace("<<DOC_TITLE>>", m["doc_title"])
                             .replace("<<SECTION_PATH>>", m["section_path"])
                             .replace("<<SEGMENT_TEXT>>", text)),
        })
    segs.sort(key=lambda s: s["segment_id"])

    leaks = screen_examples({s["segment_id"]: s["text"] for s in segs})
    if leaks:
        raise SystemExit("the rule block quotes text from the sample it is about "
                         "to annotate — replace the example, do not ship this:\n  "
                         + "\n  ".join(leaks))

    return {"segments": segs,
            "scope": "iaa_subsample" if iaa_only else "full_sample",
            "prompt_version": PROMPT_VERSION,
            "share": sample["iaa_subsample"].get("share"),
            "n_iaa": sum(1 for s in segs if s["iaa"]),
            "n_segments": len(segs)}


HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>Second annotator &mdash; EUR-Lex gold sample</title>
<style>
:root{--bg:#faf9f7;--panel:#fff;--ink:#1c1b19;--mut:#6b6862;--line:#e3e0da;
      --ok:#1f7a4d;--warn:#9a6b00;--no:#a33;--acc:#2b5c9b}
*{box-sizing:border-box}
body{margin:0;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
     background:var(--bg);color:var(--ink)}
header{position:sticky;top:0;z-index:9;background:var(--panel);border-bottom:1px solid var(--line);
       padding:10px 18px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
h1{font-size:15px;margin:0;font-weight:650}
.bar{flex:1;min-width:160px;height:8px;background:var(--line);border-radius:4px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--ok);width:0;transition:width .2s}
button{font:inherit;padding:5px 11px;border:1px solid var(--line);background:#fff;
       border-radius:6px;cursor:pointer}
button:hover{border-color:var(--mut)}
button.primary{background:var(--acc);color:#fff;border-color:var(--acc)}
button.go{background:var(--ok);color:#fff;border-color:var(--ok)}
button:disabled{opacity:.45;cursor:not-allowed}
.wrap{display:grid;grid-template-columns:1fr 1fr;gap:18px;padding:18px;align-items:start}
@media(max-width:1000px){.wrap{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}
.src{position:sticky;top:64px;max-height:calc(100vh - 96px);overflow:auto}
.meta{font-size:12px;color:var(--mut);margin-bottom:8px}
.txt{white-space:pre-wrap;font-size:14.5px;line-height:1.7}
textarea{width:100%;min-height:230px;font:12.5px/1.5 ui-monospace,monospace;
         border:1px solid var(--line);border-radius:8px;padding:9px}
.hint{font-size:12px;color:var(--mut)}
.tag{font-size:11px;padding:2px 7px;border-radius:99px;background:#f0eee9;color:var(--mut)}
.tag.ok{background:#e7f4ec;color:var(--ok)}
.tag.err{background:#fdecec;color:#a5342c}
.tag.wrn{background:#fdf4e3;color:#8a6414}
.issues{margin-top:10px;font-size:13px;line-height:1.5}
.issues .e{color:#a5342c}
.issues .w{color:#8a6414}
.fact{border:1px solid var(--line);border-left:3px solid var(--ok);border-radius:7px;
      padding:8px 10px;margin-top:8px;font-size:13px}
.fact.bad{border-left-color:var(--no)}
.fact .nl{font-weight:550}
.fact .tr{font:12px ui-monospace,monospace;color:var(--mut);margin-top:3px}
.fact .q{font-size:12px;color:var(--mut);font-style:italic;margin-top:4px}
.queue{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px}
.queue span{font:11px ui-monospace,monospace;padding:2px 6px;border-radius:5px;
            border:1px solid var(--line);cursor:pointer}
.queue span.done{background:#e7f4ec;border-color:#bcdcc8}
.queue span.iaa::before{content:"\2022 ";color:var(--acc);font-weight:700}
.queue span.cur{border-color:var(--acc);box-shadow:0 0 0 2px rgba(43,92,155,.13)}
mark{background:#ffe9a8}
</style>
<header>
  <h1>Second annotator &mdash; blind</h1>
  <span id="pos" class="hint"></span>
  <button onclick="go(-1)">&larr;</button>
  <button onclick="go(1)">&rarr;</button>
  <div class="bar"><i id="prog"></i></div>
  <span id="cnt" class="hint"></span>
  <span id="gver" class="hint" title="the guideline version stamped into every segment you accept from now on"></span>
  <button class="primary" onclick="copyPrompt()">Copy prompt</button>
  <button onclick="save()">Download bundle</button>
  <button onclick="document.getElementById('imp').click()">Load bundle&hellip;</button>
  <input id="imp" type="file" accept="application/json,.json" style="display:none"
         onchange="loadFile(this)">
</header>
<div class="wrap">
  <div class="card src">
    <div class="meta" id="smeta"></div>
    <div class="txt" id="stext"></div>
    <div class="queue" id="queue"></div>
  </div>
  <div class="card">
    <div class="hint" id="step"></div>
    <textarea id="reply" placeholder="Paste the model's JSON reply here"
              oninput="stash()"></textarea>
    <div style="margin-top:9px;display:flex;gap:6px;align-items:center;flex-wrap:wrap">
      <button onclick="check()">Check reply</button>
      <button class="go" id="acceptbtn" onclick="accept()" disabled>Accept &amp; next</button>
      <button onclick="clearOne()">Clear this segment</button>
      <span id="verdict"></span>
    </div>
    <div class="issues" id="issues"></div>
    <div id="preview"></div>
  </div>
</div>
<p class="hint" style="padding:0 18px 30px">
  Blind by construction: this page holds the segment text and nothing else &mdash; no
  fact from the first pass is embedded in it. One segment at a time: <b>Copy
  prompt</b>, paste it into the model, paste the reply back, <b>Check reply</b>.
  A reply is accepted only when every quote is verbatim and no slot is empty; fix
  by re-asking the model, not by editing the reply here. The queue is the whole
  sample; the chips marked <b>&bull;</b> are the frozen IAA subsample, which stays
  reportable on its own. Progress autosaves in
  this browser &mdash; download the bundle when done, that is the only copy that
  survives a cleared profile.
</p>
<script>
const DATA = __PAYLOAD__;
const KEY = "second-annotator-v1";
let si = 0, lastOk = null;
// A file:// document has an opaque origin and a browser may refuse it storage
// outright (Safari always; Chrome/Edge with site data blocked). Reading it
// unguarded threw before render() ever ran, which left the page as empty chrome.
// Keep working in memory instead, and say so in the header.
let volatileState = false;
const store = {
  read(){ try { return localStorage.getItem(KEY); }
          catch(e){ volatileState = true; return null; } },
  write(v){ try { localStorage.setItem(KEY, v); }
            catch(e){ volatileState = true; } },
};
let state = JSON.parse(store.read() || "null") || {replies:{}, accepted:{}};

function seg(){ return DATA.segments[si]; }
function esc(s){ return (s||"").replace(/[&<>]/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }
function persist(){ store.write(JSON.stringify(state)); paint(); }
// Folding matches src/gold_store.is_verbatim: whitespace runs collapse and the
// curly/straight quote and dash families are one character. Anything else — a
// tidied replacement char, a dropped comma — must fail, or the offsets the gold
// is anchored by would not resolve.
function fold(s){ return (s||"").replace(/[‘’]/g,"'").replace(/[“”]/g,'"')
                    .replace(/[–—−]/g,"-").replace(/\s+/g," ").trim(); }

function parseReply(raw){
  let t = (raw||"").trim();
  if (!t) throw "empty";
  const f = t.match(/```(?:json)?\s*([\s\S]*?)```/i);      // tolerate fenced output
  if (f) t = f[1].trim();
  const i = t.indexOf("{"), j = t.lastIndexOf("}");
  if (i > 0 || j < t.length-1){ if (i<0||j<0) throw "no JSON object in the reply";
                                t = t.slice(i, j+1); }
  let p;
  try { p = JSON.parse(t); } catch(e){ throw "not valid JSON: " + e.message; }
  if (!p || !Array.isArray(p.facts)) throw "no `facts` array";
  return p;
}

function validate(payload, s){
  const errs = [], warns = [], text = fold(s.text);
  // Node inventory over the WHOLE reply for this segment: R1b can only be judged
  // once you know what this annotation predicates of.
  const key = x => fold(x||"").toLowerCase().replace(/^(the|a|an)\s+/,"");
  const nodes = new Set();
  payload.facts.forEach(f => (Array.isArray(f && f.triples) ? f.triples : [])
    .forEach(t => { ["subject","object"].forEach(kk => { const v = key(t && t[kk]);
                                                         if (v) nodes.add(v); }); }));
  payload.facts.forEach((f, k) => {
    const at = "fact " + (k+1);
    if (!f || typeof f !== "object"){ errs.push(at + ": not an object"); return; }
    if (!(f.natural_language||"").trim()) errs.push(at + ": empty natural_language");
    const q = fold(f.source_quote||"");
    if (!q) errs.push(at + ": empty source_quote");
    else if (!text.includes(q))
      errs.push(at + ": source_quote is not verbatim in the segment — “"
                + (f.source_quote||"").slice(0,80) + "”");
    if (!Array.isArray(f.triples) || !f.triples.length){
      errs.push(at + ": no triples"); return; }
    f.triples.forEach((t, ti) => {
      ["subject","predicate","object"].forEach(kk => {
        if (!t || !((t[kk]||"").trim())) errs.push(at + " triple " + (ti+1) + ": empty " + kk);
      });
    });
    // R4: a date must not sit in the object and in temporal_context at once
    const tc = fold(f.temporal_context||"");
    if (tc && f.triples.some(t => fold(t.object||"") === tc))
      warns.push(at + ": the object repeats temporal_context — R4 says pick one");
    // R2: after the first, a subject should pick up something already introduced
    f.triples.forEach((t, ti) => {
      if (!ti) return;
      const earlier = f.triples.slice(0,ti).flatMap(p => [key(p.object), key(p.subject)]);
      const sub = key(t.subject);
      if (!earlier.some(e => e === sub || e.includes(sub)))
        warns.push(at + " triple " + (ti+1) + ": subject “" + t.subject
                   + "” picks up nothing earlier — the chain does not connect");
    });
    // R1b: an entity this reply predicates of, left sitting inside a predicate.
    // Counting predicate words was the earlier proxy and it was a bad one — it
    // fires on "shall be binding in its entirety and directly applicable in",
    // which hides nothing, and stays silent on "may extend the time-limits by",
    // which hides a node (BUG-2026-08-17-A, D-19).
    f.triples.forEach((t, ti) => {
      const hit = [...(t.predicate||"").matchAll(/\b(?:the|a|an)\s+[a-z][\w'-]*(?:[ -][a-z][\w'-]*)?/gi)]
        .map(m => m[0].trim())
        .flatMap(ph => { const w = ph.split(/\s+/);
                         return w.length > 2 ? [ph, w.slice(0,2).join(" ")] : [ph]; })
        .find(ph => nodes.has(key(ph)));
      if (hit)
        warns.push(at + " triple " + (ti+1) + ": the predicate carries “" + hit
                   + "”, which this segment also uses as a node — R1b says split it out");
    });
    // R8: one entity, one label. The chain check above accepts a subject that is a
    // substring of an earlier node, which is what lets "the forecast" pass next to
    // "the Commission services' Spring 2005 forecast" — one entity, two KG nodes,
    // and neither the chain check nor R1b sees it. So it is caught here instead.
    // Only a shared HEAD counts: English noun phrases are head-final, so "the
    // Commission" inside "the Commission services' forecast" is a different entity,
    // while anything ending in "forecast" is the same one. A label over six words is
    // not an entity name (entity_norm cuts there) and is left to the chain check.
    // A label carrying a PP has its head at the front ("the finding of an excessive
    // deficit" is a finding, not a deficit), and a coordination legitimately appears
    // both whole and per branch ("white and raw sugar" beside "raw sugar"), so neither
    // is an alias. Without those two exclusions this fired on 12 gold labels.
    const notAlias = s2 => /\s(?:of|on|in|for|to|by|under)\s|\s(?:and|or)\s/.test(s2);
    const seenPair = new Set();
    f.triples.forEach((t, ti) => {
      ["subject","object"].forEach(kk => {
        const v = key(t[kk]); if (!v || notAlias(v)) return;
        const full = [...nodes].find(n => n !== v && n.endsWith(" " + v)
                                          && !notAlias(n)
                                          && n.split(/\s+/).length <= 6);
        if (full && !seenPair.has(v + "|" + full)) {
          seenPair.add(v + "|" + full);
          warns.push(at + " triple " + (ti+1) + ": “" + t[kk] + "” shares its head with "
                     + "“" + full + "”, also used here — R8: one label per entity, or "
                     + "name the two apart");
        }
      });
    });
    // R8: a slot left as a pronoun or an unresolved demonstrative. A closing formula
    // is self-referential by nature, so the flag is only meaningful off boilerplate.
    if (!f.boilerplate) f.triples.forEach((t, ti) => {
      ["subject","object"].forEach(kk => {
        if (/^(it|they|them|such|this|that|these|those|which)\b/i.test(fold(t[kk]||"")))
          warns.push(at + " triple " + (ti+1) + ": " + kk + " “" + t[kk]
                     + "” is not resolved — R8 says name the entity");
      });
    });
    // R2: a chain stopped one link early. The word count was the only proxy for this
    // and it is a poor one — "the Treaty establishing the European Community" is six
    // words and holds a whole triple, while a nine-word measurement holds none.
    // What counts is a REDUCED RELATIVE CLAUSE left inside the label. An of/for/by
    // nominalisation does NOT count: `data/gold/facts_hires` keeps 24 of them ("the
    // finding of an excessive deficit", "the reporting by the Netherlands", "the sale
    // of natural gas") because each names one thing, and the earlier draft of this
    // check condemned all 24. Measured against the gold, the clause test flags none.
    const CLAUSE = /\b(establishing|annexed|issued|drawn up|provided for|referred to|setting out|laying down|amending|repealing|abrogating)\b/i;
    const FOOTNOTE = /\s\(\d{1,2}\)\s*$/;              // an OJ footnote marker, not a name
    const seenRel = new Set();
    f.triples.forEach((t, ti) => {
      ["subject","object"].forEach(kk => {
        const v = (t[kk]||"").trim();
        if (seenRel.has(v)) return;                    // a label chains twice; say it once
        if (CLAUSE.test(v)) {
          seenRel.add(v);
          warns.push(at + " triple " + (ti+1) + ": " + kk + " “" + v + "” keeps a reduced "
                     + "relative clause — R2 says that clause is its own triple");
        } else if (FOOTNOTE.test(v)) {
          seenRel.add(v);
          warns.push(at + " triple " + (ti+1) + ": " + kk + " “" + v + "” ends in an OJ "
                     + "footnote marker, which is not part of the entity's name");
        }
      });
    });
    // RETIRED 2026-09-07 (decision log D-30): "an object runs over 9 words".
    // Measured against data/gold/facts_hires it fired on 7 of 556 triples and
    // every one was legitimate — two foreign institution names ("Ministerio de
    // Ganaderia ... Unidad Ejecutora de Semillas, Montevideo"), two CN-code
    // lists ("6104 41 00, 6104 42 00, ..."), three entry-into-force formulas
    // ("the third day following its publication in the Official Journal of the
    // European Communities"). Precision 0/7. It contradicted the comment
    // fifteen lines above, which had already retired word-counting as a proxy
    // for an unsplit chain and replaced it with CLAUSE; a warning the reviewer
    // has to dismiss on every long-but-correct label teaches him to ignore the
    // badge column, which is the one cost this console cannot afford.
    // What actually catches a chain stopped early is CLAUSE/FOOTNOTE above.
  });
  if (!payload.facts.length) warns.push("the reply annotates nothing");
  return {errs, warns};
}

function check(){
  const s = seg(), box = document.getElementById("issues");
  const pv = document.getElementById("preview"), vd = document.getElementById("verdict");
  lastOk = null; document.getElementById("acceptbtn").disabled = true;
  pv.innerHTML = ""; box.innerHTML = "";
  let payload;
  try { payload = parseReply(document.getElementById("reply").value); }
  catch(e){ vd.innerHTML = '<span class="tag err">unreadable</span>';
            box.innerHTML = '<div class="e">' + esc(String(e)) + '</div>'; return; }
  const {errs, warns} = validate(payload, s);
  vd.innerHTML = errs.length ? '<span class="tag err">' + errs.length + ' error(s)</span>'
               : (warns.length ? '<span class="tag wrn">ok, ' + warns.length + ' warning(s)</span>'
                               : '<span class="tag ok">clean</span>');
  box.innerHTML = errs.map(e => '<div class="e">&times; ' + esc(e) + '</div>').join("")
                + warns.map(w => '<div class="w">! ' + esc(w) + '</div>').join("");
  pv.innerHTML = payload.facts.map((f,k) => {
    const bad = errs.some(e => e.startsWith("fact " + (k+1) + ":")
                            || e.startsWith("fact " + (k+1) + " "));
    return '<div class="fact' + (bad ? ' bad' : '') + '">'
      + '<div class="nl">' + esc(f.natural_language) + '</div>'
      + (f.triples||[]).map(t => '<div class="tr">(' + esc(t.subject) + ') &mdash;['
          + esc(t.predicate) + ']&rarr; (' + esc(t.object) + ')</div>').join("")
      + (f.condition ? '<div class="tr">condition: ' + esc(f.condition) + '</div>' : '')
      + (f.temporal_context ? '<div class="tr">temporal: ' + esc(f.temporal_context) + '</div>' : '')
      + '<div class="q">&ldquo;' + esc(f.source_quote) + '&rdquo;</div></div>';
  }).join("");
  if (!errs.length){ lastOk = payload; document.getElementById("acceptbtn").disabled = false; }
}

function accept(){
  if (!lastOk) return;
  state.accepted[seg().segment_id] = {facts: lastOk.facts,
                                      accepted_at: new Date().toISOString(),
                                      prompt_version: DATA.prompt_version};
  persist();
  if (si < DATA.segments.length - 1) go(1); else render();
}
function clearOne(){
  delete state.accepted[seg().segment_id];
  delete state.replies[seg().segment_id];
  document.getElementById("reply").value = "";
  persist(); render();
}
function stash(){ state.replies[seg().segment_id] = document.getElementById("reply").value;
                  store.write(JSON.stringify(state)); }

function copyPrompt(){
  const t = seg().prompt, v = document.getElementById("verdict");
  const ok = () => { v.innerHTML = '<span class="tag ok">prompt copied</span>'; };
  // navigator.clipboard exists only in a secure context, and file:// is not one,
  // so reaching for it unconditionally made this button throw there. Fall back to
  // execCommand, then to a window the prompt can be copied out of by hand.
  const legacy = () => {
    const ta = document.createElement("textarea");
    ta.value = t; ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select();
    let done = false;
    try { done = document.execCommand("copy"); } catch(e){ done = false; }
    ta.remove();
    if (done) return ok();
    const w = window.open("", "_blank");
    if (w) w.document.write("<pre>" + esc(t) + "</pre>");
    else v.innerHTML = '<span class="tag err">could not copy &mdash; serve the page '
                     + 'over http, or use --dump-prompt</span>';
  };
  try {
    if (navigator.clipboard && navigator.clipboard.writeText)
      navigator.clipboard.writeText(t).then(ok, legacy);
    else legacy();
  } catch(e){ legacy(); }
}

function render(){
  const s = seg();
  document.getElementById("pos").textContent = (si+1) + " / " + DATA.segments.length;
  document.getElementById("smeta").innerHTML = "<b>" + esc(s.segment_id) + "</b> &middot; "
    + esc(s.doc_title.slice(0,80)) + ' <span class="tag">' + esc(s.kind) + "</span>"
    + (state.accepted[s.segment_id] ? ' <span class="tag ok">accepted</span>' : "");
  document.getElementById("stext").textContent = s.text;
  document.getElementById("step").innerHTML =
    "1. <b>Copy prompt</b> &rarr; 2. paste it into the model &rarr; 3. paste the reply below "
    + "&rarr; 4. <b>Check reply</b> &rarr; 5. <b>Accept &amp; next</b>";
  document.getElementById("reply").value =
    state.replies[s.segment_id] || (state.accepted[s.segment_id]
      ? JSON.stringify({facts: state.accepted[s.segment_id].facts}, null, 1) : "");
  document.getElementById("issues").innerHTML = "";
  document.getElementById("preview").innerHTML = "";
  document.getElementById("verdict").innerHTML = "";
  document.getElementById("acceptbtn").disabled = true;
  lastOk = null;
  document.getElementById("queue").innerHTML = DATA.segments.map((x,i) =>
    '<span class="' + (state.accepted[x.segment_id] ? "done " : "")
    + (x.iaa ? "iaa " : "") + (i===si ? "cur" : "")
    + '" title="' + esc(x.segment_id) + (x.iaa ? " \u00b7 frozen IAA subsample" : "")
    + '" onclick="jump(' + i + ')">' + esc(chip(x)) + "</span>").join("");
  paint();
}
function paint(){
  const n = Object.keys(state.accepted).length, tot = DATA.segments.length;
  document.getElementById("prog").style.width = (100*n/tot) + "%";
  const ni = DATA.segments.filter(s => s.iaa && state.accepted[s.segment_id]).length;
  document.getElementById("cnt").innerHTML = esc(n + " / " + tot + " accepted"
    + (DATA.n_iaa && DATA.n_iaa < tot ? "  (IAA " + ni + " / " + DATA.n_iaa + ")" : ""))
    + (volatileState ? ' <span class="tag err">this browser is not storing progress '
                     + '&mdash; download the bundle before you close the tab</span>' : "");
  // Segments accepted under an earlier prompt are not comparable to ones accepted
  // under this one; the count says so instead of leaving it to be discovered at
  // import time.
  const stale = DATA.segments.filter(s => state.accepted[s.segment_id] &&
    (state.accepted[s.segment_id].prompt_version || "") !== DATA.prompt_version).length;
  document.getElementById("gver").innerHTML = esc(DATA.prompt_version)
    + (stale ? ' <span class="tag wrn">' + stale + ' accepted under an earlier '
             + 'guideline</span>' : "");
}
// 59 chips across 12 documents: the section path alone repeats, so name the doc.
function chip(x){
  return x.doc_id.replace(/^train-0*(?=[0-9])/, "") + " "
       + x.section_path.replace(/^preamble\./, "p.")
                       .replace(/^enacting\./, "e.");
}
function jump(i){ si = i; render(); }
function go(d){ si = (si + d + DATA.segments.length) % DATA.segments.length; render(); }

function save(){
  const left = DATA.segments.filter(s => !state.accepted[s.segment_id]);
  if (left.length && !confirm(left.length + " segment(s) are not accepted yet. "
      + "Download the partial bundle anyway?")) return;
  const out = {produced_on: new Date().toISOString(),
               pass: "second annotator, blind",
               scope: DATA.scope,
               prompt_version: DATA.prompt_version,
               segments: DATA.segments.map(s => ({
                 segment_id: s.segment_id, doc_id: s.doc_id,
                 section_path: s.section_path, iaa: !!s.iaa,
                 // A segment accepted before the guideline was repaired carries no
                 // stamp; say so rather than letting it inherit the current one.
                 prompt_version: (state.accepted[s.segment_id]||{}).prompt_version
                                 || "second-annotator-prompt-v1 (unstamped)",
                 facts: (state.accepted[s.segment_id]||{}).facts || null}))};
  const b = new Blob([JSON.stringify(out, null, 2)], {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(b); a.download = "second_annotation_bundle.json"; a.click();
}
function loadFile(inp){
  const file = inp.files && inp.files[0]; inp.value = "";
  if (!file) return;
  const r = new FileReader();
  r.onload = () => {
    let p; try { p = JSON.parse(r.result); } catch(e){ alert("Not a JSON file."); return; }
    if (!p || !Array.isArray(p.segments)){ alert("No `segments` array in that file."); return; }
    const n = p.segments.filter(s => s.facts).length;
    if (!confirm("Replace the progress in this browser with " + n
                 + " accepted segment(s) from " + file.name + "?")) return;
    state = {replies:{}, accepted:{}};
    p.segments.forEach(s => { if (s.facts)
      state.accepted[s.segment_id] = {facts: s.facts, accepted_at: p.produced_on,
                                      prompt_version: s.prompt_version}; });
    persist(); render();
  };
  r.readAsText(file);
}
render();
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=GOLD / "second_annotator.html")
    ap.add_argument("--iaa-only", action="store_true",
                    help="only the 9 frozen IAA segments (the page as it stood "
                         "before 2026-08-18)")
    ap.add_argument("--dump-prompt", metavar="SEGMENT_ID",
                    help="print one prompt to stdout instead of writing the page")
    args = ap.parse_args()

    payload = build_payload(iaa_only=args.iaa_only)
    if args.dump_prompt:
        hit = [s for s in payload["segments"] if s["segment_id"] == args.dump_prompt]
        if not hit:
            raise SystemExit(f"not in this page's queue: {args.dump_prompt}\n  "
                             + "\n  ".join(s["segment_id"] for s in payload["segments"]))
        # The segment text legitimately carries U+FFFD (the prompt asks for it to
        # be preserved), which a GBK console cannot encode. Write UTF-8 rather
        # than lose the character the annotator is told to copy verbatim.
        sys.stdout.reconfigure(encoding="utf-8")
        print(hit[0]["prompt"])
        return

    args.out.write_text(HTML.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False)),
                        encoding="utf-8")
    print(f"wrote {args.out}  ({payload['n_segments']} segments, "
          f"{payload['n_iaa']} of them the frozen IAA subsample, "
          f"{args.out.stat().st_size//1024} KB)")
    print("  blind: the page embeds segment text only — no fact from data/gold/ is in it")
    for s in payload["segments"]:
        print(f"    {s['segment_id']:<42} {len(s['text']):>5} chars"
              + ("   [IAA]" if s["iaa"] else ""))


if __name__ == "__main__":
    main()
