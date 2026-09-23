"""Emit a hand-off pack so a DIFFERENT model can adjudicate the gold independently.

Why this exists. The gold was pre-annotated by `claude-opus-5` and then
adjudicated by `claude-opus-5`. Discipline rule 6 rejects correctness numbers
computed against it, and the reason is circularity, not machine-ness: the same
model marking its own output is not a second opinion. Having a model from a
different family adjudicate removes exactly that defect. The result is still a
machine adjudication and must be recorded as one --- it does not turn the gold
into human-adjudicated material --- but it is no longer a model scoring itself,
and the two passes together give a cross-family adjudicator-agreement figure,
which is a number the Discussion currently does not have at all.

What is in the pack

    _INSTRUCTIONS.md   paste ONCE at the start of the session: the task, the
                       annotation protocol, the shared codebook (decision log),
                       and the exact output format.
    <doc_id>.md        paste one per turn. Blind packet for that document:
                       segment text, the pre-annotated facts, and the mechanical
                       flags. It does NOT carry the first adjudicator's notes ---
                       see `make_adjudication_packet --blind`.

The reply to each document goes back through
`scripts/import_external_adjudication.py`, which validates the ids and writes a
decisions file with `mode: "machine"` so `apply_gold_review.py` cannot mistake it
for a human sweep.

    python -m scripts.make_external_adjudication_task
    python -m scripts.make_external_adjudication_task --out DIR --model gemini
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.make_adjudication_packet import _machine_review, build_doc_packet
from src.gold_store import GOLD, GOLD_FACTS, sample_index

OUT_DEFAULT = GOLD / "external_task"
PROTOCOL = Path(__file__).resolve().parent.parent / "doc" / "annotation_protocol.md"


OUTPUT_CONTRACT = """\
## What to return

For **every** fact id in the document, return one entry. Return **only** a JSON
object in a single ```json code block, nothing before or after it:

```json
{
  "doc_id": "<the DOC id at the top of the packet>",
  "decisions": {
    "<gold_id exactly as printed>": {
      "status": "accepted" | "edited" | "rejected",
      "note": "<= 30 words: why. Required for edited and rejected.",
      "edits": { "subject": "...", "predicate": "...", "object": "...",
                 "natural_language": "...", "source_quote": "...",
                 "condition": "...", "temporal_context": "..." }
    }
  },
  "added": [
    { "segment_id": "<segment id>",
      "natural_language": "...", "source_quote": "<verbatim from SOURCE TEXT>",
      "subject": "...", "predicate": "...", "object": "...",
      "why": "<= 30 words: what the pre-annotation missed" }
  ]
}
```

Rules for the output

* `status` is mandatory and must be one of the three literals.
* `edits` is **only** present for `status: "edited"`, and contains **only** the
  slots you are changing. Omit it entirely otherwise.
* `source_quote`, whether kept or edited, must appear **verbatim** in the
  SOURCE TEXT of that segment. This is checked on import and a mismatch is
  rejected.
* `added` is for clauses the pre-annotation missed entirely. Use it sparingly
  and leave it as `[]` if there are none.
* Do not omit facts. Every `[fNN]` id printed in the packet must appear once.
* Do not renumber, reorder or invent ids.
"""


def instructions(model_label: str) -> str:
    protocol = PROTOCOL.read_text(encoding="utf-8")
    log = json.loads((GOLD / "decision_log.json").read_text(encoding="utf-8"))
    entries = log.get("entries", [])

    codebook = ["## Shared codebook — conventions already settled",
                "",
                "These were decided while annotating and are binding, so that you and the",
                "first adjudicator differ about the *text* rather than about conventions.",
                "Apply them; do not re-litigate them.",
                ""]
    for e in entries:
        codebook += [f"**{e['id']} — {e.get('title','')}**", "",
                     f"- *Problem:* {e.get('problem','').strip()}",
                     f"- *Choice:* {e.get('choice','').strip()}", ""]

    return f"""\
# Independent adjudication of a legal-text atomic-fact gold set

You are the **second, independent adjudicator** of a small annotated corpus of
EU legislation. A first pass produced the facts you will see. Your job is to
decide, for each one, whether it is what the source text actually says.

You are **not** reviewing the first adjudicator's reasoning, and you will not be
shown it. Decide from the SOURCE TEXT and the rules below, as if the facts had
arrived unattributed. Where you agree, say `accepted` — agreement is a real
finding here, not a failure to contribute.

## The task, per fact

- **accepted** — the fact is a correct, verbatim-anchored reading of the text.
- **edited** — the content is right but a slot is wrong (wrong subject, an
  object that overshoots the quote, a non-verbatim quote, a missing condition).
  Give the corrected slots.
- **rejected** — the fact is not supported by the text, duplicates another, or
  asserts something the text does not.

Judge only against the SOURCE TEXT shown with each segment. Do not use outside
knowledge of EU law to supply facts the text does not state.

Lines marked `FLAG` are **mechanical checks**, not opinions — e.g. "no content
word of the object appears in the quote". They point at things worth looking at.
A flag is not a verdict; plenty of flagged facts are fine.

{OUTPUT_CONTRACT}

---

## Annotation protocol

{protocol}

---

{chr(10).join(codebook)}
---

## How the session runs

Paste this file once. Then paste one document packet per message. Reply to each
with the JSON block for that document and nothing else. Adjudicator recorded as:
`{model_label}`.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--model", default="gemini",
                    help="label recorded as the adjudicator (default: gemini)")
    ap.add_argument("--doc", default=None, help="single doc_id (default: all)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    flags = _machine_review()
    expect = sample_index()

    (args.out / "_INSTRUCTIONS.md").write_text(instructions(args.model), encoding="utf-8")
    print(f"  + {args.out / '_INSTRUCTIONS.md'}   <- paste this FIRST, once")

    doc_ids = [args.doc] if args.doc else sorted(p.stem for p in GOLD_FACTS.glob("*.json"))
    total = 0
    for doc_id in doc_ids:
        packet = build_doc_packet(doc_id, flags, expect, blind=True)
        n = sum(1 for line in packet.splitlines() if line.strip().startswith("[f"))
        total += n
        body = (f"Adjudicate this document. Reply with the JSON block only.\n\n"
                f"```\n{packet}\n```\n")
        path = args.out / f"{doc_id}.md"
        path.write_text(body, encoding="utf-8")
        print(f"  + {path.name}  ({n} facts, {len(body)//1024} KB)")

    print(f"\n{len(doc_ids)} document(s), {total} facts. "
          f"Paste _INSTRUCTIONS.md once, then one document per message.")
    print("Save each reply as <doc_id>.json, then:")
    print(f"    python -m scripts.import_external_adjudication <dir> --model {args.model}")


if __name__ == "__main__":
    main()
