"""Emit one self-contained adjudication packet per gold document.

Adjudication means deciding, for every pre-annotated fact, whether it is what
the segment text actually says: `accepted`, `edited` or `rejected`. That
decision can only be made against the *source text*, so the packet puts the
segment text first and the pre-annotated facts second, in reading order, with
the machine pre-review flags attached and the decision log inlined.

    python -m scripts.make_adjudication_packet            # -> data/gold/packets/
    python -m scripts.make_adjudication_packet --out DIR --doc train-000005

The reverse direction is `scripts/apply_gold_review.py`, which takes a
decisions file in the shape this packet documents:

    {"reviewed_on": "...", "reviewer": "...", "mode": "machine|human",
     "decisions": {"<gold_id>": {"status": "accepted|edited|rejected",
                                 "note": "...", "rule": "D-02",
                                 "edits": {"subject": "...", ...}}},
     "added": [{"segment_id": "...", "fact": { ...gold record... }}]}

Nothing here writes to the gold. The packet is read-only working material.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.gold_store import GOLD, GOLD_FACTS, sample_index, segment_text

OUT_DEFAULT = GOLD / "packets"


def _machine_review() -> dict:
    p = GOLD / "machine_review.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8")).get("facts", {})


def _fact_block(f: dict, flags: dict, blind: bool = False) -> list[str]:
    t = f["triples"][0] if f["triples"] else {}
    lines = [
        f"  [{f['gold_id'].rsplit('::', 1)[1]}] {f['gold_id']}",
        f"      quote : {f['source_quote']!r}",
        f"      nl    : {f['natural_language']}",
        f"      spo   : ({t.get('subject','')}) -- ({t.get('predicate','')}) --> ({t.get('object','')})",
    ]
    if len(f["triples"]) > 1:
        for extra in f["triples"][1:]:
            lines.append(
                f"      spo+  : ({extra.get('subject','')}) -- "
                f"({extra.get('predicate','')}) --> ({extra.get('object','')})")
    qual = []
    if f.get("condition"):
        qual.append(f"condition={f['condition']!r}")
    if f.get("temporal_context"):
        qual.append(f"temporal={f['temporal_context']!r}")
    if f.get("logic_op"):
        qual.append(f"logic={f['logic_op']}/{f['logic_group']}")
    if f.get("boilerplate"):
        qual.append("boilerplate=true")
    if qual:
        lines.append("      qual  : " + "  ".join(qual))
    # `reviewer_note` is the FIRST adjudicator's reasoning ("D-15: coordination
    # left unexpanded ..."). A packet meant for an INDEPENDENT second pass must
    # not carry it: an adjudicator shown the first one's justification is not a
    # second opinion, it is a review of the first opinion, and the agreement rate
    # computed from it means nothing. The flags stay in both modes because
    # `machine_review_gold` only runs mechanically decidable checks — they are
    # properties of the text, not anyone's judgement.
    if f.get("reviewer_note") and not blind:
        lines.append(f"      note  : {f['reviewer_note']}")
    fl = flags.get(f["gold_id"])
    if fl:
        for c in fl["checks"]:
            if blind and c["code"] == "carries_annotator_note":
                continue          # same leak, wearing a flag
            lines.append(f"      FLAG  : [{c['severity']}] {c['code']} — {c['detail']}")
    return lines


def build_doc_packet(doc_id: str, flags: dict, expect: dict,
                     blind: bool = False) -> str:
    payload = json.loads((GOLD_FACTS / f"{doc_id}.json").read_text(encoding="utf-8"))
    out: list[str] = [
        "=" * 78,
        f"DOC {doc_id} — {payload.get('doc_title', '')}",
        "=" * 78,
    ]
    n_facts = 0
    for seg in payload["segments"]:
        sid = seg["segment_id"]
        meta = expect[sid]
        text = segment_text(doc_id, meta["section_path"], meta["char_start"], meta["char_end"])
        out += [
            "",
            "-" * 78,
            f"SEGMENT {sid}   ({len(seg['facts'])} facts)",
            "-" * 78,
            "SOURCE TEXT:",
            text.strip(),
            "",
            "PRE-ANNOTATED FACTS:",
        ]
        for f in seg["facts"]:
            n_facts += 1
            out += _fact_block(f, flags, blind=blind)
            out.append("")
    out.append(f"[{doc_id}: {len(payload['segments'])} segments, {n_facts} facts]")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--doc", default=None, help="single doc_id (default: all)")
    ap.add_argument("--blind", action="store_true",
                    help="omit the first adjudicator's reviewer_note (and the flag that "
                         "merely reports one). Use this for an INDEPENDENT second pass: an "
                         "adjudicator shown the first one's reasoning is reviewing it, not "
                         "re-deciding, and the agreement rate becomes meaningless.")
    args = ap.parse_args()

    flags = _machine_review()
    expect = sample_index()
    args.out.mkdir(parents=True, exist_ok=True)

    doc_ids = [args.doc] if args.doc else sorted(p.stem for p in GOLD_FACTS.glob("*.json"))
    for doc_id in doc_ids:
        text = build_doc_packet(doc_id, flags, expect, blind=args.blind)
        path = args.out / f"{doc_id}.txt"
        path.write_text(text, encoding="utf-8")
        print(f"  + {path}  ({len(text.splitlines())} lines)")

    # The rules the adjudicator has to apply, in one place next to the packets.
    rules = {
        "protocol": (GOLD.parent.parent / "doc" / "annotation_protocol.md").read_text(encoding="utf-8"),
        "decision_log": json.loads((GOLD / "decision_log.json").read_text(encoding="utf-8")),
    }
    (args.out / "_rules.json").write_text(
        json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  + {args.out / '_rules.json'}")


if __name__ == "__main__":
    main()
