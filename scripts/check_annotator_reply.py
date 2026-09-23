"""Validate one second-annotator reply before it goes anywhere near the bundle.

    python -m scripts.check_annotator_reply <segment_id> reply.json

`data/gold/second_annotator.html` validates a reply in the browser as it is pasted
in. That is the normal route and it should stay the normal route. This script is
the same checks off the page, for the case where a reply arrives some other way —
handed over in a chat, recovered from a log, rebuilt from a transcript — and would
otherwise reach `scripts/import_second_annotation.py` unchecked.

It is deliberately stricter than the importer. The importer's job is to refuse a
malformed artefact; this one's job is to say "re-ask the model" while re-asking is
still one prompt rather than a discovery made after the whole queue is done.

ERRORS (the reply must be re-asked):
    * a `source_quote` that is not verbatim in this segment
    * an empty natural_language, no triples, an empty slot
    * `logic_op` outside AND / OR / XOR, or a `logic_group` with one member
    * the SAME triple asserted twice — on a flattened table that means two rows
      were merged under one label, which is rule 11 / rule 6(a)

WARNINGS (read them, then decide):
    * a quote that matches the segment in more than one place, or is too short to
      anchor — a bare cell value does not say which row a fact came from
    * a chain whose subject picks up nothing earlier
    * a node that is a flattened row rather than its cells (rule 11)
    * two labels that may be one entity (rule 9)

Note what the rule-11 check does NOT flag: a table the source text itself names
(`the table in Part I (2)`). That is a real legal object and a legitimate node.
Only `row 3`-shaped labels are artefacts of the flattening.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import gold_store as gs  # noqa: E402

VALID_OPS = {"", "AND", "OR", "XOR"}
MIN_QUOTE_CHARS = 12
# rule 11: the flattening's own row handle, never a node. `the table in Part I (2)`
# is NOT this — the source names that table itself.
ROW_NODE = re.compile(r"^(the\s+)?rows?\s*\d*\s*$|^row\s+\d+\b", re.I)


def norm(s: str) -> str:
    s = (s or "").strip().lower()
    for art in ("the ", "a ", "an "):
        if s.startswith(art):
            return s[len(art):]
    return s


def check(segment_id: str, facts: list[dict]) -> tuple[list[str], list[str]]:
    idx = gs.sample_index()
    if segment_id not in idx:
        raise SystemExit(f"{segment_id}: not in the frozen 59-segment sample")
    m = idx[segment_id]
    text = gs.segment_text(m["doc_id"], m["section_path"], m["char_start"], m["char_end"])
    folded = gs.fold(text)

    errors: list[str] = []
    warnings: list[str] = []

    for k, f in enumerate(facts, start=1):
        gid = f"b{k:02d}"
        if not (f.get("natural_language") or "").strip():
            errors.append(f"{gid}: empty natural_language")

        quote = f.get("source_quote", "")
        if not gs.is_verbatim(quote, text):
            errors.append(f"{gid}: source_quote NOT verbatim — {quote[:70]!r}")
        else:
            fq = gs.fold(quote)
            hits = folded.count(fq)
            if hits > 1:
                warnings.append(f"{gid}: source_quote {fq[:40]!r} matches this segment "
                                f"in {hits} places — it does not say which one")
            if len(fq) < MIN_QUOTE_CHARS:
                warnings.append(f"{gid}: source_quote {fq!r} is {len(fq)} chars — "
                                "too short to anchor (rule 11: quote the whole row)")

        triples = f.get("triples") or []
        if not triples:
            errors.append(f"{gid}: no triples")
        for ti, t in enumerate(triples, start=1):
            for slot in ("subject", "predicate", "object"):
                if not ((t or {}).get(slot) or "").strip():
                    errors.append(f"{gid}: triple {ti} has an empty {slot}")
            for slot in ("subject", "object"):
                lab = ((t or {}).get(slot) or "").strip()
                if ROW_NODE.match(lab):
                    warnings.append(f"{gid}: triple {ti} {slot} {lab!r} is a flattened "
                                    "row, not content (rule 11)")

        if (f.get("logic_op") or "") not in VALID_OPS:
            errors.append(f"{gid}: logic_op {f.get('logic_op')!r} is not AND/OR/XOR")

        for ti, t in enumerate(triples[1:], start=1):
            earlier = {norm(x.get("object", "")) for x in triples[:ti]}
            earlier |= {norm(x.get("subject", "")) for x in triples[:ti]}
            sub = norm((t or {}).get("subject", ""))
            if sub and not any(sub == e or sub in e for e in earlier):
                warnings.append(f"{gid}: triple {ti + 1} subject "
                                f"{t['subject']!r} picks up nothing earlier")

    groups = Counter(f.get("logic_group") for f in facts
                     if (f.get("logic_group") or "").strip())
    for g, n in groups.items():
        if n < 2:
            errors.append(f"logic_group {g!r} has a single member (rule 8)")
        quotes = {f["source_quote"] for f in facts if f.get("logic_group") == g}
        if len(quotes) > 1:
            warnings.append(f"logic_group {g!r} spans {len(quotes)} different quotes — "
                            "rule 8 expects one span per coordination")

    seen: dict[tuple, list[str]] = {}
    for k, f in enumerate(facts, start=1):
        sig = tuple(sorted((norm(t.get("subject", "")), norm(t.get("predicate", "")),
                            norm(t.get("object", "")))
                           for t in f.get("triples") or []))
        seen.setdefault(sig, []).append(f"b{k:02d}")
    for sig, who in seen.items():
        if len(who) > 1:
            errors.append(f"the same triple set is asserted by {', '.join(who)}: "
                          f"{sig[0] if sig else '()'} — on a table this means two rows "
                          "were merged under one label (rules 11, 6a)")

    nodes = {t.get(s, "") for f in facts for t in (f.get("triples") or [])
             for s in ("subject", "object")}
    labels = sorted(x for x in nodes if x)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            na, nb = norm(a), norm(b)
            if na == nb:
                continue
            long_, short_ = (na, nb) if len(na) > len(nb) else (nb, na)
            if long_.endswith(" " + short_) and len(short_) > 2:
                head = long_[: -(len(short_) + 1)].split()
                if len(head) == 1 and head[0] not in {
                        "of", "by", "to", "for", "in", "on", "under", "with", "from"}:
                    warnings.append(f"rule 9: {a!r} and {b!r} may be one entity")
    return errors, warnings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("segment_id")
    ap.add_argument("reply", type=Path)
    args = ap.parse_args()

    payload = json.loads(args.reply.read_text(encoding="utf-8"))
    facts = payload["facts"] if isinstance(payload, dict) else payload
    errors, warnings = check(args.segment_id, facts)

    n_tr = sum(len(f.get("triples") or []) for f in facts)
    print(f"{args.segment_id}: {len(facts)} facts, {n_tr} triples")
    if errors:
        print(f"\n  {len(errors)} ERROR(S) — re-ask the model, do not edit the reply:")
        for e in errors:
            print("   !!", e)
    else:
        print("\n  no errors: every quote verbatim, no empty slot, no duplicate fact")
    if warnings:
        print(f"\n  {len(warnings)} warning(s):")
        for w in warnings:
            print("   ~", w)
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
