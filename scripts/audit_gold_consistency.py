"""Does the gold apply its own rules the same way twice?

    python -m scripts.audit_gold_consistency
    python -m scripts.audit_gold_consistency --facts-root data/gold/annotator_b

Not agreement between annotators — that is `scripts/compare_annotators.py`. This
asks a narrower question of ONE annotation: where the guideline is structural, is
it followed; and where it under-determines, does the same construction get handled
two ways inside the same artefact?

It exists because the numbers it produces are quoted in
`data/gold/calibration/_RULES.md` (rule 9's "kept as a check, not an invariant")
and in `doc/CHANGELOG.md`. A figure in the guideline with no script behind it is
the same defect as a provenance stamp with no decisions artefact behind it
(BUG-2026-08-27-A) — so the measurement is here, and re-runnable.

The four checks:

1. **Rule 9, single-modifier siblings.** Two labels in one segment differing by
   exactly one leading word, that word not a preposition. `substantial savings
   measures` beside `the savings measures` is one entity in two KG nodes.
   The preposition exclusion matters: rule 2's D-22 amendment explicitly BLESSES
   `the finding of an excessive deficit` beside `an excessive deficit`, and a
   check that flags those condemns 44 correct labels.
2. **Rule 9, bare demonstratives and pronouns** left in a slot.
3. **Rule 2 / rule 7, chain connectivity** — a subject picking up nothing earlier
   in its own fact. Legitimate only where the fact declares `nolink`.
4. **Rule 5 vs rule 11, list cells** — how each flattened-table segment treats a
   cell holding a list. Two treatments of one construction is the finding.
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

PREPOSITIONS = {"of", "by", "to", "for", "in", "on", "under", "with", "from", "as",
                "at", "into", "over", "per"}
DEMONSTRATIVE = re.compile(r"^(that|this|those|these|such|it|they|its)\b", re.I)
ROW_SENTENCE = re.compile(r"In the .{0,40}? set out in this Article, row \d+:")


def norm(s: str) -> str:
    s = (s or "").strip().lower()
    for art in ("the ", "a ", "an "):
        if s.startswith(art):
            return s[len(art):]
    return s


def load(root: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for p in sorted(root.glob("*.json")):
        for seg in json.loads(p.read_text(encoding="utf-8"))["segments"]:
            out[seg["segment_id"]] = seg["facts"]
    if not out:
        raise SystemExit(f"no annotation under {root}")
    return out


def labels_of(facts: list[dict]) -> set[str]:
    out = set()
    for f in facts:
        for tr in f.get("triples") or []:
            out.add(tr.get("subject", ""))
            out.add(tr.get("object", ""))
    return {x for x in out if x}


def modifier_pairs(A: dict) -> tuple[list, set]:
    pairs, involved = [], set()
    for sid, facts in A.items():
        L = sorted(labels_of(facts))
        for i, a in enumerate(L):
            for b in L[i + 1:]:
                na, nb = norm(a), norm(b)
                if na == nb:
                    continue
                long_, short_ = (na, nb) if len(na) > len(nb) else (nb, na)
                if not long_.endswith(" " + short_):
                    continue
                head = long_[: -(len(short_) + 1)].split()
                if len(head) != 1 or head[0] in PREPOSITIONS:
                    continue
                pairs.append((sid, a, b))
                involved.add((sid, a))
                involved.add((sid, b))
    return pairs, involved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--facts-root", default="data/gold/facts_hires")
    ap.add_argument("--show", type=int, default=99, help="how many items to list")
    args = ap.parse_args()

    root = ROOT / args.facts_root
    A = load(root)
    n_facts = sum(len(v) for v in A.values())
    print(f"{args.facts_root}: {n_facts} facts over {len(A)} segments\n")

    # 1 -------------------------------------------------------------- rule 9 a
    pairs, involved = modifier_pairs(A)
    print(f"1. rule 9 — a label and a sibling differing by one non-preposition word")
    print(f"   {len(pairs)} pairs over {len(involved)} labels "
          f"in {len(set(s for s, _, _ in pairs))} segments")
    by_seg = Counter(s for s, _, _ in pairs)
    for sid, n in by_seg.most_common(4):
        print(f"     {sid:<44} {n}")
    for sid, a, b in pairs[: args.show]:
        print(f"       {sid.split('::')[1]:<22} {a!r} / {b!r}")

    # 2 -------------------------------------------------------------- rule 9 b
    dem = sorted({(sid, tr[slot]) for sid, facts in A.items() for f in facts
                  for tr in (f.get("triples") or []) for slot in ("subject", "object")
                  if DEMONSTRATIVE.match((tr.get(slot) or "").strip())})
    print(f"\n2. rule 9 — bare demonstrative or pronoun in a slot: {len(dem)} distinct")
    for sid, lab in dem[: args.show]:
        print(f"     {sid.split('::')[1]:<22} {lab!r}")
    print(f"\n   OUTSTANDING RULE-9 ITEMS: {len(involved)} + {len(dem)} = "
          f"{len(involved) + len(dem)} labels")

    # 3 ------------------------------------------------------------- chains
    broken, declared = [], 0
    for sid, facts in A.items():
        for f in facts:
            tr = f.get("triples") or []
            if f.get("nolink"):
                declared += 1
            for i, t in enumerate(tr[1:], start=1):
                earlier = {norm(x.get("object", "")) for x in tr[:i]}
                earlier |= {norm(x.get("subject", "")) for x in tr[:i]}
                s = norm(t.get("subject", ""))
                if s and not any(s == e or s in e for e in earlier):
                    broken.append((sid, f.get("gold_id", "?"), t["subject"],
                                   bool(f.get("nolink"))))
    undeclared = [b for b in broken if not b[3]]
    print(f"\n3. rule 2 / 7 — subject picking up nothing earlier: {len(broken)}"
          f"   ({declared} facts declare nolink; {len(undeclared)} undeclared)")
    for sid, gid, sub, _ in undeclared[: args.show]:
        print(f"     {gid:<44} {sub!r}")

    # 4 ------------------------------------------------------------- tables
    idx = gs.sample_index()
    print("\n4. rule 5 vs rule 11 — how a cell holding a list is treated")
    any_table = False
    for sid in sorted(A):
        m = idx.get(sid)
        if not m:
            continue
        text = gs.segment_text(m["doc_id"], m["section_path"],
                               m["char_start"], m["char_end"])
        if not ROW_SENTENCE.search(text):
            continue
        any_table = True
        facts = A[sid]
        quotes = sorted(len(gs.fold(f["source_quote"])) for f in facts)
        whole_row = sum(1 for f in facts if ROW_SENTENCE.match(gs.fold(f["source_quote"])))
        ranges = sum(1 for f in facts
                     if re.search(r"\b\d[\d ]{3,}\s+to\s+\d", f["natural_language"]))
        print(f"     {sid:<44} {len(facts):>2} facts · median quote "
              f"{quotes[len(quotes) // 2]:>3} chars · {whole_row} anchored on a whole row"
              + (f" · {ranges} compress a list into a range" if ranges else ""))
    if not any_table:
        print("     no flattened-table segment in this annotation")
    print("\n   Two treatments of one construction inside one artefact is the finding,\n"
          "   not a defect to fix mid-pass — see _RULES.md, amendment D-28.")


if __name__ == "__main__":
    main()
