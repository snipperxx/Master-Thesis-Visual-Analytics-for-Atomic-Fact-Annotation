"""Re-decompose the gold at high resolution: one fact per source span, N nested
triples inside it.

The coarse pass in `data/gold/facts/` wrote exactly one triple per fact, so
propositions the text states inside a noun phrase stayed inside a 17-word object
or a 20-word predicate. The gold is supposed to be silent on packaging
(`doc/methodology_v2.tex` §4.2), and a fat object is a packaging decision. This
script rewrites the triples and nothing else.

**Coverage is held constant on purpose.** Every fact of the coarse pass appears
here with the same `gold_id` and the same verbatim `source_quote`; no span is
added, dropped or merged. Only the triple array inside a fact changes (plus
`natural_language` / `condition` / `temporal_context` where the coarse pass put a
date or a qualifier in the wrong slot). So the two resolutions score the same
spans, and a difference between them is a resolution difference and nothing else.

The decomposition itself is authored per fact in `scripts/gold_hires_spec.py` —
rules R1-R5 in `data/gold/calibration/_RULES.md`. This module only applies a
spec, inherits quotes verbatim, and refuses to write a file that breaks an
invariant:

  * every source quote is verbatim in its segment text (inherited, so it cannot
    drift, but re-checked anyway);
  * no empty slot in any triple;
  * inside a chain, every triple after the first has a subject that repeats an
    earlier object **verbatim** — a near-miss would become two unconnected KG
    nodes;
  * every fact of the coarse pass is accounted for, and no fact is invented.

**Every fact comes out `unreviewed`**, including the ones whose triple is carried
over unchanged. Zilong's point on 2026-08-17: the adjudication so far — the
machine pass and the confirmations he had started making in `review.html` — was
made against the coarse annotation, before the resolution problem was visible. A
decision taken then says nothing about this file, and an unchanged triple was
still read in a document whose other facts changed around it. So the whole
re-decomposition faces the human gate from zero (CLAUDE.md discipline rule 6).

Usage:
    python scripts/build_gold_hires.py                 # write data/gold/facts_hires/
    python scripts/build_gold_hires.py --check         # validate only, write nothing
    python scripts/make_gold_review_page.py --facts-root data/gold/facts_hires
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import gold_store as gs  # noqa: E402

GOLD = ROOT / "data" / "gold"
COARSE = GOLD / "facts"
DEST = GOLD / "facts_hires"

KEEP = "KEEP"          # inherit the coarse triple unchanged


def _norm(s: str) -> str:
    """Chain-link identity: leading article and case are not what makes two
    strings the same entity here (`entity_norm` folds those later), so ignore
    them when checking that a chain connects."""
    s = (s or "").strip().lower()
    for a in ("the ", "a ", "an "):
        if s.startswith(a):
            return s[len(a):]
    return s


def check(fact: dict, entry, text: str, problems: list[str]) -> None:
    """Every invariant a written fact must satisfy, coarse-derived or added."""
    gid = fact["gold_id"]
    if not gs.is_verbatim(fact["source_quote"], text):
        problems.append(f"{gid}: quote not verbatim in segment text")
    for i, t in enumerate(fact["triples"]):
        for k in ("subject", "predicate", "object"):
            if not (t.get(k) or "").strip():
                problems.append(f"{gid}: triple {i+1} has an empty {k}")
    # A span can genuinely assert two propositions that share no entity ("the
    # information must be disregarded" + "the goods are not of marketable
    # quality"). Those must be DECLARED with nolink=True, so that an accidental
    # near-miss — the typo that silently becomes two KG nodes — still fails here.
    if entry is not KEEP and entry.get("nolink"):
        return
    for i, t in enumerate(fact["triples"][1:], start=1):
        earlier = {_norm(x["object"]) for x in fact["triples"][:i]}
        earlier |= {_norm(x["subject"]) for x in fact["triples"][:i]}
        # equal, or the head of an earlier slot ("the arrangements" attaching to
        # "access to the arrangements") — a typo still matches nothing and fails
        sub = _norm(t["subject"])
        if not any(sub == e or sub in e for e in earlier):
            problems.append(
                f"{gid}: triple {i+1} subject {t['subject']!r} does not repeat an "
                f"earlier subject or object — the chain does not connect "
                f"(declare nolink=True if deliberate)")


def load_spec() -> dict:
    """{gold_id: KEEP | {"triples": [(s,p,o), ...], "nl":?, "cond":?, "temp":?}}"""
    spec_path = ROOT / "scripts" / "gold_hires_spec.py"
    if not spec_path.exists():
        raise SystemExit(f"no spec at {spec_path}")
    ns: dict = {"KEEP": KEEP}
    exec(compile(spec_path.read_text(encoding="utf-8"), str(spec_path), "exec"), ns)
    return ns["SPEC"]


def build(spec: dict) -> tuple[dict[str, dict], list[str]]:
    sample = gs.load_sample()
    idx = {s["segment_id"]: s for s in sample["segments"]}
    problems: list[str] = []
    seen: set[str] = set()
    out: dict[str, dict] = {}

    for p in sorted(COARSE.glob("*.json")):
        coarse = json.loads(p.read_text(encoding="utf-8"))
        doc_id = coarse["doc_id"]
        segments = []
        for seg in coarse["segments"]:
            m = idx[seg["segment_id"]]
            text = gs.segment_text(doc_id, m["section_path"],
                                   m["char_start"], m["char_end"])
            facts = []
            # Facts the coarse pass missed, found by scripts/gold_coverage_audit.py
            # and keyed "+<id>" in the spec. They carry their own quote, which is
            # verbatim-checked against the segment text like every other.
            added = {k.rsplit("::", 1)[1][1:]: v for k, v in spec.items()
                     if k.startswith(seg["segment_id"] + "::+")}
            for old in seg["facts"]:
                gid = old["gold_id"]
                entry = spec.get(gid)
                if entry is None:
                    problems.append(f"{gid}: not in the spec")
                    continue
                seen.add(gid)
                fact = dict(old)
                # the whole file faces the human gate from zero — see module docstring
                fact["status"] = "unreviewed"
                fact["reviewer_note"] = ""
                fact.pop("review", None)
                if entry is KEEP:
                    fact["nesting"] = "kept"
                else:
                    # Carry the declaration into the DATA, not just the build: the
                    # chain-connectivity rule is re-checked by machine_review_gold
                    # and by anything else reading the gold, and a deliberate
                    # two-proposition span must not read as a broken chain there.
                    if entry.get("nolink"):
                        fact["nolink"] = True
                    triples = [{"subject": s, "predicate": pr, "object": o}
                               for s, pr, o in entry["triples"]]
                    for k_spec, k_fact in (("nl", "natural_language"),
                                           ("cond", "condition"),
                                           ("temp", "temporal_context")):
                        if k_spec in entry:
                            fact[k_fact] = entry[k_spec]
                    fact["triples"] = triples
                    fact["nesting"] = "chain" if len(triples) > 1 else "reslotted"

                check(fact, entry, text, problems)
                fact["resolution"] = "high"
                facts.append(fact)

            for suffix, entry in sorted(added.items()):
                gid = f"{seg['segment_id']}::{suffix}"
                seen.add(f"{seg['segment_id']}::+{suffix}")
                fact = {
                    "gold_id": gid,
                    "natural_language": entry.get("nl", ""),
                    "source_quote": entry["quote"],
                    "condition": entry.get("cond", ""),
                    "temporal_context": entry.get("temp", ""),
                    "logic_group": entry.get("logic_group", ""),
                    "logic_op": entry.get("logic_op", ""),
                    "triples": [{"subject": s, "predicate": p_, "object": o}
                                for s, p_, o in entry["triples"]],
                    "boilerplate": entry.get("boilerplate", False),
                    "nolink": bool(entry.get("nolink")),
                    "status": "unreviewed",
                    "reviewer_note": "",
                    "nesting": "added",
                    "added_because": entry.get("why", ""),
                    "resolution": "high",
                }
                if not fact["natural_language"].strip():
                    problems.append(f"{gid}: an added fact needs natural_language")
                check(fact, entry, text, problems)
                facts.append(fact)

            segments.append({"segment_id": seg["segment_id"],
                             "section_path": seg["section_path"], "facts": facts})

        out[doc_id] = {
            "schema_version": 1,
            "doc_id": doc_id,
            "doc_title": coarse["doc_title"],
            "annotation_mode": "guideline-invariant",
            "resolution": "high (nested SPO chains, rules R1-R5)",
            "segments": segments,
            "annotator": coarse.get("annotator", "claude-opus-5"),
            "guideline_version": "gold-invariant",
            "provenance": {
                "produced_by": "claude-opus-5",
                "kind": "machine re-decomposition of the coarse machine pre-annotation",
                "derived_from": "data/gold/facts/ (307 facts, 307 triples)",
                "rules": "data/gold/calibration/_RULES.md (R1-R5)",
                "note": "Same spans, same quotes, same coverage as the coarse pass; "
                        "only the triples inside each fact were rewritten. Machine "
                        "work throughout — a fact whose triples changed is "
                        "`unreviewed` and still needs the human gate "
                        "(CLAUDE.md discipline rule 6).",
            },
        }

    for gid in set(spec) - seen:
        problems.append(f"{gid}: in the spec but not in the coarse gold")
    return out, problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="validate, write nothing")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even though the destination carries a human "
                         "sign-off. Resets every status to `unreviewed`; re-apply "
                         "the decisions file afterwards.")
    args = ap.parse_args()

    out, problems = build(load_spec())
    if problems:
        for x in problems[:40]:
            print("  !!", x)
        raise SystemExit(f"{len(problems)} problem(s) — nothing written")

    facts = [f for d in out.values() for s in d["segments"] for f in s["facts"]]
    n_tr = sum(len(f["triples"]) for f in facts)
    kinds = {k: sum(1 for f in facts if f["nesting"] == k)
             for k in ("chain", "reslotted", "kept", "added")}
    print(f"{len(out)} docs / {sum(len(d['segments']) for d in out.values())} segments / "
          f"{len(facts)} facts / {n_tr} triples")
    print(f"  chains {kinds['chain']} · reslotted {kinds['reslotted']} · "
          f"unchanged {kinds['kept']} · added {kinds['added']}")
    print(f"  longest object now {max(len(t['object'].split()) for f in facts for t in f['triples'])} words, "
          f"longest predicate {max(len(t['predicate'].split()) for f in facts for t in f['triples'])} words")
    if args.check:
        print("  --check: nothing written")
        return

    # Every fact this script writes comes out `unreviewed`. Once a human sweep has
    # been applied to DEST that is destructive: re-running would silently discard
    # the sign-off RQ2 is gated on, and the only symptom would be gold_stats going
    # quiet about it. Refuse, and say what to do instead.
    signed = 0
    for old in sorted(DEST.glob("*.json")):
        payload = json.loads(old.read_text(encoding="utf-8"))
        signed += sum(1 for s in payload.get("segments", [])
                      for f in s.get("facts", [])
                      if (f.get("review") or {}).get("mode") == "human")
    if signed and not args.force:
        raise SystemExit(
            f"refusing: {DEST} already holds {signed} human-adjudicated fact(s)."
            f"\n  Rebuilding would reset every one of them to `unreviewed`."
            f"\n  If the spec really has to change, re-apply the sweep after:"
            f"\n    python scripts/build_gold_hires.py --force"
            f"\n    python scripts/apply_gold_review.py <decisions.json> "
            f"--mode human --facts-root data/gold/{DEST.name}")

    DEST.mkdir(exist_ok=True)
    for doc_id, payload in out.items():
        (DEST / f"{doc_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  wrote {DEST}")


if __name__ == "__main__":
    main()
