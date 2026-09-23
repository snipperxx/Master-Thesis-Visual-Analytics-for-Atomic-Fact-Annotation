"""Validate a second-annotator bundle into `data/gold/annotator_b/`.

    python -m scripts.import_second_annotation data/gold/second_annotation_bundle.json \\
        --annotator gemini-3-pro

The bundle comes from `data/gold/second_annotator.html` (built by
`scripts/make_second_annotator_console.py`): one blind annotation per segment of
the frozen sample. This writes it out in the same fact schema as the gold, so the
two annotations are directly comparable objects — one file per document under
`data/gold/annotator_b/`.

Scope, since 2026-08-18: **any of the 59 sampled segments**, not only the 9 of the
IAA subsample. Those 9 were drawn to bound a human annotator's hours; the
2026-08-17 decision makes the second annotator a model, and 9 segments carry just
24 gold facts (7.7 % of the annotation mass), too thin for the agreement number
they were meant to support. Every segment records `iaa: true|false`, so the
subsample stays reportable on its own.

Nothing is written unless every check passes:

* the bundle covers only segments of the frozen 59-segment sample, and no segment twice
* every `source_quote` is verbatim in that segment's text (the invariant the whole
  gold is anchored on)
* every fact has a natural_language, at least one triple, and no empty slot
* `logic_op`, where present, is AND / OR / XOR

Chain connectivity is reported as a warning, not an error: the second annotator's
job is to annotate independently, and a disconnected chain is a finding about the
protocol, not a malformed file.

**Provenance.** `--annotator` names who produced it and `--mode` records whether
that was a human or a machine; both go into every fact. A machine second pass
bounds how reproducibly the protocol can be applied — it is not agreement between
two independent human annotators, and `doc/methodology_v2.tex` §4.5 has to say
which of the two the thesis reports.
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
DEST = GOLD / "annotator_b"
_LOGIC_OPS = {"", "AND", "OR", "XOR"}


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    for a in ("the ", "a ", "an "):
        if s.startswith(a):
            return s[len(a):]
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--annotator", required=True,
                    help="who annotated, e.g. gemini-3-pro or human:initials")
    ap.add_argument("--mode", choices=("machine", "human"), default="machine")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    sample = gs.load_sample()
    idx = {s["segment_id"]: s for s in sample["segments"]}
    iaa = list(sample["iaa_subsample"]["segment_ids"])

    problems: list[str] = []
    warnings: list[str] = []
    by_doc: dict[str, list[dict]] = {}
    seen: set[str] = set()
    versions: set[str] = set()

    for entry in bundle.get("segments", []):
        sid = entry.get("segment_id")
        if sid not in idx:
            problems.append(f"{sid}: not in the frozen 59-segment sample")
            continue
        if sid in seen:
            problems.append(f"{sid}: appears twice in the bundle")
            continue
        seen.add(sid)
        if entry.get("facts") is None:
            continue                      # not annotated yet — reported at the end
        prompt_version = (entry.get("prompt_version")
                          or bundle.get("prompt_version") or "unstamped")
        versions.add(prompt_version)
        m = idx[sid]
        text = gs.segment_text(m["doc_id"], m["section_path"],
                              m["char_start"], m["char_end"])
        facts = []
        for k, f in enumerate(entry["facts"], start=1):
            gid = f"{sid}::b{k:02d}"
            if not (f.get("natural_language") or "").strip():
                problems.append(f"{gid}: empty natural_language")
            if not gs.is_verbatim(f.get("source_quote", ""), text):
                problems.append(f"{gid}: source_quote not verbatim — "
                                f"{(f.get('source_quote') or '')[:70]!r}")
            triples = f.get("triples") or []
            if not triples:
                problems.append(f"{gid}: no triples")
            for ti, t in enumerate(triples, start=1):
                for slot in ("subject", "predicate", "object"):
                    if not ((t or {}).get(slot) or "").strip():
                        problems.append(f"{gid}: triple {ti} has an empty {slot}")
            if (f.get("logic_op") or "") not in _LOGIC_OPS:
                problems.append(f"{gid}: logic_op {f.get('logic_op')!r} is not AND/OR/XOR")
            for ti, t in enumerate(triples[1:], start=1):
                earlier = {_norm(x.get("object", "")) for x in triples[:ti]}
                earlier |= {_norm(x.get("subject", "")) for x in triples[:ti]}
                sub = _norm((t or {}).get("subject", ""))
                if sub and not any(sub == e or sub in e for e in earlier):
                    warnings.append(f"{gid}: triple {ti+1} subject "
                                    f"{t['subject']!r} picks up nothing earlier")
            facts.append({
                "gold_id": gid,
                "natural_language": f.get("natural_language", ""),
                "source_quote": f.get("source_quote", ""),
                "condition": f.get("condition", "") or "",
                "temporal_context": f.get("temporal_context", "") or "",
                "logic_group": f.get("logic_group", "") or "",
                "logic_op": f.get("logic_op", "") or "",
                "triples": [{"subject": t.get("subject", ""),
                             "predicate": t.get("predicate", ""),
                             "object": t.get("object", "")} for t in triples],
                "boilerplate": bool(f.get("boilerplate", False)),
                "status": "unreviewed",
                "reviewer_note": "",
                "resolution": "high",
                # Which guideline the annotator was actually holding. v2 (2026-08-18)
                # repaired a contradiction between R1 and R1b and stated two rules the
                # reviewer had always enforced, so a v1 fact and a v2 fact are not the
                # same measurement.
                "annotated_by": {"by": args.annotator, "mode": args.mode,
                                 "prompt_version": prompt_version},
            })
        by_doc.setdefault(m["doc_id"], []).append(
            {"segment_id": sid, "section_path": m["section_path"],
             "iaa": sid in iaa, "facts": facts})

    annotated = {seg["segment_id"] for segs in by_doc.values() for seg in segs}
    missing = [s for s in sorted(idx) if s not in annotated]
    missing_iaa = [s for s in sorted(iaa) if s not in annotated]
    # What the file covers, named for the provenance block: the two are scored
    # separately, so "all 59" and "the 9" must not both be called the same pass.
    scope = ("the frozen 59-segment sample" if annotated - set(iaa)
             else "the frozen IAA subsample")

    if len(versions) > 1:
        warnings.append("the bundle mixes guideline versions: "
                        + ", ".join(sorted(versions))
                        + " — those segments are not one condition, and §4.5 has to "
                          "say which was annotated under which")

    if problems:
        for x in problems[:40]:
            print("  !!", x)
        raise SystemExit(f"{len(problems)} problem(s) — nothing written")
    for w in warnings[:20]:
        print("  ~", w)

    n_facts = sum(len(seg["facts"]) for segs in by_doc.values() for seg in segs)
    n_tr = sum(len(f["triples"]) for segs in by_doc.values()
               for seg in segs for f in seg["facts"])
    print(f"{len(annotated)}/{len(idx)} segments "
          f"({len(iaa) - len(missing_iaa)}/{len(iaa)} of the IAA subsample), "
          f"{n_facts} facts, {n_tr} triples, by {args.annotator} ({args.mode})")
    if missing:
        print(f"  still to annotate: {len(missing)} segment(s)")
        if missing_iaa:
            print(f"    of the IAA subsample: {', '.join(missing_iaa)}")
    if args.dry_run:
        print("  --dry-run: nothing written")
        return

    DEST.mkdir(exist_ok=True)
    for doc_id, segs in sorted(by_doc.items()):
        titles = [idx[s["segment_id"]]["doc_title"] for s in segs]
        (DEST / f"{doc_id}.json").write_text(json.dumps({
            "schema_version": 1,
            "doc_id": doc_id,
            "doc_title": titles[0],
            "annotation_mode": "guideline-invariant",
            "resolution": "high (nested SPO chains, rules R1-R6)",
            "segments": sorted(segs, key=lambda s: s["segment_id"]),
            "annotator": args.annotator,
            "guideline_version": "gold-invariant",
            "prompt_versions": sorted(versions),
            "provenance": {
                "produced_by": args.annotator,
                "kind": f"independent second annotation ({args.mode}), blind",
                "pass": f"second annotator over {scope}",
                "note": "Annotated from the segment text alone, without sight of the "
                        "first pass, via data/gold/second_annotator.html. Used for "
                        "annotator-vs-annotator agreement only; it is not part of "
                        "the gold and must never be merged into data/gold/facts/. "
                        "`iaa` on each segment marks the frozen 9-segment subsample, "
                        "which is reported separately from the full sample.",
            },
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  wrote {DEST}  ({len(by_doc)} document file(s))")
    print("  next: annotator-vs-annotator P/R/F1 with the soft matcher — "
          "gold_compare.compare_soft is many-to-many since 2026-09-23 "
          "(methodology §4.5, §6.2); not yet wired to this annotation")


if __name__ == "__main__":
    main()
