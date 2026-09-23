"""Fold review decisions back into the gold files.

    python scripts/apply_gold_review.py data/gold/review_decisions.json
    python scripts/apply_gold_review.py data/gold/decisions --mode machine \
        --reviewer claude-opus-5

Accepted facts keep their content; edited facts take the reviewer's slot text;
rejected facts stay in the file with status 'rejected' (they are excluded from
scoring but kept as evidence of what the pre-annotation got wrong); facts the
reviewer added are appended.

Two reviewer modes, and the difference is the whole point of the field:

    --mode human    a person adjudicated. Sets provenance.kind to
                    "machine-preannotation, human-adjudicated" — the phrase the
                    thesis is allowed to use, and the gate RQ2 waits on.
    --mode machine  a model adjudicated. Sets
                    "machine-preannotation, machine-adjudicated". Downstream
                    consumes this exactly like the human form; the difference is
                    only what `src.gold_eurlex_eval` prints on the report, since
                    the pre-annotator and the adjudicator being the same model is
                    a fact a reader of the numbers needs.

A human pass applied on top of a machine pass overwrites both status and
provenance, so the upgrade path is just: run it again with --mode human.

**Per-fact, not per-run (since 2026-08-09).** `review.html` downloads every
entry, confirmed or not, and marks the ones still carried over from the machine
pass with `src: "machine"`. A `--mode human` run honours that flag per fact, so
confirming 200 of 306 records 200 as human and 106 as machine, and the affected
documents get `provenance.kind = "...PARTIALLY human-adjudicated..."`. Before
this, one uniform reviewer was stamped over the whole file and a partial sweep
would have claimed a human signed off on facts nobody had opened.

The decisions file (or a directory of them, one per document):

    {"reviewed_on": ..., "reviewer": ..., "mode": ...,
     "decisions": {"<gold_id>": {"status": "accepted|edited|rejected",
                                 "note": ..., "rule": "D-02",
                                 "edits": {"subject": ..., "condition": ...}}},
     "added": [{"segment_id": ..., "fact": {...}}]}
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FACTS = ROOT / "data" / "gold" / "facts"

PROVENANCE_KIND = {
    "human": "machine-preannotation, human-adjudicated",
    "machine": "machine-preannotation, machine-adjudicated",
    # A `--mode human` run over a decisions file that still holds machine
    # entries. `review.html` downloads ALL entries, confirmed or not, so this is
    # the normal state of a partly-finished sweep — and stamping it with the
    # human phrase would attribute to a person facts nobody looked at.
    "mixed": ("machine-preannotation, PARTIALLY human-adjudicated "
              "(remainder machine-adjudicated)"),
}


def _entry_mode(d: dict, run_mode: str) -> str:
    """Provenance mode for ONE decision.

    `review.html` marks every entry it carries over from the machine pass with
    `src: "machine"` and clears that flag when a human confirms or overrides it.
    Honouring it per fact is what keeps a partial sweep honest: before 2026-08-09
    `--mode human` stamped one uniform reviewer over every entry in the file, so
    confirming 200 of 306 and applying would have recorded all 306 as
    human-adjudicated (discipline rule 6)."""
    if run_mode != "human":
        return run_mode
    return "machine" if (d or {}).get("src") == "machine" else "human"

# Edits addressed to the fact record itself rather than to its first triple.
# Before this list existed every unknown key was written into triples[0], so an
# edit to `condition` silently created a bogus triple slot.
FACT_LEVEL = {"natural_language", "source_quote", "condition", "temporal_context",
              "logic_group", "logic_op", "boilerplate", "reviewer_note"}
TRIPLE_LEVEL = {"subject", "predicate", "object"}


def load_decisions(path: Path) -> tuple[dict, list, dict]:
    """One file, or a directory of per-document files, merged."""
    paths = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not paths:
        raise SystemExit(f"no decision files under {path}")
    decisions: dict = {}
    added: list = []
    meta: dict = {}
    for p in paths:
        payload = json.loads(p.read_text(encoding="utf-8"))
        for gid, d in payload["decisions"].items():
            if gid in decisions and decisions[gid] != d:
                raise SystemExit(f"conflicting decisions for {gid} in {p}")
            decisions[gid] = d
        added += payload.get("added", [])
        meta = {**meta, **{k: v for k, v in payload.items()
                           if k not in ("decisions", "added")}}
    return decisions, added, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("decisions", type=Path,
                    help="a decisions JSON file, or a directory of them")
    ap.add_argument("--mode", choices=sorted(PROVENANCE_KIND), default="human")
    ap.add_argument("--reviewer", default=None,
                    help="who decided; defaults to the file's `reviewer` field")
    ap.add_argument("--facts-root", type=Path, default=FACTS,
                    help="gold fact files to write (default data/gold/facts). "
                         "Must match the `facts_root` the decisions file was "
                         "swept from - the guard below compares them.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-human", action="store_true",
                    help="apply --mode human to a decisions file that carries no "
                         "human confirmations at all. Only correct if the sign-off "
                         "happened outside review.html; it is otherwise the "
                         "mistake the guard below exists to catch.")
    args = ap.parse_args()

    dec, added, meta = load_decisions(args.decisions)

    # The machine artefact in data/gold/decisions/ has file-level mode "machine"
    # and no per-entry `src`, so every entry would read as human-confirmed.
    # Pointing --mode human at it is an easy mistake — the documented workflow is
    # "confirm in review.html, then re-apply with --mode human", and the two
    # inputs live one directory apart.
    # `review.html` stamps its download `mode: "human"`; the machine pass wrote
    # `mode: "machine"`. That file-level field is the only reliable signal —
    # a confirmed entry and a machine-artefact entry both simply lack `src`.
    if (args.mode == "human" and meta.get("mode") == "machine"
            and not args.force_human):
        raise SystemExit(
            f"refusing: --mode human, but {args.decisions} declares "
            f"mode={meta.get('mode')!r} (reviewer={meta.get('reviewer')!r}) and "
            f"holds {len(dec)} decisions.\n"
            f"That is the machine artefact, not a sweep of review.html. Applying "
            f"it would record every fact as human-adjudicated and falsify the "
            f"gold's provenance (CLAUDE.md discipline rule 6).\n"
            f"  Correct workflow: open data/gold/review.html, confirm or override "
            f"each fact, click Download decisions, then apply THAT file.\n"
            f"  If the sign-off genuinely happened elsewhere, pass --force-human.")
    # A decisions file records which annotation it was swept from, and it has to
    # be written back to that same annotation. The gold has more than one root:
    # `facts` (coarse) and `facts_hires` (D-18) describe the same 59 segments with
    # the same gold_ids, so a cross-applied sweep does not fail, it silently
    # records decisions about triples nobody looked at (the same shape of fault as
    # BUG-2026-08-17-A). The second annotator's blind pass has its own applier and
    # must never land here at all.
    target = args.facts_root.resolve()
    target_rel = target.name
    root = meta.get("facts_root")
    if root == "annotator_b" or target_rel == "annotator_b":
        raise SystemExit(
            f"refusing: {args.decisions} concerns the second annotator's blind "
            f"pass.\nUse scripts/apply_second_annotation_review.py for that.")
    if root and root != target_rel:
        raise SystemExit(
            f"refusing: {args.decisions} was swept from data/gold/{root}, but "
            f"--facts-root points at data/gold/{target_rel}.\nApplying it would "
            f"write one annotation's decisions into another.\nPass "
            f"--facts-root data/gold/{root}.")
    if not target.is_dir():
        raise SystemExit(f"refusing: {target} is not a directory")

    reviewer = args.reviewer or meta.get("reviewer") or ("model" if args.mode == "machine" else "annotator")
    on = str(date.today())
    stats: Counter = Counter()
    unknown_edits: set[str] = set()

    mode_counts: Counter = Counter()

    def stamp(f: dict, d: dict) -> None:
        m = _entry_mode(d, args.mode)
        mode_counts[m] += 1
        # A machine-carried entry keeps the machine adjudicator's name even in a
        # human run; only what the human actually touched gets their name.
        by = (d or {}).get("by") or reviewer if m == "machine" else reviewer
        f["review"] = {"by": by, "mode": m, "on": on}
        if d.get("rule"):
            f["review"]["rule"] = d["rule"]

    for p in sorted(target.glob("*.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        touched = False
        for seg in doc["segments"]:
            for f in seg["facts"]:
                d = dec.get(f["gold_id"])
                if not d:
                    continue
                stats[d["status"]] += 1
                touched = True
                f["status"] = d["status"]
                f["reviewer_note"] = d.get("note", "")
                stamp(f, d)
                for k, v in (d.get("edits") or {}).items():
                    if v is None or v == "":
                        continue
                    if k in FACT_LEVEL:
                        f[k] = v
                    elif k in TRIPLE_LEVEL:
                        f["triples"][0][k] = v
                    elif k == "triples":
                        f["triples"] = v
                    else:
                        unknown_edits.add(k)
            for a in added:
                if a["segment_id"] == seg["segment_id"] and \
                   not any(x["gold_id"] == a["fact"]["gold_id"] for x in seg["facts"]):
                    fact = dict(a["fact"])
                    fact.setdefault("status", "edited")
                    fact["provenance_override"] = f"written by reviewer ({reviewer})"
                    stamp(fact, a.get("fact", {}))
                    seg["facts"].append(fact)
                    stats["added"] += 1
                    touched = True
        if touched and not args.dry_run:
            # Per-document, from what this document's facts actually received —
            # a document whose every fact is still machine-carried must not
            # inherit the human phrase from a sibling document.
            modes = {f.get("review", {}).get("mode")
                     for seg in doc["segments"] for f in seg["facts"]
                     if f.get("review")}
            doc_mode = ("mixed" if modes == {"human", "machine"}
                        else ("human" if modes == {"human"} else "machine"))
            doc["provenance"]["kind"] = PROVENANCE_KIND[doc_mode]
            doc["provenance"]["adjudicated_by"] = reviewer
            doc["provenance"]["adjudicated_on"] = on
            doc["provenance"]["adjudication_mode"] = doc_mode
            p.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")

    print(("DRY RUN " if args.dry_run else "") +
          f"applied ({args.mode}, {reviewer}):", dict(stats))
    if args.mode == "human" and mode_counts["machine"]:
        n_h, n_m = mode_counts["human"], mode_counts["machine"]
        print(f"  !! PARTIAL human pass: {n_h} fact(s) carry your sign-off, "
              f"{n_m} are still machine-adjudicated and were recorded as such.")
        print(f"     Those {n_m} were NOT confirmed in review.html. The affected "
              f"documents get provenance.kind = {PROVENANCE_KIND['mixed']!r},")
        print( "     so the thesis may not yet call this gold human-adjudicated. "
               "Finish the sweep and re-apply.")
    if unknown_edits:
        print("  !! ignored unknown edit fields:", sorted(unknown_edits))
    seen = sum(stats[k] for k in ("accepted", "edited", "rejected"))
    if seen != len(dec):
        print(f"  !! {len(dec) - seen} decision(s) matched no gold fact")

    from src import gold_store as gs
    rep = gs.validate_all()
    print("validation:", "OK" if rep.ok else f"{len(rep.errors)} error(s)")
    for e in rep.errors[:10]:
        print("  ", e)


if __name__ == "__main__":
    main()
