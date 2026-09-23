"""RQ2 scoring: model extractions against the EUR-Lex gold.

`gold_eval.py` scores against `parsed["gold_triples"]`, which only WebNLG and
Re-DocRED carry. The EUR-Lex gold is a different shape and needs a different
driver, for one reason that dominates everything else:

**The gold covers 59 sampled segments, not whole documents.** A model run
produces facts for every segment of all 12 evaluation documents. Scoring those
against a gold that only knows 59 of them would count every fact from an
unsampled segment as a false positive and collapse precision for a reason that
has nothing to do with the model. So comparison is **segment-scoped**: for each
sampled segment, the gold facts of that segment are matched against the model
facts anchored to that segment, and everything else is set aside and counted
separately as `out_of_scope`. Out-of-scope facts are neither right nor wrong
here — nobody annotated the segments they came from.

Model facts are anchored by `source_locator.section_path`, the same key the
gold sample uses, so the scoping is exact rather than heuristic.

What is held fixed (CLAUDE.md, "Non-negotiable experimental discipline"):
soft slot gates subject/object >= 0.70, predicate >= 0.45, overall >= 0.62,
taken from `gold_compare.DEFAULT_THRESHOLDS`. Boilerplate gold facts are
excluded by default (decision log D-03); `--include-boilerplate` checks that the
choice does not drive the result.

**Units and counting (many-to-many coverage, design of record since
2026-09-23).** Recall is counted over gold *triples* (a nested gold fact of N
triples is N units) and precision over predicted *triples*, each deduplicated
after normalisation within its segment. Every (gold, prediction) pair that
clears the match criterion is a link; a gold triple with >= 1 link is
*covered*, a prediction with >= 1 link is *supported*. Because these are two
different counts, every row and every aggregate carries both:

    precision = n_pred_supported / n_extracted
    recall    = n_gold_covered   / n_gold

Micro figures sum the four counts over rows before dividing; macro F1 is the
mean of per-document F1 (each document's segments pooled first), as the design
states; the per-segment mean is reported separately as `*_segment_macro_f1`. The greedy one-to-one matcher used before 2026-09-23 is
available only as `--assignment one_to_one`, an ablation. The mode is recorded
in the result as `assignment` and printed in the report, like
`gold_facts_root`. Strict matching is assignment-invariant (see
`gold_compare.compare`).

The encoder backend is reported on every result. **A TF-IDF number is dev-only
and must never be reported**; the runner refuses to write a report under the
fallback backend unless `--allow-tfidf` is passed.

CLI:
    python -m src.gold_eurlex_eval
    python -m src.gold_eurlex_eval --version v2 --out runs/rq2_v2.json \
        --report runs/rq2_v2_errors.md
    python -m src.gold_eurlex_eval --assignment one_to_one   # ablation

Without `--out` / `--report` the files are `runs/rq2_eurlex_<version>.json` and
`..._errors.md`, with `__one_to_one` appended for the ablation so that it can
never overwrite the result of record.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

from src import gold_compare as gc
from src import gold_store as gs

ROOT = Path(__file__).resolve().parent.parent
DOC_LIST = ROOT / "data" / "doc_lists" / "eurlex_eval.json"

# The gold root of record (D-18): 312 facts, human-adjudicated 2026-08-31.
# data/gold/facts is the coarse root it superseded and is still only
# machine-adjudicated; the two share gold_ids, so pointing at the wrong one
# yields numbers instead of an error (BUG-2026-08-17-A).
DEFAULT_GOLD_FACTS = gs.GOLD_FACTS_HIRES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def eval_doc_ids(doc_list: Path = DOC_LIST) -> list[str]:
    payload = json.loads(doc_list.read_text(encoding="utf-8"))
    return [d["doc_id"] for d in payload["docs"]]


def gold_by_segment(doc_id: str, include_boilerplate: bool = False,
                    facts_root: Path | None = None) -> dict[str, list[dict]]:
    """Sampled segment -> its gold triples, in the flat shape gold_compare wants."""
    out: dict[str, list[dict]] = defaultdict(list)
    for f in gs.load_gold_facts(doc_id, include_boilerplate=include_boilerplate,
                                facts_root=facts_root or DEFAULT_GOLD_FACTS):
        for t in f["triples"]:
            out[f["section_path"]].append({
                "subject": t["subject"], "predicate": t["predicate"], "object": t["object"],
                "condition": f.get("condition", ""),
                "temporal_context": f.get("temporal_context", ""),
                "gold_id": f["gold_id"], "segment_id": f["segment_id"],
                "source_quote": f["source_quote"],
            })
    return dict(out)


def model_facts_by_segment(doc_id: str, version: str = "v1") -> dict[str, dict[str, list[dict]]]:
    """annotator -> section_path -> facts. Facts with no anchor land under ''."""
    from scripts.run_phase2 import discover_annotators
    from src.reextract_worker import facts_root_for_version

    root = facts_root_for_version(version or "v1")
    if not root.exists():
        return {}
    out: dict[str, dict[str, list[dict]]] = {}
    # `annotators=None`: take every annotator on disk. Scoring is PER
    # annotator (one row each), so an extra one adds a row rather than
    # contaminating a shared structure — unlike Phase-2 graph construction,
    # whose default filter is the frozen set. Subsetting here is the caller's
    # job, via `evaluate(annotators=...)` / `--annotators`.
    for annotator, facts in discover_annotators(root, doc_id,
                                                annotators=None).items():
        by_sec: dict[str, list[dict]] = defaultdict(list)
        for f in facts:
            sec = ((f.get("source_locator") or {}).get("section_path")) or ""
            by_sec[sec].append(f)
        out[annotator] = dict(by_sec)
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _rate(n_pred_supported: int, n_extracted: int,
          n_gold_covered: int, n_gold: int) -> dict:
    """P/R/F1 from the two separate numerators many-to-many coverage needs.

    Under many-to-many one prediction can cover several gold triples and one
    gold triple can be covered by several predictions, so there is no single
    "true positive" count: precision's numerator (supported predictions) and
    recall's (covered gold triples) are different numbers. The old
    tp/fp/fn signature computed both from one `n_matched`, which was only
    correct under one-to-one; it is gone so that nothing can quietly go back to
    it. Under the one_to_one ablation the two numerators are equal and the
    figures are the old ones.
    """
    p = n_pred_supported / n_extracted if n_extracted else 0.0
    r = n_gold_covered / n_gold if n_gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "n_extracted": n_extracted, "n_pred_supported": n_pred_supported,
            "n_gold": n_gold, "n_gold_covered": n_gold_covered,
            "n_spurious": n_extracted - n_pred_supported,
            "n_missing": n_gold - n_gold_covered}


# Fan-out counts: how much of the credit came from one prediction covering
# several gold triples, or several predictions covering one. Summed when the
# rows carry them (soft does; strict cannot fan out).
_FANOUT_KEYS = ("n_links", "n_pred_multi_gold", "n_gold_multi_pred")


def _accumulate(rows: list[dict], kind: str) -> dict:
    # Read the explicit numerators, never `n_matched`: under many-to-many that
    # key is the covered-gold count and would make precision wrong.
    out = _rate(sum(r[kind]["n_pred_supported"] for r in rows),
                sum(r[kind]["n_extracted"] for r in rows),
                sum(r[kind]["n_gold_covered"] for r in rows),
                sum(r[kind]["n_gold"] for r in rows))
    for k in _FANOUT_KEYS:
        if rows and all(k in r[kind] for r in rows):
            out[k] = sum(r[kind][k] for r in rows)
    return out


def _macro_f1(rows: list[dict], kind: str) -> float:
    """Macro F1 as pre-registered: the mean of per-DOCUMENT F1.

    Each document's segments are pooled first (micro within the document), then
    the documents are averaged. Until 2026-09-23 this averaged per-segment rows,
    which the design never said and which lets a one-triple segment weigh as
    much as a 38-fact one; the per-segment mean survives as
    `_segment_macro_f1`, reported under its own name.
    """
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_doc[r["doc_id"]].append(r)
    if not by_doc:
        return 0.0
    f1s = [_accumulate(drows, kind)["f1"] for drows in by_doc.values()]
    return round(sum(f1s) / len(f1s), 4)


def _segment_macro_f1(rows: list[dict], kind: str) -> float:
    """Mean of per-segment F1 (the pre-2026-09-23 'macro'; not the design's)."""
    return round(sum(r[kind]["f1"] for r in rows) / len(rows), 4) if rows else 0.0


def evaluate(doc_ids: list[str] | None = None, *, version: str = "v1",
             annotators: list[str] | None = None,
             include_boilerplate: bool = False,
             thresholds: dict | None = None,
             facts_root: Path | None = None,
             assignment: str = gc.DEFAULT_ASSIGNMENT,
             max_error_segments: int = 60) -> dict:
    """Score every annotator on every sampled gold segment of `doc_ids`.

    `assignment` selects how soft links become credit (module docstring):
    "many_to_many" is the measure of record, "one_to_one" the ablation.
    """
    gc.check_assignment(assignment)
    facts_root = facts_root or DEFAULT_GOLD_FACTS
    doc_ids = doc_ids or eval_doc_ids()
    rows: list[dict] = []           # one row per (doc, annotator, segment)
    errors: list[dict] = []
    reasons: Counter = Counter()
    out_of_scope: Counter = Counter()
    unanchored: Counter = Counter()
    skipped: list[str] = []
    backend = None

    for doc_id in doc_ids:
        gold_secs = gold_by_segment(doc_id, include_boilerplate, facts_root)
        if not gold_secs:
            skipped.append(f"{doc_id}: no gold facts")
            continue
        per_annotator = model_facts_by_segment(doc_id, version)
        if annotators:
            per_annotator = {a: v for a, v in per_annotator.items() if a in annotators}
        if not per_annotator:
            skipped.append(f"{doc_id}: no model facts (version={version})")
            continue

        for ann, by_sec in sorted(per_annotator.items()):
            unanchored[ann] += len(by_sec.get("", []))
            out_of_scope[ann] += sum(len(v) for k, v in by_sec.items()
                                     if k and k not in gold_secs)
            for section_path, gold in sorted(gold_secs.items()):
                facts = by_sec.get(section_path, [])
                strict = gc.compare(gold, facts)
                soft = gc.compare_soft(gold, facts, thresholds,
                                       assignment=assignment)
                backend = soft["backend"]
                rows.append({"doc_id": doc_id, "annotator": ann,
                             "segment_id": f"{doc_id}::{section_path}",
                             "section_path": section_path,
                             "strict": strict["metrics"], "soft": soft["metrics"]})
                for m in soft["missing"]:
                    reasons[f"missing:{m.get('reason', '?')}"] += 1
                for s in soft["spurious"]:
                    reasons[f"spurious:{s.get('reason', '?')}"] += 1
                if (soft["missing"] or soft["spurious"]) and len(errors) < max_error_segments:
                    errors.append({"doc_id": doc_id, "annotator": ann,
                                   "segment_id": f"{doc_id}::{section_path}",
                                   "missing": soft["missing"],
                                   "spurious": soft["spurious"]})

    by_ann: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_ann[r["annotator"]].append(r)

    summary = {}
    for ann, arows in sorted(by_ann.items()):
        scored = sum(r["soft"]["n_extracted"] for r in arows)
        summary[ann] = {
            "n_docs": len({r["doc_id"] for r in arows}),
            "n_segments": len(arows),
            "n_facts_scored": scored,
            "n_facts_out_of_scope": out_of_scope[ann],
            "n_facts_unanchored": unanchored[ann],
            "strict_micro": _accumulate(arows, "strict"),
            "soft_micro": _accumulate(arows, "soft"),
            "strict_macro_f1": _macro_f1(arows, "strict"),
            "soft_macro_f1": _macro_f1(arows, "soft"),
            "macro_unit": "document",
            "strict_segment_macro_f1": _segment_macro_f1(arows, "strict"),
            "soft_segment_macro_f1": _segment_macro_f1(arows, "soft"),
        }

    gold_status = _gold_status(facts_root)
    return {"corpus": "eurlex", "split": "eval", "version": version,
            "backend": backend, "scope": "sampled segments only",
            "include_boilerplate": include_boilerplate,
            "thresholds": thresholds or gc.DEFAULT_THRESHOLDS,
            # How soft links became credit. Check it before comparing two
            # result files, exactly as gold_facts_root is checked: the
            # one_to_one ablation yields plausible, lower numbers, not an error.
            "assignment": assignment,
            "unit": {"recall": "gold triple (deduplicated per segment)",
                     "precision": "predicted triple (deduplicated per segment)"},
            "gold_facts_root": _display_path(facts_root),
            "gold_status": gold_status,
            "reportable": gold_status["adjudication_mode"] == "human" and backend == "sbert",
            "n_docs_requested": len(doc_ids),
            "summary": summary, "rows": rows, "errors": errors,
            "error_reasons": dict(reasons.most_common()), "skipped": skipped}


def _display_path(p: Path) -> str:
    """Repository-relative when possible (the form every result of record
    carries), absolute otherwise, e.g. a test's tmp root."""
    for cand in (p, p.resolve()):
        try:
            return cand.relative_to(ROOT).as_posix()
        except ValueError:
            continue
    return p.resolve().as_posix()


def _gold_status(facts_root: Path | None = None) -> dict:
    """Who adjudicated the gold these numbers are computed against."""
    modes: Counter = Counter()
    for p in sorted((facts_root or DEFAULT_GOLD_FACTS).glob("*.json")):
        payload = json.loads(p.read_text(encoding="utf-8"))
        for seg in payload["segments"]:
            for f in seg["facts"]:
                if f.get("status", "unreviewed") == "unreviewed":
                    modes["unreviewed"] += 1
                else:
                    modes[(f.get("review") or {}).get("mode", "unrecorded")] += 1
    mode = "human" if set(modes) == {"human"} else (
        "unreviewed" if modes.get("unreviewed") else
        ("machine" if modes.get("machine") else "mixed"))
    return {"adjudication_mode": mode, "counts": dict(modes)}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report_markdown(result: dict, max_segments: int = 25) -> str:
    st = result["gold_status"]
    out = [f"# RQ2 — EUR-Lex eval split, guideline {result['version']}", ""]
    if not result["reportable"]:
        out += [f"> **NOT REPORTABLE.** Gold adjudication mode = "
                f"`{st['adjudication_mode']}` ({st['counts']}); encoder backend = "
                f"`{result['backend']}`. A number here is a working figure only.", ""]
    assignment = result.get("assignment", "one_to_one (unrecorded; pre-2026-09-23)")
    out += [f"Gold root: `{result.get('gold_facts_root', 'data/gold/facts')}` "
            f"(adjudication: {st['adjudication_mode']}).", "",
            f"Assignment: `{assignment}`"
            + ("" if assignment == gc.MANY_TO_MANY else
               " (**ablation**, not the measure of record)")
            + ". Precision = supported predictions / predictions; recall = "
              "covered gold triples / gold triples.", "",
            f"Scope: {result['scope']}. Boilerplate "
            f"{'included' if result['include_boilerplate'] else 'excluded'} (D-03). "
            f"Thresholds {result['thresholds']}.", "",
            "| annotator | docs | segs | facts scored | out of scope | "
            "strict P | strict R | strict F1 | soft P | soft R | soft F1 |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for ann, s in result["summary"].items():
        sm, so = s["strict_micro"], s["soft_micro"]
        out.append(
            f"| `{ann}` | {s['n_docs']} | {s['n_segments']} | {s['n_facts_scored']} | "
            f"{s['n_facts_out_of_scope']} | {sm['precision']:.3f} | {sm['recall']:.3f} | "
            f"{sm['f1']:.3f} | {so['precision']:.3f} | {so['recall']:.3f} | {so['f1']:.3f} |")

    out += ["", "## Failure reasons (soft matcher)", ""]
    for k, v in result["error_reasons"].items():
        out.append(f"- `{k}` — {v}")

    out += ["", "## Segment-level errors", ""]
    for e in result["errors"][:max_segments]:
        out.append(f"### {e['segment_id']} · `{e['annotator']}`")
        for m in e["missing"]:
            out.append(f"- **missing** {m['triple']}  _({m.get('reason', '?')})_")
        for s in e["spurious"]:
            out.append(f"- **spurious** {s['triple']}  _({s.get('reason', '?')})_")
        out.append("")
    if result["skipped"]:
        out += ["## Skipped", ""] + [f"- {s}" for s in result["skipped"]]
    return "\n".join(out)


def default_out_paths(version: str, assignment: str) -> tuple[Path, Path]:
    """Where a run writes when `--out` / `--report` are not given.

    Named per guideline version (the old defaults said `v1` whatever
    `--version` was, so a v2 run overwrote the v1 result), and the ablation
    gets its own suffix so it can never overwrite the result of record.
    """
    gc.check_assignment(assignment)
    stem = f"rq2_eurlex_{version}" + (
        "" if assignment == gc.MANY_TO_MANY else f"__{assignment}")
    return ROOT / "runs" / f"{stem}.json", ROOT / "runs" / f"{stem}_errors.md"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v1", help="guideline version (facts root)")
    ap.add_argument("--annotators", default=None,
                    help="comma-separated subset of annotator names")
    ap.add_argument("--include-boilerplate", action="store_true")
    ap.add_argument("--facts-root", type=Path, default=DEFAULT_GOLD_FACTS,
                    help="gold root to score against (default: the hi-res root "
                         "of record; data/gold/facts is the superseded coarse one)")
    ap.add_argument("--assignment", choices=gc.ASSIGNMENTS, default=gc.DEFAULT_ASSIGNMENT,
                    help="how soft links become credit: many_to_many (design of "
                         "record) or one_to_one (the pre-2026-09-23 greedy "
                         "matcher, ablation only)")
    ap.add_argument("--out", type=Path, default=None,
                    help="default runs/rq2_eurlex_<version>[__one_to_one].json")
    ap.add_argument("--report", type=Path, default=None,
                    help="default runs/rq2_eurlex_<version>[__one_to_one]_errors.md")
    ap.add_argument("--allow-tfidf", action="store_true",
                    help="write the report even under the TF-IDF fallback backend")
    args = ap.parse_args()

    default_out, default_report = default_out_paths(args.version, args.assignment)
    args.out = args.out or default_out
    args.report = args.report or default_report

    result = evaluate(version=args.version,
                      annotators=args.annotators.split(",") if args.annotators else None,
                      include_boilerplate=args.include_boilerplate,
                      facts_root=args.facts_root,
                      assignment=args.assignment)

    if result["backend"] != "sbert" and not args.allow_tfidf:
        raise SystemExit(
            f"backend={result['backend']!r}: TF-IDF numbers are dev-only and are not "
            f"written. Install torch + sentence-transformers, or pass --allow-tfidf "
            f"to keep a clearly-labelled working figure.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(report_markdown(result), encoding="utf-8")

    st = result["gold_status"]
    print(f"gold root={result['gold_facts_root']}  assignment={result['assignment']}")
    print(f"backend={result['backend']}  gold adjudication={st['adjudication_mode']}  "
          f"reportable={result['reportable']}")
    for ann, s in result["summary"].items():
        sm, so = s["strict_micro"], s["soft_micro"]
        print(f"  {ann:22s} docs={s['n_docs']:2d} segs={s['n_segments']:3d} "
              f"scored={s['n_facts_scored']:4d} out-of-scope={s['n_facts_out_of_scope']:4d}  "
              f"strict F1={sm['f1']:.3f}  soft F1={so['f1']:.3f}")
    print(f"wrote {args.out}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
