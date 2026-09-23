"""Build the segment population of an RQ1 comparison (method.tex sec:missing-segments).

The population is the set of segments (doc_id + section_path) that EVERY
annotator of record returned a validated output for under EVERY listed
version. Phase-2 is then run once per version with `--segment-population`,
which removes all other triples before alignment. Logic and the definition of
"returned": `src/segment_population.py`.

    python -m scripts.build_segment_population \\
        --version v1=data/facts --version v2=data/facts_v2 \\
        --doc-list data/doc_lists/eurlex_dev.json \\
        --out runs/segment_population_dev_v1_v2.json

    # then, per version, into a suffix (never over the unrestricted baseline):
    python -m scripts.run_phase2 --doc-list data/doc_lists/eurlex_dev.json \\
        --facts-root data/facts --segment-population runs/segment_population_dev_v1_v2.json \\
        --out-suffix v1_pop_v1v2 ...

Reads manifests and facts files only; writes nothing but `--out`. Refuses a
pass that is not finished (a (version, annotator, doc) with no record) unless
`--allow-missing-records`, and refuses locked eval documents unless
`--include-eval` (discipline rule 1).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_phase2 import ANNOTATORS_OF_RECORD                 # noqa: E402
from src.segment_population import (PopulationError, VersionSource,  # noqa: E402
                                    build_population, sections_from_parsed,
                                    write_population)

EVAL_LIST = ROOT / "data" / "doc_lists" / "eurlex_eval.json"


def _doc_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload["docs"] if isinstance(payload, dict) else payload
    return [d["doc_id"] if isinstance(d, dict) else str(d) for d in entries]


def _short(model: str) -> str:
    for tag in ("Nemotron", "gemma", "qwen"):
        if tag.lower() in model.lower():
            return tag
    return model[:24]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="append", required=True, metavar="LABEL=FACTS_ROOT",
                    help="a guideline version and its facts root (manifest: "
                         "<FACTS_ROOT>/manifest.json). Repeat; RQ1 uses exactly two.")
    ap.add_argument("--doc-list", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--annotators", default=None,
                    help="comma-separated; default: the set of record (run_phase2)")
    ap.add_argument("--parsed-root", type=Path, default=ROOT / "data" / "parsed")
    ap.add_argument("--allow-missing-records", action="store_true",
                    help="build even though some (version, annotator, doc) has no "
                         "record yet; those segments are excluded as `no_record`. "
                         "For a dry look mid-pass only, never for the run of record.")
    ap.add_argument("--include-eval", action="store_true",
                    help="permit locked eval documents (the stage-C run only)")
    args = ap.parse_args(argv)

    versions = []
    for spec in args.version:
        label, sep, root = spec.partition("=")
        if not sep or not label or not root:
            ap.error(f"--version must be LABEL=FACTS_ROOT, got {spec!r}")
        versions.append(VersionSource(label=label.strip(), facts_root=Path(root.strip())))
    annotators = (list(ANNOTATORS_OF_RECORD) if args.annotators is None else
                  [a.strip() for a in args.annotators.split(",") if a.strip()])

    doc_ids = _doc_ids(args.doc_list)
    if EVAL_LIST.exists() and not args.include_eval:
        locked = set(_doc_ids(EVAL_LIST)) & set(doc_ids)
        if locked:
            ap.error(f"{len(locked)} document(s) of {args.doc_list} are in the locked "
                     f"eval set; pass --include-eval only for the stage-C run")

    try:
        population = build_population(
            versions, sections_from_parsed(args.parsed_root, doc_ids), annotators,
            allow_missing_records=args.allow_missing_records)
    except PopulationError as exc:
        print(f"[population] REFUSED: {exc}", file=sys.stderr)
        return 2
    write_population(population, args.out)

    p = population
    print(f"[population] {len(p['doc_ids'])} docs, {p['n_segments']} of "
          f"{p['n_segments_universe']} segments kept, {p['n_segments_excluded']} "
          f"excluded; hash {p['hash'][:16]}")
    for v in p["versions"]:
        per = ", ".join(f"{_short(a)} {n}" for a, n in p["excluded_counts"][v["label"]].items())
        print(f"  {v['label']:<6} ({v['facts_root']}): not returned: {per}")
    print(f"  by reason: {p['excluded_by_reason'] or '{}'}; failed sections by cause: "
          f"{p['excluded_by_cause'] or '{}'}")
    whole_doc: dict[tuple, int] = {}
    for e in p["excluded"]:
        if e["reason"] in ("no_record", "document_failed"):
            k = (e["version"], _short(e["annotator"]), e["doc_id"], e["reason"])
            whole_doc[k] = whole_doc.get(k, 0) + 1
            continue
        print(f"    {e['version']:<6} {_short(e['annotator']):<9} {e['doc_id']} "
              f"{e['section_path']:<24} {e['reason']} ({e['detail'][:40]})")
    for (ver, ann, doc, reason), n in sorted(whole_doc.items()):
        print(f"    {ver:<6} {ann:<9} {doc} (all {n} segments)       {reason}")
    n_net = p["excluded_by_cause"].get("network", 0)
    if n_net:
        print(f"[population] WARNING: {n_net} segment(s) are out because the CALL "
              f"failed (network), not because the model's answer did. That is infra "
              f"noise, not a property of the guideline: re-run those pairs "
              f"(run_extraction resumes failed pairs) before building the population "
              f"of record")
    if not p["complete_passes"]:
        print("[population] WARNING: built over an unfinished pass (no_record); "
              "not a population of record")
    print(f"[population] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
