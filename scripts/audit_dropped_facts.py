"""Recover, from a run log, the facts that were dropped INSIDE a kept section.

`audit_facts_coverage.py` closes the section-shaped hole: a section that
produced nothing is visible because its `section_path` is absent from the facts
file. This closes the fact-shaped one underneath it.

When a section's answer contains some valid facts and some invalid ones, the
extractor keeps the valid ones and logs `keeping N fact(s) with M UNCORRECTED
defect(s)`. That is the right call — trading good facts for a retry is worse —
but the manifest does not record it: `incomplete_sections` counts skipped
sections only, so a section that kept 1 of 3 facts is recorded as complete and
clean. The run log is then the ONLY record that anything was lost, which is
exactly the "the run believed it had a clean document" failure mode this
project keeps hitting.

So: parse the log, attribute each retention to (model, doc, section), and
report the per-annotator fact-level loss. Read-only, safe to run mid-pass.

The count is a LOWER bound. A defect names one slot of one fact, so M defects
means at least M facts were rejected, and a fact rejected on two slots at once
counts twice; the number of *distinct* dropped facts is recovered from the
`facts[i]` indices when they are present in the line.

    python scripts/audit_dropped_facts.py runs/dev_pass_20260908.log
    python scripts/audit_dropped_facts.py runs/*.log --json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODEL = re.compile(r"\[cloud:([^\]]+)\]")
DOC = re.compile(r"run\s+(train-\d+)")
SECTION = re.compile(r"((?:preamble|enacting|annex)[.\w]*(?:\[\d+\])?)")
KEPT = re.compile(r"keeping (\d+) fact\(s\) with (\d+) UNCORRECTED defect\(s\)")
SKIPPED = re.compile(r"SKIPPED, no parseable response")
FACT_IX = re.compile(r"facts\[(\d+)\]")
SLOT = re.compile(r'facts\[\d+\]\.(?:triples\[\d+\]\.)?"(\w+)"')


def short(model: str) -> str:
    """A readable annotator name: the hf.co path is unreadable in a table."""
    name = model.replace("hf.co/unsloth/", "")
    for tag in ("Nemotron", "gemma", "qwen"):
        if tag.lower() in name.lower():
            return tag
    return name[:20]


def scan(paths: list[Path]) -> dict:
    doc = model = None
    kept: dict[str, int] = defaultdict(int)
    dropped: dict[str, int] = defaultdict(int)
    sections: dict[str, set] = defaultdict(set)
    skipped: dict[str, set] = defaultdict(set)
    slots: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    worst: list[tuple] = []

    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            found_doc = DOC.search(line)
            if found_doc:
                doc = found_doc.group(1)
            found_model = MODEL.search(line)
            if not found_model:
                continue
            model = short(found_model.group(1))
            found_section = SECTION.search(line)
            section = found_section.group(1) if found_section else "?"

            if SKIPPED.search(line):
                skipped[model].add((doc, section))
                continue

            hit = KEPT.search(line)
            if not hit:
                continue
            n_kept, n_defects = int(hit.group(1)), int(hit.group(2))
            # Distinct facts, not distinct defects: one fact can be rejected on
            # two slots at once and would otherwise be counted twice.
            n_dropped = len(set(FACT_IX.findall(line))) or n_defects
            kept[model] += n_kept
            dropped[model] += n_dropped
            sections[model].add((doc, section))
            for slot in SLOT.findall(line):
                slots[model][slot] += 1
            worst.append((n_dropped, n_kept, model, doc, section))

    return {
        "annotators": sorted({*kept, *dropped, *skipped}),
        "kept": dict(kept),
        "dropped": dict(dropped),
        "sections_affected": {m: len(s) for m, s in sections.items()},
        "sections_skipped": {m: len(s) for m, s in skipped.items()},
        "slots": {m: dict(s) for m, s in slots.items()},
        "worst": sorted(worst, reverse=True)[:10],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+", help="run log(s) to scan")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    paths = [Path(p) for p in args.logs]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit("no such log: " + ", ".join(str(p) for p in missing))

    report = scan(paths)
    if args.json:
        print(json.dumps(report, indent=2))
        return

    if not report["annotators"]:
        print("no retention or skip events in these logs — nothing was lost")
        return

    print(f"{'annotator':<12}{'facts kept':>11}{'facts dropped':>15}"
          f"{'loss %':>9}{'sections hit':>14}{'skipped':>9}")
    for model in report["annotators"]:
        kept = report["kept"].get(model, 0)
        dropped = report["dropped"].get(model, 0)
        total = kept + dropped
        pct = f"{100.0 * dropped / total:.1f}" if total else "-"
        print(f"{model:<12}{kept:>11}{dropped:>15}{pct:>9}"
              f"{report['sections_affected'].get(model, 0):>14}"
              f"{report['sections_skipped'].get(model, 0):>9}")

    print("\nrejected slots (a defect names the slot it was rejected on):")
    for model in report["annotators"]:
        by_slot = report["slots"].get(model)
        if by_slot:
            ranked = sorted(by_slot.items(), key=lambda kv: -kv[1])
            print(f"  {model:<10} " + ", ".join(f"{k}={v}" for k, v in ranked))

    if report["worst"]:
        print("\nworst sections (dropped/kept):")
        for n_dropped, n_kept, model, doc, section in report["worst"]:
            print(f"  {n_dropped} dropped, {n_kept} kept  {model:<10}"
                  f" {doc}::{section}")

    print("\nNOTE: a lower bound, and NOT recorded in data/facts/manifest.json —"
          "\nthese sections are marked complete there. The log is the only"
          "\nrecord, so keep it with the run.")


if __name__ == "__main__":
    main()
