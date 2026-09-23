"""Audit written facts against the sections they were supposed to cover.

Why this exists as a SEPARATE check, not a flag inside the run: every silent
under-count this project has hit (BUG-2026-09-04-A, and the schema-exhaustion
door closed on 2026-09-07) was a *bookkeeping* failure — the run believed it
had a clean document. So this recomputes coverage from the two artefacts that
cannot lie about each other: the section list `enumerate_sections()` derives
from `data/parsed/<doc>.json`, and the `section_path`s actually present in
`data/facts/<model>/<doc>.json`. A record can only pass if the facts file
accounts for every section of the document.

A missing section is NOT by itself an error — a citation block can legitimately
yield no facts. What is reported is the combination the thesis's denominator
cares about:

  * `status: ok` records whose facts file is missing sections (the loss mode)
  * records with zero facts (nothing was extracted at all)
  * records whose facts file is absent or unreadable
  * `status: ok` records carrying `incomplete_sections > 0` (contradiction)

Run it any time during or after a pass — it only reads.

    python scripts/audit_facts_coverage.py
    python scripts/audit_facts_coverage.py --json          # machine-readable
    python scripts/audit_facts_coverage.py --requeue       # drop bad ok records
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import re                                                      # noqa: E402

from src.extractor import enumerate_sections, load_parsed_doc   # noqa: E402

#: A section that legitimately yields no facts. Absence is then expected, and
#: reporting it as a loss trains the reader to skim the report — the same cost
#: that retired the 9-word warning from the annotation console (D-30).
#:
#: `OJ_FOOTNOTE`: an OJ reference line that `eurlex_parse` typed as a recital.
#: 4 of the corpus's 345 recital sections, all in `train-000149`, none in the
#: gold (measured 2026-09-07). The parse is FROZEN — the dev/eval split and the
#: gold sample were drawn on it — so this is carried as a known inconsistency,
#: not fixed by re-parsing.
#: `CLOSING_FORMULA`: the addressee/signature clause, contentless by D-03
#: ("This Directive is addressed to the Member States. Done at Brussels ...").
OJ_FOOTNOTE = re.compile(r"^\(\s*\d{1,2}\s*\)\s*OJ\b", re.I)
#: The addressee clause names whoever the act is addressed to, so matching the
#: literal "Member States" missed `train-000035 enacting.article_3` ("addressed
#: to the Kingdom of the Netherlands") and flagged a section D-03 says holds
#: nothing. Match the construction, not one of its objects.
CLOSING_FORMULA = re.compile(
    r"is addressed to the|binding in its entirety"
    r"|directly applicable in all Member States"
    r"|Done at [A-Z][a-z]+,", re.I)


def _expected_empty(text: str) -> str:
    """Name the reason this section legitimately holds no facts, else ""."""
    t = text.strip()
    if OJ_FOOTNOTE.match(t):
        return "OJ footnote mis-typed as a recital"
    if CLOSING_FORMULA.search(t):
        return "contentless closing/addressee formula (D-03)"
    return ""

MANIFEST = ROOT / "data" / "facts" / "manifest.json"


def _sections_of(doc_path: Path) -> list[tuple[str, str]]:
    """(section_path, text) for every section of a parsed document.

    load_parsed_doc, not ParsedDocument(**json) — the on-disk file wraps the
    document under a "document" key alongside parse metadata.
    """
    return [(s.section_path, s.text)
            for s in enumerate_sections(load_parsed_doc(doc_path))]


def audit(manifest_path: Path = MANIFEST) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for key, rec in sorted(manifest.get("runs", {}).items()):
        row = {"key": key, "model": rec.get("model"), "doc_id": rec.get("doc_id"),
               "status": rec.get("status"), "fact_count": rec.get("fact_count", 0),
               "problems": [], "missing_sections": [], "n_sections": None}
        rows.append(row)

        if rec.get("status") != "ok":
            continue  # a failed record is already marked for re-run

        if rec.get("incomplete_sections"):
            row["problems"].append(
                f"status=ok but incomplete_sections={rec['incomplete_sections']}")

        out = rec.get("output_path")
        fp = ROOT / out if out else None
        if not out or not fp.exists():
            row["problems"].append("facts file missing")
            continue
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            row["problems"].append(f"facts file unreadable: {exc}")
            continue

        facts = payload.get("facts", [])
        if not facts:
            row["problems"].append("zero facts written")

        doc_path = ROOT / rec["doc_path"]
        if not doc_path.exists():
            row["problems"].append("parsed doc missing")
            continue
        expected = _sections_of(doc_path)
        seen = {f.get("source_locator", {}).get("section_path") for f in facts}
        missing, expected_empty = [], {}
        for path, text in expected:
            if path in seen:
                continue
            why = _expected_empty(text)
            if why:
                expected_empty[path] = why
            else:
                missing.append(path)
        row["n_sections"] = len(expected)
        row["missing_sections"] = missing
        row["expected_empty"] = expected_empty
        if missing:
            row["problems"].append(
                f"{len(missing)}/{len(expected)} sections absent from the facts "
                f"file without an explanation")

    bad = [r for r in rows if r["problems"]]
    return {"n_records": len(rows), "n_bad": len(bad), "rows": rows, "bad": bad}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--requeue", action="store_true",
                    help="delete flagged status=ok records so a resumed run "
                         "picks them up as pending (never touches clean ones)")
    args = ap.parse_args()

    result = audit(args.manifest)
    if args.json:
        print(json.dumps(result, indent=1, ensure_ascii=False))
        return 1 if result["n_bad"] else 0

    ok = result["n_records"] - result["n_bad"]
    print(f"{result['n_records']} records — {ok} clean, {result['n_bad']} flagged")
    for r in result["bad"]:
        print(f"\n  {r['model']} × {r['doc_id']}  "
              f"(status={r['status']}, {r['fact_count']} facts)")
        for p in r["problems"]:
            print(f"    - {p}")
        if r["missing_sections"]:
            print(f"    missing: {', '.join(r['missing_sections'][:8])}"
                  + (" ..." if len(r["missing_sections"]) > 8 else ""))
        for path, why in list(r.get("expected_empty", {}).items())[:4]:
            print(f"    (empty as expected: {path} — {why})")

    if args.requeue and result["bad"]:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        removed = [r["key"] for r in result["bad"]
                   if manifest["runs"].pop(r["key"], None) is not None]
        bak = args.manifest.with_suffix(".json.bak_audit")
        bak.write_text(json.dumps(json.loads(args.manifest.read_text(encoding="utf-8")),
                                  indent=1), encoding="utf-8")
        args.manifest.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        print(f"\nrequeued {len(removed)} record(s); backup at {bak.name}")
    return 1 if result["n_bad"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
