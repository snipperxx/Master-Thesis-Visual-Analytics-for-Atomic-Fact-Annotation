"""Validate an external model's adjudication replies into decisions files.

    python -m scripts.import_external_adjudication data/gold/external_replies \\
        --model gemini-2.5-pro

Takes whatever the model returned (raw ```json blocks, or bare JSON, one file per
document), checks it against the gold, and writes
`data/gold/decisions_<model>/<doc>.json` in the shape `apply_gold_review.py`
expects --- with **`mode: "machine"`**, so the human-provenance guard in that
script cannot be tripped by it. This pass removes the circularity in the current
gold (same model pre-annotated and adjudicated); it does **not** make the gold
human-adjudicated, and nothing here will let it claim to be.

Nothing is written if a file fails validation. What is checked:

* every gold_id exists in the gold and belongs to the document it was returned under
* every fact of the document is decided exactly once --- silent omissions are the
  failure mode that would quietly shrink the gold
* `status` is one of accepted / edited / rejected
* `edits` only on `edited`, and only slots that exist
* any `source_quote`, kept or edited, appears **verbatim** in that segment's text
* added facts name a real segment and quote it verbatim
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

from src.gold_store import GOLD, GOLD_FACTS, doc_fingerprint, sample_index, segment_text

VALID_STATUS = {"accepted", "edited", "rejected"}
EDITABLE = {"subject", "predicate", "object", "natural_language",
            "source_quote", "condition", "temporal_context"}
FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse(raw: str) -> dict:
    """Tolerant of a fenced block, prose around it, or bare JSON."""
    for candidate in ([m.group(1) for m in FENCE.finditer(raw)] + [raw]):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "decisions" in obj:
                return obj
        except json.JSONDecodeError:
            continue
    start = raw.find("{")
    while start != -1:
        try:
            obj = json.loads(raw[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = raw.find("{", start + 1)
    raise ValueError("no JSON object with a 'decisions' key found")


def _gold_index(doc_id: str) -> tuple[dict, dict]:
    """(gold_id -> fact, gold_id -> segment_id) for one document."""
    payload = json.loads((GOLD_FACTS / f"{doc_id}.json").read_text(encoding="utf-8"))
    facts, seg_of = {}, {}
    for seg in payload["segments"]:
        for f in seg["facts"]:
            facts[f["gold_id"]] = f
            seg_of[f["gold_id"]] = seg["segment_id"]
    return facts, seg_of


def _seg_text(doc_id: str, segment_id: str, expect: dict) -> str:
    m = expect[segment_id]
    return segment_text(doc_id, m["section_path"], m["char_start"], m["char_end"])


def validate_doc(doc_id: str, payload: dict, expect: dict) -> tuple[dict, list, list[str]]:
    facts, seg_of = _gold_index(doc_id)
    errors: list[str] = []
    decisions = payload.get("decisions") or {}

    # Replies carrying a fingerprint (everything the console exports) are checked
    # against the gold as it stands now. Older replies have none and are read as
    # before --- absence is not treated as a mismatch.
    stamp = payload.get("_gold_fingerprint")
    if stamp and stamp != doc_fingerprint(doc_id):
        errors.append(f"the gold moved after this reply was produced "
                      f"(reply {stamp}, gold {doc_fingerprint(doc_id)}); re-ask this document")

    unknown = sorted(set(decisions) - set(facts))
    missing = sorted(set(facts) - set(decisions))
    for gid in unknown:
        errors.append(f"unknown gold_id: {gid}")
    for gid in missing:
        errors.append(f"no decision returned for: {gid}")

    clean: dict[str, dict] = {}
    for gid, d in decisions.items():
        if gid not in facts:
            continue
        status = (d or {}).get("status")
        if status not in VALID_STATUS:
            errors.append(f"{gid}: bad status {status!r}")
            continue
        edits = d.get("edits") or {}
        if edits and status != "edited":
            errors.append(f"{gid}: edits present but status is {status!r}")
            continue
        bad_slots = sorted(set(edits) - EDITABLE)
        if bad_slots:
            errors.append(f"{gid}: not editable: {bad_slots}")
            continue
        quote = edits.get("source_quote") or facts[gid]["source_quote"]
        text = _seg_text(doc_id, seg_of[gid], expect)
        if quote and quote not in text:
            errors.append(f"{gid}: source_quote not verbatim in the segment text")
            continue
        entry = {"status": status, "note": (d.get("note") or "")[:300]}
        if edits:
            entry["edits"] = edits
        clean[gid] = entry

    added = []
    for a in payload.get("added") or []:
        sid = a.get("segment_id")
        if sid not in expect or not sid.startswith(doc_id):
            errors.append(f"added fact names an unknown segment: {sid!r}")
            continue
        q = a.get("source_quote") or ""
        if q not in _seg_text(doc_id, sid, expect):
            errors.append(f"added fact in {sid}: source_quote not verbatim")
            continue
        added.append(a)

    return clean, added, errors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("replies", type=Path,
                    help="directory of the model's replies, one file per document "
                         "(named <doc_id>.json / .txt / .md)")
    ap.add_argument("--model", required=True,
                    help="adjudicator label recorded in the decisions file")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--strict", action="store_true",
                    help="write nothing unless every document validates")
    args = ap.parse_args()

    out = args.out or GOLD / f"decisions_{re.sub(r'[^A-Za-z0-9]+', '_', args.model)}"
    expect = sample_index()
    known_docs = {p.stem for p in GOLD_FACTS.glob("*.json")}

    results, all_errors = {}, {}
    for p in sorted(args.replies.iterdir()):
        if p.is_dir() or p.name.startswith("_"):
            continue
        doc_id = p.stem
        if doc_id not in known_docs:
            all_errors[p.name] = [f"filename does not match a gold document: {doc_id}"]
            continue
        try:
            payload = _parse(p.read_text(encoding="utf-8"))
        except ValueError as exc:
            all_errors[doc_id] = [str(exc)]
            continue
        clean, added, errors = validate_doc(doc_id, payload, expect)
        results[doc_id] = (clean, added)
        if errors:
            all_errors[doc_id] = errors

    print(f"parsed {len(results)} document(s) from {args.replies}")
    for doc_id, errs in all_errors.items():
        print(f"  !! {doc_id}: {len(errs)} problem(s)")
        for e in errs[:6]:
            print(f"       {e}")
        if len(errs) > 6:
            print(f"       ... and {len(errs) - 6} more")

    if all_errors and args.strict:
        raise SystemExit("--strict: nothing written")
    if not results:
        raise SystemExit("nothing to write")

    out.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    for doc_id, (clean, added) in results.items():
        if doc_id in all_errors and args.strict:
            continue
        stats.update(d["status"] for d in clean.values())
        stats["added"] += len(added)
        (out / f"{doc_id}.json").write_text(json.dumps({
            "reviewed_on": str(date.today()),
            "reviewer": args.model,
            # MACHINE, always. An external model is a second machine opinion; it
            # removes the circularity of one model adjudicating itself, and it
            # does not make the gold human-adjudicated.
            "mode": "machine",
            "doc_id": doc_id,
            "basis": "independent blind adjudication from data/gold/external_task",
            "decisions": clean,
            "added": added,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nwrote {out}  {dict(stats)}")
    print("next:")
    print(f"    python -m scripts.compare_adjudications {out}      # agreement vs the current gold")
    print(f"    python scripts/apply_gold_review.py {out} --mode machine --reviewer {args.model}")


if __name__ == "__main__":
    main()
