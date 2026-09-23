"""Stamp provenance and review status onto every gold fact.

The gold in data/gold/facts/ was produced by a language model as a
PRE-ANNOTATION. It is not a human gold standard until each fact has been
adjudicated by the annotator, and it must never be reported as one: the models
under evaluation are themselves LLMs, so an unreviewed machine gold would make
RQ2 a comparison of a model against a model.

Every fact therefore carries:
    provenance = "machine-preannotation"
    status     = "unreviewed"   ->  "accepted" | "edited" | "rejected"

`python -m src.gold_store` refuses nothing on account of status; the review
gate is enforced by scripts/gold_stats.py, which reports the reviewed share and
warns while it is below 100 %.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FACTS = ROOT / "data" / "gold" / "facts"

PROVENANCE = {
    "produced_by": "claude-opus-5",
    "kind": "machine-preannotation",
    "produced_on": "2026-07-31",
    "model_under_evaluation": False,
    "note": (
        "Pre-annotation only. Not a human gold standard until every fact has "
        "status 'accepted' or 'edited' after adjudication by the annotator. "
        "The IAA subsample in data/gold/segment_sample.json must be annotated "
        "blind by the second annotator, without sight of these records."
    ),
}


def main() -> None:
    n_files = n_facts = 0
    for p in sorted(FACTS.glob("*.json")):
        payload = json.loads(p.read_text(encoding="utf-8"))
        payload["annotator"] = payload.get("annotator", "machine:pre-annotation")
        payload["guideline_version"] = "gold-invariant"
        payload["provenance"] = PROVENANCE
        payload["stamped_on"] = date.today().isoformat()
        for seg in payload["segments"]:
            for f in seg["facts"]:
                f.setdefault("boilerplate", False)
                f.setdefault("status", "unreviewed")
                f.setdefault("reviewer_note", "")
                n_facts += 1
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        n_files += 1
    print(f"stamped {n_facts} facts in {n_files} files")


if __name__ == "__main__":
    main()
