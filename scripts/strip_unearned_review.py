"""Remove the human sign-off that `data/gold/facts_hires/` claims but cannot show.

    python -m scripts.strip_unearned_review            # dry run
    python -m scripts.strip_unearned_review --write

Every one of the 312 facts in `facts_hires` carries
`review = {"by": "Zilong Yan", "mode": "human", "on": "2026-08-18"}` and
`status = "accepted"`, and the document header says
`provenance.kind = "machine-preannotation, human-adjudicated"`.

Nothing supports that:

* `data/gold/decisions/` holds twelve files and every one is
  `reviewer: claude-opus-5, mode: machine`. There is no human decisions artefact
  anywhere under `data/gold/` or `runs/`.
* The string "Zilong Yan" occurs in no script in this repository. The stamp was
  written straight into the data, so no run of `apply_gold_review.py` produced it
  and none of that script's guards ever saw it — including the one added for
  BUG-2026-08-09-F, which exists to stop exactly this: a `--mode human` run over a
  machine artefact recording a person's sign-off on facts nobody opened.
* `build_gold_hires.py` writes every fact `unreviewed`. The 2026-08-17 rebuild
  reset the sweep (CHANGELOG, "the sweep is reset"); the stamp is dated the day
  after, and no sweep is recorded between the two.

What IS attested is narrower and stays: Zilong opened `review_facts_hires.html`,
found 44 flagged facts, and the reading produced decisions D-19 and D-16 and nine
rewritten facts. That is a human contribution to the *rules*, recorded in
`decision_log.json` and the CHANGELOG. It is not a per-fact adjudication of 312
facts, and reporting it as one would make RQ2 a model scored against a model with a
person's name on it -- the failure `stamp_gold_provenance.py` opens by warning about.

So: drop `review`, reset `status` to `unreviewed`, and say in `provenance` what the
artefact actually is. The originals are copied to a timestamped backup first.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "gold"
DEST = GOLD / "facts_hires"

KIND = "machine-preannotation, NOT adjudicated"
NOTE = (
    "Pre-annotated by claude-opus-5 and re-decomposed under "
    "data/gold/calibration/_RULES.md. No per-fact adjudication has been applied to "
    "this artefact: every fact is `unreviewed`. A human sign-off claimed on "
    "2026-08-18 for all 312 facts was removed on 2026-08-27 because no decisions "
    "artefact records it (all of data/gold/decisions/ is reviewer claude-opus-5, "
    "mode machine). Human input to this pass took the form of guideline decisions "
    "(D-16, D-19) and nine fact rewrites, which are in decision_log.json and the "
    "CHANGELOG -- not a fact-by-fact review. Do not report this as a human gold "
    "standard; RQ2 would then score a model against a model."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--facts-root", default=str(DEST), type=Path)
    args = ap.parse_args()
    root = Path(args.facts_root)

    files = sorted(root.glob("*.json"))
    if not files:
        raise SystemExit(f"no gold files under {root}")

    stamps: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    payloads: dict[Path, dict] = {}
    n_facts = 0

    for p in files:
        payload = json.loads(p.read_text(encoding="utf-8"))
        for seg in payload["segments"]:
            for f in seg["facts"]:
                n_facts += 1
                rev = f.get("review") or {}
                stamps[f"{rev.get('by', '-')} / {rev.get('mode', '-')}"] += 1
                statuses[f.get("status", "-")] += 1
                f.pop("review", None)
                f["status"] = "unreviewed"
        prov = payload.setdefault("provenance", {})
        prov["kind"] = KIND
        prov["note"] = NOTE
        prov["review_stripped_on"] = "2026-08-27"
        payloads[p] = payload

    print(f"{root.relative_to(ROOT)}: {len(files)} documents, {n_facts} facts")
    print("  review stamp found:")
    for k, v in stamps.most_common():
        print(f"    {v:>4}x  {k}")
    print("  status found:")
    for k, v in statuses.most_common():
        print(f"    {v:>4}x  {k}")
    print(f"  -> all {n_facts} become status=unreviewed with no `review` block")
    print(f"  -> provenance.kind := {KIND!r}")

    if not args.write:
        print("\n  dry run: nothing written (pass --write)")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = GOLD / f"_backup_{root.name}_prestrip_{stamp}"
    shutil.copytree(root, backup)
    print(f"\n  backed up to {backup.relative_to(ROOT)}")
    for p, payload in payloads.items():
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  rewrote {len(payloads)} file(s)")


if __name__ == "__main__":
    main()
