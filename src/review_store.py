"""
Conflict-review persistence — the bridge between "looking at conflicts"
and "knowing what to change in guideline v2".

Each review attaches analyst judgement to ONE aligned pair:

    {
      "rules":        ["2", "6"],          # guideline rule ids implicated
      "note":         "demonstrative not resolved",
      "resolution":   "agree" | "relabel:<label>" | "dismiss",
      "label_at_review": "contradiction",  # pipeline label when reviewed
      "annotators":   ["qwen3.5:4b", "gemma3:4b"],
      "ts":           "2026-06-10T12:00:00+00:00"
    }

Storage: data/reviews/<doc_id>.json  — { pair_key: record }.
pair_key = "<fact_a_id>|<fact_b_id>" ("-" for an orphan side).

summarize_reviews() aggregates rule attributions across every reviewed
doc; the Experiment tab renders it as the evidence panel for authoring v2.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Two heading conventions, because the guideline body was replaced on
# 2026-08-31 (CHANGELOG: the ad-hoc "Guideline (v5)" gave way to R1-R10
# restated from data/gold/calibration/_RULES.md):
#
#   legacy   "### 6. Conditions and temporal context"   -> id "6"
#   current  "R1b. IF A NOUN PHRASE INSIDE ..."          -> id "R1b"
#
# Only the legacy form was recognised until 2026-09-04, so
# parse_guideline_rules() silently returned [] against the current guideline
# and every conflict in the UI lost its rule attribution — the input to the
# g_v1 -> g_v2 diagnosis this thesis is built on (BUG-2026-09-04-B).
_RULE_RE = re.compile(r"^###\s*(\d+)\.\s*(.+?)\s*$", re.MULTILINE)
# The R-form runs the title on into the rule body, so take the leading
# sentence only: "R1. THE PREDICATE CARRIES THE RELATION ONLY. It must ..."
_RULE_R_RE = re.compile(
    r"^(R\d+[a-z]?)\.\s+(.+?)(?:\.\s|\.$|\n\n)", re.MULTILINE | re.DOTALL
)
_GUIDELINE_BLOCK_RE = re.compile(r"<guideline>(.*?)</guideline>", re.DOTALL)

VALID_RESOLUTIONS = ("agree", "dismiss")  # plus "relabel:<label>"
CONFLICT_LABELS = ("contradiction", "granularity", "redundancy", "no_conflict", "unlabeled")


def parse_guideline_rules(version: str, prompts_dir: Path) -> list[dict]:
    """Extract the guideline's rules from prompts/extract_<version>.md.

    Handles both heading conventions (see _RULE_RE / _RULE_R_RE): the legacy
    `### N. Title` form and the current `R1. TITLE. body...` form. Only looks
    inside the <guideline>...</guideline> block so schema/output headings never
    leak in. Returns [] when the file or block is missing.
    """
    path = Path(prompts_dir) / f"extract_{version}.md"
    if not path.exists():
        return []
    body = path.read_text(encoding="utf-8")
    m = _GUIDELINE_BLOCK_RE.search(body)
    block = m.group(1) if m else body

    rules = [{"id": rid, "title": " ".join(title.split())}
             for rid, title in _RULE_RE.findall(block)]
    if rules:
        return rules
    return [{"id": rid, "title": " ".join(title.split())}
            for rid, title in _RULE_R_RE.findall(block)]


def _reviews_path(reviews_root: Path, doc_id: str) -> Path:
    return Path(reviews_root) / f"{doc_id}.json"


def load_reviews(reviews_root: Path, doc_id: str) -> dict[str, dict]:
    path = _reviews_path(reviews_root, doc_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_review(
    reviews_root: Path,
    doc_id: str,
    pair_key: str,
    payload: dict,
) -> dict:
    """Validate + persist one review record. Empty payload deletes the record."""
    reviews_root = Path(reviews_root)
    reviews_root.mkdir(parents=True, exist_ok=True)
    cur = load_reviews(reviews_root, doc_id)

    if payload.get("delete"):
        cur.pop(pair_key, None)
        _reviews_path(reviews_root, doc_id).write_text(
            json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"deleted": True, "pair_key": pair_key, "n_reviews": len(cur)}

    resolution = (payload.get("resolution") or "agree").strip()
    if resolution not in VALID_RESOLUTIONS and not (
            resolution.startswith("relabel:")
            and resolution.split(":", 1)[1] in CONFLICT_LABELS):
        raise ValueError(f"invalid resolution {resolution!r}")

    rules = payload.get("rules") or []
    if not isinstance(rules, list) or not all(isinstance(r, str) for r in rules):
        raise ValueError("rules must be a list of rule-id strings")

    record = {
        "rules": sorted(set(rules)),
        "note": str(payload.get("note") or "")[:2000],
        "resolution": resolution,
        "label_at_review": payload.get("label_at_review") or "unlabeled",
        "annotators": payload.get("annotators") or [],
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    cur[pair_key] = record
    _reviews_path(reviews_root, doc_id).write_text(
        json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"pair_key": pair_key, "record": record, "n_reviews": len(cur)}


def summarize_reviews(
    reviews_root: Path,
    *,
    prompts_dir: Path | None = None,
    guideline_version: str = "v1",
) -> dict[str, Any]:
    """Aggregate rule attributions across every data/reviews/<doc>.json.

    Output drives the v2-authoring evidence panel:
      rules: [{id, title, n_conflicts, by_label, sample_notes, docs}]
      sorted by n_conflicts desc; un-attributed reviews counted separately.
    """
    reviews_root = Path(reviews_root)
    rule_titles = {r["id"]: r["title"]
                   for r in parse_guideline_rules(guideline_version, prompts_dir)} \
        if prompts_dir else {}

    per_rule: dict[str, dict] = defaultdict(
        lambda: {"n_conflicts": 0, "by_label": defaultdict(int),
                 "sample_notes": [], "docs": set()})
    n_reviews = 0
    n_unattributed = 0
    docs_seen: set[str] = set()

    if reviews_root.exists():
        for fp in sorted(reviews_root.glob("*.json")):
            doc_id = fp.stem
            try:
                reviews = json.loads(fp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            for pair_key, rec in reviews.items():
                n_reviews += 1
                docs_seen.add(doc_id)
                rules = rec.get("rules") or []
                if not rules:
                    n_unattributed += 1
                for rid in rules:
                    slot = per_rule[rid]
                    slot["n_conflicts"] += 1
                    slot["by_label"][rec.get("label_at_review", "unlabeled")] += 1
                    slot["docs"].add(doc_id)
                    note = (rec.get("note") or "").strip()
                    if note and len(slot["sample_notes"]) < 5:
                        slot["sample_notes"].append(
                            {"doc_id": doc_id, "pair_key": pair_key, "note": note})

    rules_out = []
    for rid, slot in per_rule.items():
        rules_out.append({
            "id": rid,
            "title": rule_titles.get(rid, ""),
            "n_conflicts": slot["n_conflicts"],
            "by_label": dict(slot["by_label"]),
            "sample_notes": slot["sample_notes"],
            "docs": sorted(slot["docs"]),
        })
    rules_out.sort(key=lambda r: (-r["n_conflicts"], int(r["id"]) if r["id"].isdigit() else 999))

    return {
        "guideline_version": guideline_version,
        "n_reviews": n_reviews,
        "n_docs": len(docs_seen),
        "n_unattributed": n_unattributed,
        "rules": rules_out,
    }
