"""
Synthesize multi-annotator facts for Phase-2 algorithm development.

Why this exists
---------------
Phase-1 batch extraction with 3 models × 50 docs is not yet run (only
qwen3.5:4b × train-000000 exists). Phase-2 alignment / conflict detection
needs ≥ 2 annotators per document to have anything to align. This script
derives two simulated annotators ("gemma3-4b-sim", "phi4-mini-sim") from
the real qwen facts via mechanical perturbations.

The perturbations intentionally inject *known-truth* instances of each
ConflictLabel so the Phase-2 pipeline has a deterministic regression set:

    * REDUNDANCY    — same fact paraphrased / entity surface variant
    * GRANULARITY   — single fact split into two
    * CONTRADICTION — numeric value or polarity flipped
    * NO_CONFLICT   — independent fact extracted by only one annotator

Every synthesized fact is marked with `extra.synthesized = True` and
`extra.synth_op = "<operation>"` so they can be filtered out (or used as
labels) at any point.

Replace these JSON outputs with real `data/facts/<model>/<doc>.json` files
the moment the real 3-model extraction matrix finishes — the rest of the
Phase-2 pipeline does not care about the source.

Usage
-----
    python -m scripts.synthesize_facts \
        --source data/facts/qwen3.5_4b/train-000000.json \
        --out-dir data/facts \
        --seed 42
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Surface-form rewrite tables. Kept small and deliberately verifiable — we
# want a human reading the synthesized JSON to recognise *why* a perturbation
# would or would not collide with the qwen original under SBERT.
# ---------------------------------------------------------------------------

ENTITY_VARIANTS: dict[str, list[str]] = {
    "the Council": ["Council of the European Union", "the EU Council", "the Council"],
    "the Commission": ["European Commission", "EC", "the Commission"],
    "the Netherlands": ["Netherlands", "the Kingdom of the Netherlands"],
    "general government deficit": [
        "general government balance",  # near-duplicate; same referent
        "government deficit",
        "the deficit",
    ],
    "cyclically adjusted deficit": [
        "structural deficit",  # near-synonym
        "cyclically adjusted balance",
    ],
    "savings measures": ["austerity measures", "fiscal measures"],
    "Decision 2005/136/EC": ["the 2005/136/EC Decision", "Decision (2005/136/EC)"],
}

PREDICATE_PARAPHRASE: dict[str, list[str]] = {
    "decided": ["determined", "ruled"],
    "made": ["issued", "produced"],
    "made a Recommendation": ["issued a Recommendation"],
    "established a deadline": ["set a deadline", "fixed a deadline"],
    "is to be abrogated": ["shall be abrogated", "must be repealed"],
    "are laid down": ["are defined", "are specified"],
    "are provided": ["are supplied", "are reported"],
    "is estimated at": ["was estimated at", "stood at"],
    "was": ["stood at", "reached"],
    "is in compliance with": ["complies with", "is consistent with"],
    "was to be reduced below": ["had to fall under", "was required to drop below"],
    "was pursued in 2004": ["was carried out in 2004", "was undertaken in 2004"],
    "were partly contained in": ["were partly included in"],
    "were partly decided in": ["were partly defined in"],
    "added up to": ["totalled", "amounted to"],
    "was projected to reach": ["was forecast to hit", "was expected to reach"],
    "was projected to fall to": ["was forecast to drop to"],
    "was projected to decrease in 2005 by": ["was forecast to decline in 2005 by"],
    "was expected to reach close to balance": ["was forecast to approach balance"],
    "fell to": ["dropped to", "declined to"],
    "was compared to": ["compared with"],
    "was kept below": ["stayed under", "remained below"],
    "was projected to remain below": ["was forecast to stay under"],
    "was completed": ["was finalised", "was concluded"],
    "is hereby abrogated": ["is repealed", "shall be abrogated"],
    "is addressed to": ["is directed at"],
    "was projected to show a further decline in 2005": [
        "was forecast to decline further in 2005",
    ],
    "followed": ["came after"],
}

NUMBER_RE = re.compile(r"(\d+),(\d+)\s*%")  # EUR-Lex "2,3 %" form


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_fact_id(model: str, doc_id: str, idx: int, op: str) -> str:
    """Stable 16-char hash, distinct namespace per (annotator, op)."""
    raw = f"{model}|{doc_id}|{idx}|{op}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _maybe_swap_entity(s: str, rng: random.Random, prob: float) -> tuple[str, bool]:
    """Replace a known entity in `s` with a surface variant; report if changed."""
    if rng.random() > prob:
        return s, False
    for canonical, variants in ENTITY_VARIANTS.items():
        if canonical.lower() in s.lower():
            choice = rng.choice(variants)
            if choice == canonical:
                continue
            # Case-insensitive single-pass replace, preserve a leading "The "
            pat = re.compile(re.escape(canonical), re.IGNORECASE)
            return pat.sub(choice, s, count=1), True
    return s, False


def _maybe_paraphrase_predicate(pred: str, rng: random.Random, prob: float) -> tuple[str, bool]:
    if pred in PREDICATE_PARAPHRASE and rng.random() < prob:
        new = rng.choice(PREDICATE_PARAPHRASE[pred])
        if new != pred:
            return new, True
    return pred, False


def _build_nl(subj: str, pred: str, obj: str) -> str:
    """Cheap natural-language reconstruction used after subj/pred/obj edits."""
    obj_str = "" if obj in ("", "null", None) else f" {obj}"
    sentence = f"{subj} {pred}{obj_str}.".strip()
    return sentence[0].upper() + sentence[1:]


# ---------------------------------------------------------------------------
# Perturbation operations
# ---------------------------------------------------------------------------


def op_paraphrase(
    fact: dict, rng: random.Random, *, p_entity: float, p_pred: float
) -> tuple[dict, list[str]]:
    """REDUNDANCY-flavour: keep meaning, rephrase surface form."""
    out = copy.deepcopy(fact)
    ops = []

    new_subj, ch_s = _maybe_swap_entity(out["subject"], rng, p_entity)
    new_obj, ch_o = _maybe_swap_entity(out["object"], rng, p_entity)
    new_pred, ch_p = _maybe_paraphrase_predicate(out["predicate"], rng, p_pred)

    if ch_s:
        out["subject"] = new_subj
        ops.append("entity_swap_subject")
    if ch_o:
        out["object"] = new_obj
        ops.append("entity_swap_object")
    if ch_p:
        out["predicate"] = new_pred
        ops.append("predicate_paraphrase")

    if ops:
        out["natural_language"] = _build_nl(out["subject"], out["predicate"], out["object"])
    return out, ops


def op_contradict_number(fact: dict, rng: random.Random) -> tuple[dict, list[str]] | None:
    """CONTRADICTION-flavour: nudge a numeric value in subject or object."""
    target_field = None
    for fname in ("object", "subject"):
        if NUMBER_RE.search(fact[fname] or ""):
            target_field = fname
            break
    if target_field is None:
        return None
    out = copy.deepcopy(fact)

    def _bump(m: re.Match) -> str:
        # Add 0.3–0.7 percentage points to introduce a real disagreement.
        whole, frac = m.group(1), m.group(2)
        bumped = float(f"{whole}.{frac}") + rng.choice([0.3, 0.5, 0.7])
        return f"{int(bumped)},{int(round((bumped - int(bumped)) * 10))} %"

    out[target_field] = NUMBER_RE.sub(_bump, out[target_field], count=1)
    out["natural_language"] = _build_nl(out["subject"], out["predicate"], out["object"])
    return out, [f"number_perturb_{target_field}"]


def op_split_granularity(
    fact: dict, rng: random.Random, model: str, idx: int
) -> tuple[list[dict], list[str]] | None:
    """GRANULARITY: split one fact into two thinner ones sharing the locator.

    Only triggers when the object contains a conjunction ("and") or a
    'for/of' clause that can plausibly carry its own atomic fact.
    """
    obj = fact["object"] or ""
    parts = None
    for sep in [" by ", " for ", " in ", " of ", " and "]:
        if sep in obj:
            head, _, tail = obj.partition(sep)
            if 3 < len(head) < 80 and 3 < len(tail) < 80:
                parts = (head.strip(), sep.strip(), tail.strip())
                break
    if parts is None:
        return None

    head, sep, tail = parts
    base = copy.deepcopy(fact)
    base["object"] = head
    base["natural_language"] = _build_nl(base["subject"], base["predicate"], head)
    base["fact_id"] = _new_fact_id(model, fact["doc_id"], idx, "split_a")
    base["extra"] = dict(base.get("extra", {}))

    tail_fact = copy.deepcopy(fact)
    tail_fact["object"] = tail
    tail_fact["predicate"] = f"qualified {sep}" if sep != "and" else f"also {fact['predicate']}"
    tail_fact["natural_language"] = _build_nl(
        tail_fact["subject"], tail_fact["predicate"], tail
    )
    tail_fact["fact_id"] = _new_fact_id(model, fact["doc_id"], idx, "split_b")
    tail_fact["extra"] = dict(tail_fact.get("extra", {}))

    return [base, tail_fact], ["granularity_split"]


# ---------------------------------------------------------------------------
# Per-annotator pipelines. The two profiles below were chosen so that
# alignment + Layer-2 will see all four ConflictLabel categories on
# train-000000 deterministically (seed=42).
# ---------------------------------------------------------------------------


def build_gemma_sim(
    qwen_facts: list[dict], rng: random.Random, *, model: str
) -> list[dict]:
    """Friendly twin: mostly REDUNDANCY-grade variants + occasional GRANULARITY."""
    out: list[dict] = []
    dropped = set(rng.sample(range(len(qwen_facts)), k=min(4, len(qwen_facts) // 8)))

    for idx, src in enumerate(qwen_facts):
        if idx in dropped:
            continue
        # Try granularity split first (rarer)
        split = op_split_granularity(src, rng, model, idx) if rng.random() < 0.10 else None
        if split is not None:
            new_facts, ops = split
            for nf in new_facts:
                nf["annotator"] = model
                nf["extra"]["synthesized"] = True
                nf["extra"]["synth_ops"] = ops
                nf["extra"]["derived_from"] = src["fact_id"]
                out.append(nf)
            continue

        new_fact, ops = op_paraphrase(src, rng, p_entity=0.35, p_pred=0.40)
        new_fact["annotator"] = model
        new_fact["fact_id"] = _new_fact_id(model, src["doc_id"], idx, "paraphrase")
        extra = dict(new_fact.get("extra", {}))
        extra["synthesized"] = True
        extra["synth_ops"] = ops or ["identity"]
        extra["derived_from"] = src["fact_id"]
        new_fact["extra"] = extra
        out.append(new_fact)
    return out


def build_phi_sim(
    qwen_facts: list[dict], rng: random.Random, *, model: str
) -> list[dict]:
    """Adversarial twin: drops more, introduces CONTRADICTION and noisier paraphrase."""
    out: list[dict] = []
    dropped = set(rng.sample(range(len(qwen_facts)), k=min(6, len(qwen_facts) // 6)))

    for idx, src in enumerate(qwen_facts):
        if idx in dropped:
            continue

        # Contradiction has highest priority — only ~10% of facts have a number
        contradiction = (
            op_contradict_number(src, rng) if rng.random() < 0.50 else None
        )
        if contradiction is not None:
            new_fact, ops = contradiction
        else:
            new_fact, ops = op_paraphrase(src, rng, p_entity=0.55, p_pred=0.55)

        new_fact["annotator"] = model
        new_fact["fact_id"] = _new_fact_id(model, src["doc_id"], idx, "phi")
        extra = dict(new_fact.get("extra", {}))
        extra["synthesized"] = True
        extra["synth_ops"] = ops or ["identity"]
        extra["derived_from"] = src["fact_id"]
        new_fact["extra"] = extra
        out.append(new_fact)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _safe_model_dir(name: str) -> str:
    return name.replace(":", "_").replace("/", "_").replace(".", "_")


def synthesize(source_path: Path, out_dir: Path, seed: int) -> dict[str, Path]:
    with source_path.open("r", encoding="utf-8") as fh:
        src_doc = json.load(fh)

    qwen_facts = src_doc["facts"]
    doc_id = src_doc["doc_id"]

    rng = random.Random(seed)
    written: dict[str, Path] = {}

    plans = [
        ("gemma3-4b-sim", build_gemma_sim, "v1"),
        ("phi4-mini-sim", build_phi_sim, "v1"),
    ]

    for model_name, builder, guideline in plans:
        # Use a per-model derived rng so adding a new annotator doesn't
        # silently reshuffle the others.
        per_model_rng = random.Random(seed ^ hash(model_name) & 0xFFFFFFFF)
        facts = builder(qwen_facts, per_model_rng, model=model_name)
        for f in facts:
            f["guideline_version"] = guideline

        out_path = out_dir / _safe_model_dir(model_name) / f"{doc_id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "doc_id": doc_id,
            "annotator": model_name,
            "guideline_version": guideline,
            "fact_count": len(facts),
            "facts": facts,
            "_synthetic": True,
            "_source": str(source_path),
            "_seed": seed,
        }
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        written[model_name] = out_path
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        default=Path("data/facts/qwen3.5_4b/train-000000.json"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("data/facts"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    written = synthesize(args.source, args.out_dir, args.seed)
    print(f"[synthesize_facts] source = {args.source}")
    for model, path in written.items():
        print(f"  + {model:20s} -> {path}")


if __name__ == "__main__":
    main()
