"""
Phase-2 batch driver — wires alignment, Layer-1 filter, optional Layer-2
arbitration, and SBERT-based entity clustering into a single command.

Usage
-----
    # Single doc, skip the LLM (use a deterministic stub instead):
    python -m scripts.run_phase2 --doc train-000000 --skip-layer2

    # All docs with >=2 annotators on disk (real Phase-1 output or synthetic):
    python -m scripts.run_phase2 --all --skip-layer2

    # Full run with the arbitrator of record (cloud:qwen3.6:35b, the default
    # since 2026-09-23; needs OLLAMA_CLOUD_URL / OLLAMA_CLOUD_API_KEY in .env):
    python -m scripts.run_phase2 --doc train-000004

Inputs:   data/facts/<annotator>/<doc_id>.json   (one file per annotator)
Outputs:  data/conflicts/<doc_id>.json           (everything the UI needs)
"""

from __future__ import annotations
import os

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from src.alignment import (SLOT_GATES, align_all_pairs, build_groups, encode_texts,
                           group_to_dict, pair_to_dict)
from src.conflict_layer1 import Layer1Config, classify_all, classify_groups
from src.conflict_layer2 import Layer2Unavailable
from src.conflict_layer2 import (ARBITRABLE_SHAPES_DEFAULT, arbitrate_all,
                                 arbitrate_groups)
from src import segment_population as segpop


#: The annotator set of record, frozen 2026-09-03 (CLAUDE.md, blocker 6).
#: `data/facts/` still holds the three retired local 4B directories from the
#: August pilot, and both discovery helpers used to sweep EVERY subdirectory —
#: so a Phase-2 run (or the UI, which calls the same helpers) would silently
#: build a SIX-annotator conflict graph and label it the v1 dev baseline. That
#: is the "the run believed it had a clean document" failure one level up, so
#: the filter is a DEFAULT, not a flag: pass `--annotators all` to opt out.
ANNOTATORS_OF_RECORD = (
    "cloud:qwen3.6:35b",
    "cloud:hf.co/unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL",
    "cloud:hf.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF:UD-Q4_K_M",
)

#: The Layer-2 arbitrator of record, decided 2026-09-22 (CHANGELOG): all four
#: arms share it, so the v1 dev Phase-2 is re-run under it. It is the CLI
#: default since 2026-09-23 — the default used to be the retired local
#: `qwen3.5:4b`, so a run that forgot `--layer2-model` produced a local-4B
#: conflict set that looks like every other one and must never be mixed with
#: the cloud-arbitrated set (BUG-2026-09-23-G). The interface reads this too.
LAYER2_OF_RECORD = "cloud:qwen3.6:35b"


def _safe_dir(model_name: str) -> str:
    """Directory `extractor.write_facts` writes a model's facts to."""
    return model_name.replace(":", "_").replace("/", "_")


def _allowed_dirs(annotators) -> set | None:
    """Directory names to accept, or None for 'every annotator on disk'."""
    if annotators is None:
        return None
    return {_safe_dir(a) for a in annotators}


def discover_annotators(facts_root: Path, doc_id: str,
                        annotators=ANNOTATORS_OF_RECORD) -> dict:
    """Facts per annotator for one doc. `annotators=None` disables the filter."""
    out: dict[str, list[dict]] = {}
    if not facts_root.exists():   # fresh guideline version: no extractions yet
        return out
    allowed = set(annotators) if annotators is not None else None
    allowed_dirs = _allowed_dirs(annotators)
    for sub in sorted(facts_root.iterdir()):
        if not sub.is_dir():
            continue
        fp = sub / f"{doc_id}.json"
        if not fp.exists():
            continue
        try:
            doc = json.loads(fp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        annotator = doc.get("annotator") or sub.name
        # Match on the recorded annotator name, and fall back to the directory
        # name, so a file written before the field existed is still filtered.
        if allowed is not None and annotator not in allowed                 and sub.name not in allowed_dirs:
            continue
        out[annotator] = doc.get("facts", [])
    return out


def discover_all_doc_ids(facts_root: Path,
                         annotators=ANNOTATORS_OF_RECORD) -> list:
    """Doc ids that have at least 2 annotators with facts on disk."""
    by_doc: dict[str, set[str]] = {}
    if not facts_root.exists():
        return []
    allowed_dirs = _allowed_dirs(annotators)
    for sub in sorted(facts_root.iterdir()):
        if not sub.is_dir():
            continue
        if allowed_dirs is not None and sub.name not in allowed_dirs:
            continue
        for fp in sub.glob("*.json"):
            by_doc.setdefault(fp.stem, set()).add(sub.name)
    return sorted(d for d, a in by_doc.items() if len(a) >= 2)


from src.entity_norm import normalize_entity, split_entity  # noqa: F401
# (re-exported: tests and ui import normalize_entity from this module)


def cluster_entities(facts_per_annotator: dict, *, merge_threshold: float = 0.78,
                     core_entities: bool = False) -> dict:
    surface_forms: list[str] = []
    seen: set[str] = set()
    for facts in facts_per_annotator.values():
        for f in facts:
            for k in ("subject", "object"):
                s = (f.get(k) or "").strip()
                if not s or s.lower() == "null" or s in seen:
                    continue
                seen.add(s)
                surface_forms.append(s)

    if not surface_forms:
        return {}

    keys = [normalize_entity(s) for s in surface_forms] if core_entities else surface_forms
    # These are entity strings, not facts. They used to be wrapped in sham fact
    # dicts carrying only `natural_language`, which worked solely because the
    # encoder fell back to that field — the same silent fallback that made every
    # gold-to-gold number a natural_language figure (BUG-2026-08-27-B), and which
    # `fact_text_report` now counts as `fell_back`. `encode_texts` is the entry
    # point for arbitrary strings and is byte-identical here under both backends.
    emb = encode_texts(keys)
    sims = emb @ emb.T

    n = len(surface_forms)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= merge_threshold:
                union(i, j)

    cluster_members: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        cluster_members[find(i)].append(i)

    surface_to_cluster: dict[str, str] = {}
    for root, members in cluster_members.items():
        rep = min((surface_forms[i] for i in members), key=lambda s: (len(s), s))
        cluster_id = "ent_" + re.sub(r"[^a-z0-9]+", "_", rep.lower())[:24].strip("_") + f"_{root:03d}"
        for m in members:
            surface_to_cluster[surface_forms[m]] = cluster_id
    return surface_to_cluster


_NUM_RE = re.compile(r"\d+[,.]?\d+\s*%")


def _stub_layer2(prompt: str, **_kw) -> str:
    nums = set(_NUM_RE.findall(prompt))
    if len(nums) >= 2:
        label = "CONTRADICTION"
    elif "split" in prompt.lower() or "qualified" in prompt.lower():
        label = "GRANULARITY"
    else:
        label = "REDUNDANCY"
    return json.dumps({"label": label, "reason": f"stub:{label.lower()}"})


def run(doc_id, *, facts_root, parsed_root, out_root, align_threshold,
        redundancy_cosine, merge_threshold, layer2_model, skip_layer2, layer2_url,
        out_suffix: str = "", arbitrate_version: str = "v1",
        one_to_many: bool = True, slot_gates: dict | None = None,
        arbitrate_shapes: tuple = ARBITRABLE_SHAPES_DEFAULT,
        annotators=ANNOTATORS_OF_RECORD,
        segment_population: dict | None = None):
    facts_per_annotator = discover_annotators(facts_root, doc_id,
                                              annotators=annotators)
    # RQ1's segment rule (method.tex sec:missing-segments): with a population,
    # every triple outside it is removed HERE, before alignment, because a
    # triple from a segment another annotator did not return would otherwise
    # become an orphan and move Phi. None = unrestricted (ordinary diagnosis,
    # and the sensitivity figure). Filtering the output afterwards cannot undo
    # the orphans, so there is no other place for this.
    population_summary = None
    if segment_population is not None:
        try:
            facts_per_annotator, population_summary = segpop.restrict_to_population(
                facts_per_annotator, doc_id, segment_population, facts_root=facts_root)
        except segpop.PopulationError as exc:
            raise SystemExit(f"[phase2] segment population refused {doc_id}: {exc}")
        d = population_summary["doc"]
        print(f"[phase2] segment population {population_summary['hash'][:12]} "
              f"({population_summary['version']}): {d['n_kept']}/{d['n_segments']} "
              f"segments kept, facts removed {d['facts_removed']}")
    if not facts_per_annotator:
        raise SystemExit(f"No annotators found under {facts_root} for doc {doc_id!r}")
    if len(facts_per_annotator) < 2:
        # A 1-annotator "comparison" has zero aligned pairs and would fabricate
        # a perfect-convergence result (observed when one extraction cell of a
        # pipeline run failed). Refuse instead of writing a misleading file.
        raise SystemExit(
            f"Only {len(facts_per_annotator)} annotator(s) for doc {doc_id!r} under "
            f"{facts_root} — need >= 2 for a meaningful conflict analysis. "
            f"(Did an extraction cell fail?)")

    print(f"[phase2] annotators: {list(facts_per_annotator.keys())}")
    print(f"[phase2] fact counts: { {a: len(f) for a,f in facts_per_annotator.items()} }")

    pairs = align_all_pairs(facts_per_annotator, threshold=align_threshold,
                            one_to_many=one_to_many, slot_gates=slot_gates)
    print(f"[phase2] aligned pairs: {len(pairs)} "
          f"({'one-to-many' if one_to_many else 'one-to-one (ablation)'})")

    l1_counts = classify_all(pairs, Layer1Config(redundancy_cosine=redundancy_cosine))
    print(f"[phase2] Layer-1 counts: {l1_counts}")

    # Groups are the unit of analysis (method.tex); the pair list stays for the
    # KG edges and the UI, which draw fact-to-fact links.
    groups = build_groups(pairs)
    g_counts_l1 = classify_groups(groups, Layer1Config(redundancy_cosine=redundancy_cosine))
    shapes: dict[str, int] = {}
    for g in groups:
        shapes[g.shape] = shapes.get(g.shape, 0) + 1
    print(f"[phase2] groups: {len(groups)} {shapes}")
    print(f"[phase2] group labels (Layer-1/shape): {g_counts_l1}")

    # How much of the one-to-many rate is the reification schema rather than a
    # disagreement: a fan-out entirely inside ONE annotation of one annotator is
    # that annotation's own triples, which share a natural_language and therefore
    # embed identically. Diagnostic only — no label changes on it.
    n_1n = shapes.get("one_to_many", 0)
    if n_1n:
        schema_fanout = sum(1 for g in groups
                            if group_to_dict(g)["partners_one_annotation"])
        print(f"[phase2] of {n_1n} one-to-many groups, {schema_fanout} "
              f"({schema_fanout / n_1n * 100:.0f}%) have every partner inside a "
              f"SINGLE annotation group — schema fan-out, not a granularity split")

    # One cache across BOTH passes: a one-to-one group and the pair it came from
    # ask the identical question, so the group pass is nearly free on top of the
    # pair pass instead of doubling the model calls.
    l2_cache: dict = {}
    title = doc_id
    if not skip_layer2:
        parsed_path = parsed_root / f"{doc_id}.json"
        if parsed_path.exists():
            parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
            title = parsed.get("document", {}).get("title", doc_id)
    l2_kwargs = (dict(chat_fn=_stub_layer2, doc_title=doc_id) if skip_layer2 else
                 dict(model_name=layer2_model, base_url=layer2_url, doc_title=title))
    tag = "STUB" if skip_layer2 else layer2_model

    # Group pass FIRST — it is the unit of analysis, and before 2026-08-08 it did
    # not exist: escalated one-to-one groups kept 'escalate' forever because only
    # the pair list was ever arbitrated.
    g_counts = arbitrate_groups(groups, version=arbitrate_version, cache=l2_cache,
                                shapes=arbitrate_shapes, **l2_kwargs)
    print(f"[phase2] Layer-2 ({tag}) group labels: "
          f"{ {k: v for k, v in g_counts.items() if not k.startswith('_')} } "
          f"[{g_counts.get('_layer2_calls', 0)} calls, shapes={list(arbitrate_shapes)}]")

    final = arbitrate_all(pairs, version=arbitrate_version, cache=l2_cache, **l2_kwargs)
    print(f"[phase2] Layer-2 ({tag}) pair labels: "
          f"{ {k: v for k, v in final.items() if not k.startswith('_')} } "
          f"[{final.get('_layer2_calls', 0)} calls, {len(l2_cache)} distinct comparisons]")

    surface_to_cluster = cluster_entities(facts_per_annotator,
                                          merge_threshold=merge_threshold,
                                          core_entities=True)
    n_clusters = len(set(surface_to_cluster.values()))
    print(f"[phase2] entity clusters: {n_clusters} (from {len(surface_to_cluster)} surface forms)")

    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / (f"{doc_id}__{out_suffix}.json"
                           if out_suffix else f"{doc_id}.json")
    payload = {
        "doc_id": doc_id,
        "params": {
            "align_threshold": align_threshold,
            "redundancy_cosine": redundancy_cosine,
            "merge_threshold": merge_threshold,
            "layer2_model": layer2_model if not skip_layer2 else "stub",
            "core_entities": True,
            "facts_root": str(facts_root),
            "variant": out_suffix or None,
            "alignment": "one_to_many" if one_to_many else "one_to_one",
            "slot_gates": slot_gates,
            "arbitrate_shapes": list(arbitrate_shapes),
            # None = unrestricted. Otherwise the population (hash, excluded
            # counts per version and annotator, this document's excluded
            # segments and removed facts); distribution_shift refuses to
            # compare files computed over different populations.
            "segment_population": population_summary,
        },
        "annotators": list(facts_per_annotator.keys()),
        "facts_per_annotator": facts_per_annotator,
        "aligned_pairs": [pair_to_dict(p) for p in pairs],
        "groups": [group_to_dict(g) for g in groups],
        "group_shapes": shapes,
        # Final (post-Layer-2) group distribution, plus the pre-arbitration one
        # so the arbitrator's contribution is readable off the file alone.
        "group_label_counts": {k: v for k, v in g_counts.items() if not k.startswith("_")},
        "group_label_counts_layer1": g_counts_l1,
        "group_layer2_calls": g_counts.get("_layer2_calls", 0),
        "entity_clusters": surface_to_cluster,
        "label_counts": {k: v for k, v in final.items() if not k.startswith("_")},
        "layer2_calls": final.get("_layer2_calls", 0),
        "layer2_distinct_comparisons": len(l2_cache),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[phase2] wrote {out_path}")
    try:
        from src.neo4j_export import autosync as _autosync
        _autosync(payload)
    except Exception as exc:  # never let neo4j break phase-2
        print(f"[phase2] neo4j autosync skipped: {exc}")
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--doc", default=None,
                   help="A single doc_id. Mutually exclusive with --all.")
    p.add_argument("--all", action="store_true",
                   help="Run every doc_id that has >=2 annotators under --facts-root. "
                        "NOTE: this sweeps the LOCKED EVAL documents too unless "
                        "--include-eval is given; see the guard below.")
    p.add_argument("--doc-list", type=Path, default=None,
                   help="a doc-list JSON (e.g. data/doc_lists/eurlex_dev.json); "
                        "run every doc in it that has >=2 annotators.")
    p.add_argument("--include-eval", action="store_true",
                   help="permit documents from data/doc_lists/eurlex_eval.json. "
                        "The locked eval set is touched ONCE per guideline version, "
                        "only to record results (discipline rule 1) — so this flag "
                        "is for that run and nothing else.")
    p.add_argument("--facts-root", type=Path, default=Path("data/facts"))
    p.add_argument("--annotators", default=None,
                   help="comma-separated annotator names to include. Default: the "
                        "set frozen 2026-09-03 (the three cloud 30B models). "
                        "`--annotators all` takes every directory under "
                        "--facts-root, which on this repo also picks up the three "
                        "retired local 4B pilots — only for an explicit ablation.")
    p.add_argument("--parsed-root", type=Path, default=Path("data/parsed"))
    p.add_argument("--out-root", type=Path, default=Path("data/conflicts"))
    # Frozen thresholds (CLAUDE.md discipline rule 4). These defaults used to be
    # 0.55 / 0.90 while ui/app.py and reextract_worker passed 0.78 / 0.95, so a
    # command-line run silently produced numbers from UNFROZEN parameters that
    # looked like everything else. Keep all three call sites on one value.
    p.add_argument("--align-threshold", type=float, default=0.78)
    p.add_argument("--redundancy-cosine", type=float, default=0.95)
    p.add_argument("--merge-threshold", type=float, default=0.78)
    p.add_argument("--layer2-model", default=LAYER2_OF_RECORD,
                   help="Layer-2 arbitrator. Default: the arbitrator of record "
                        f"({LAYER2_OF_RECORD}, decided 2026-09-22). Check "
                        "params.layer2_model before comparing conflict files.")
    p.add_argument("--layer2-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    p.add_argument("--skip-layer2", action="store_true",
                   help="Use deterministic stub instead of calling Ollama.")
    p.add_argument("--slot-gates", action="store_true",
                   help="require the frozen per-slot gates (subject/object >= 0.70, "
                        "predicate >= 0.45) in addition to the whole-fact cosine. "
                        "Without this, whole-string cosine alone labels ~40%% of "
                        "groups one-to-many on dev (runs/alignment_calibration_dev*.json) "
                        "because EUR-Lex facts share institutional vocabulary. "
                        "Off by default because method.tex still describes tau on "
                        "whole-fact cosine — turning it on is a method change.")
    p.add_argument("--out-suffix", default="",
                   help="write data/conflicts/<doc>__<suffix>.json instead of "
                        "<doc>.json, so a run lands beside the baseline instead "
                        "of replacing it and shows up in the UI's variant "
                        "dropdown. ui/app.py and reextract_worker already passed "
                        "this; only the CLI could not.")
    p.add_argument("--arbitrate-shapes", choices=("one_to_one", "all"),
                   default="one_to_one",
                   help="which GROUP shapes Layer-2 may judge. 'one_to_one' "
                        "(default) fills the gap Layer-2 exists for and changes "
                        "no method. 'all' additionally lets the model re-judge "
                        "one-to-many groups that the shape rule called "
                        "granularity — that is a METHOD CHANGE (method.tex "
                        "labels 1:n from shape) and is the experiment behind the "
                        "open linking-criterion question, not the run of record. "
                        "Overridden shape verdicts are kept in groups[].prior_label.")
    p.add_argument("--one-to-one", action="store_true",
                   help="Hungarian one-to-one assignment instead of one-to-many. "
                        "This is the prior-work ablation: it cannot express a "
                        "granularity split, so the granularity count will be ~0 "
                        "by construction. Do not use it for the run of record.")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip documents whose output file already exists. For "
                        "resuming an aborted batch without redoing hours of "
                        "arbitration; a document is only written once it is "
                        "complete, so an existing file is a finished one.")
    p.add_argument("--fail-fast", action="store_true",
                   help="Abort the batch on first doc failure (default: continue).")
    p.add_argument("--segment-population", type=Path, default=None,
                   help="RQ1 mode (method.tex sec:missing-segments): a population "
                        "from scripts/build_segment_population.py. Every triple "
                        "outside it is removed BEFORE alignment, and the population "
                        "is recorded in params. Requires --out-suffix, so a "
                        "restricted run never replaces the unrestricted baseline. "
                        "Default: unrestricted (ordinary diagnosis, and the "
                        "all-segments sensitivity figure).")
    args = p.parse_args()

    population = None
    if args.segment_population is not None:
        try:
            population = segpop.load_population(args.segment_population)
            pop_version = segpop.version_for_root(population, args.facts_root)
        except segpop.PopulationError as exc:
            p.error(str(exc))
        if not args.out_suffix:
            p.error("--segment-population requires --out-suffix")
        want = (set(ANNOTATORS_OF_RECORD) if args.annotators is None else
                {a.strip() for a in args.annotators.split(",") if a.strip()})
        if want != set(population["annotators"]):
            p.error(f"annotator set {sorted(want)} is not the population's "
                    f"{sorted(population['annotators'])}")
        print(f"[phase2] segment population {population['hash'][:12]} "
              f"({args.segment_population}): version {pop_version} of "
              f"{[v['label'] for v in population['versions']]}, "
              f"{population['n_segments']}/{population['n_segments_universe']} segments")

    if args.annotators is None:
        annotators = ANNOTATORS_OF_RECORD
    elif args.annotators.strip().lower() == "all":
        annotators = None
    else:
        annotators = tuple(a.strip() for a in args.annotators.split(",") if a.strip())
    print(f"[phase2] annotator set: "
          f"{'ALL on disk' if annotators is None else list(annotators)}")

    if sum(bool(x) for x in (args.all, args.doc, args.doc_list)) > 1:
        p.error("--all, --doc and --doc-list are mutually exclusive")

    if args.all:
        doc_ids = discover_all_doc_ids(args.facts_root, annotators)
        if not doc_ids:
            p.error(f"No docs with >=2 annotators found under {args.facts_root}")
        preview = doc_ids[:5]
        suffix = "..." if len(doc_ids) > 5 else ""
        print(f"[phase2-all] discovered {len(doc_ids)} doc(s): {preview}{suffix}")
    elif args.doc_list:
        payload = json.loads(args.doc_list.read_text(encoding="utf-8"))
        entries = payload["docs"] if isinstance(payload, dict) else payload
        listed = [d["doc_id"] if isinstance(d, dict) else str(d) for d in entries]
        have = set(discover_all_doc_ids(args.facts_root, annotators))
        doc_ids = [d for d in listed if d in have]
        print(f"[phase2-list] {args.doc_list.name}: {len(listed)} listed, "
              f"{len(doc_ids)} with >=2 annotators on disk")
        if not doc_ids:
            p.error(f"no doc in {args.doc_list} has >=2 annotators under {args.facts_root}")
    else:
        doc_ids = [args.doc or "train-000000"]

    # Discipline rule 1: the locked eval set is touched once per guideline
    # version, only to record results. `--all` used to sweep it silently, which
    # is how an eval document ended up in a threshold sweep on 2026-08-08.
    eval_list = Path("data/doc_lists/eurlex_eval.json")
    if eval_list.exists() and not args.include_eval:
        payload = json.loads(eval_list.read_text(encoding="utf-8"))
        entries = payload["docs"] if isinstance(payload, dict) else payload
        locked = {d["doc_id"] if isinstance(d, dict) else str(d) for d in entries}
        hit = sorted(set(doc_ids) & locked)
        if hit:
            kept = [d for d in doc_ids if d not in locked]
            print(f"[phase2] SKIPPING {len(hit)} LOCKED EVAL doc(s) "
                  f"(discipline rule 1; pass --include-eval for the run of "
                  f"record): {', '.join(hit[:5])}{'...' if len(hit) > 5 else ''}")
            doc_ids = kept
            if not doc_ids:
                p.error("every requested document is in the locked eval set")

    if population is not None:
        # Named, not silent: a document outside the population, or one where
        # no segment survived, contributes nothing to a restricted Phi.
        outside = [d for d in doc_ids if d not in population["segments"]]
        empty = [d for d in doc_ids
                 if d in population["segments"] and not population["segments"][d]]
        if outside or empty:
            print(f"[phase2] segment population: SKIPPING {len(outside)} doc(s) "
                  f"outside it {outside[:5]} and {len(empty)} with no segment "
                  f"left {empty[:5]}")
        doc_ids = [d for d in doc_ids if d not in outside and d not in empty]
        if not doc_ids:
            p.error("no requested document has a segment in the population")

    if args.skip_existing:
        def _out(d):
            return args.out_root / (f"{d}__{args.out_suffix}.json"
                                    if args.out_suffix else f"{d}.json")
        keep = [d for d in doc_ids if not _out(d).exists()]
        if len(keep) != len(doc_ids):
            print(f"[phase2] --skip-existing: {len(doc_ids) - len(keep)} "
                  f"document(s) already written, {len(keep)} to do")
        doc_ids = keep
        if not doc_ids:
            print("[phase2] nothing to do — every document is already written")
            return

    agg_counts: dict[str, int] = {}
    n_ok = 0
    n_err = 0
    for i, doc_id in enumerate(doc_ids, 1):
        if len(doc_ids) > 1:
            print(f"\n=== [{i}/{len(doc_ids)}] {doc_id} ===")
        try:
            out_path = run(
                doc_id,
                facts_root=args.facts_root,
                parsed_root=args.parsed_root,
                out_root=args.out_root,
                align_threshold=args.align_threshold,
                redundancy_cosine=args.redundancy_cosine,
                merge_threshold=args.merge_threshold,
                layer2_model=args.layer2_model,
                skip_layer2=args.skip_layer2,
                layer2_url=args.layer2_url,
                one_to_many=not args.one_to_one,
                out_suffix=args.out_suffix,
                slot_gates=SLOT_GATES if args.slot_gates else None,
                arbitrate_shapes=(("one_to_one", "one_to_many")
                                  if args.arbitrate_shapes == "all"
                                  else ARBITRABLE_SHAPES_DEFAULT),
                annotators=annotators,
                segment_population=population,
            )
            confl = json.loads(out_path.read_text(encoding="utf-8"))
            for k, v in confl.get("label_counts", {}).items():
                agg_counts[k] = agg_counts.get(k, 0) + v
            n_ok += 1
        except SystemExit:
            raise
        except Layer2Unavailable:
            # The arbitrator is down. Continuing would write documents whose
            # groups are all `escalate` and still report ok=N (2026-09-10).
            # --fail-fast must not be needed to stop this one.
            print(f"\n[phase2-all] ABORTING at {doc_id}: the Layer-2 "
                  f"arbitrator stopped answering. Documents already written "
                  f"are complete; restart the arbitrator and re-run with "
                  f"--skip-existing to resume.")
            raise
        except Exception as e:
            n_err += 1
            print(f"[phase2-all] FAILED on {doc_id}: {type(e).__name__}: {e}")
            if args.fail_fast:
                raise

    if len(doc_ids) > 1:
        print(f"\n[phase2-all] done. ok={n_ok}  err={n_err}")
        print(f"[phase2-all] aggregate label counts: {agg_counts}")


if __name__ == "__main__":
    main()
