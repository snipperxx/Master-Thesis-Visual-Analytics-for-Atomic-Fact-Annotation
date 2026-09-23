"""Is the one-to-many rate a real granularity signal, or a copied text field?

The claim under test (CHANGELOG 2026-08-08): one extractor annotation is
`natural_language` + `source_quote` + N triples, flattening copies the SAME
natural_language onto each triple, alignment embeds natural_language, therefore
an annotation's own triples are embedding-identical, all link to the same
partner, and the group comes out one-to-many for a reason that has nothing to do
with content.

**That claim can be wrong in an important way.** If annotator B really did emit
three atomic facts where A emitted one, that IS a granularity difference —
regardless of how B's output is packaged. What is suspect is not the *shape* but
the *evidence*: the links were formed by a duplicated field, so we do not know
whether each triple individually corresponds to A's fact.

So the test is a falsification, not a demonstration. Re-embed on the triple
(`subject predicate object`), which is the one representation the duplication
cannot touch, and ask whether the suspect links survive:

  * if most survive          -> the fan-out is content-bearing, the claim is
                                WRONG and must be softened;
  * if they collapse, while
    links in NON-schema 1:n
    groups survive           -> the claim holds, and the difference between the
                                two categories is the evidence for it.

The control matters. Triples are shorter than sentences, so *all* cosines move
under this encoding; only the gap between the categories is interpretable.

Diagnostic only. Nothing here changes what alignment does — the shadow facts are
built locally and thrown away. Dev only (discipline rule 1).

    python -m scripts.check_schema_fanout
    python -m scripts.check_schema_fanout --n-docs 12 --out runs/x.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_phase2 import discover_annotators  # noqa: E402
from src.alignment import (align_all_pairs, backend_name, build_groups,  # noqa: E402
                           encode_facts_joint, group_to_dict)

DEV_LIST = ROOT / "data" / "doc_lists" / "eurlex_dev.json"
EVAL_LIST = ROOT / "data" / "doc_lists" / "eurlex_eval.json"


def _doc_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    docs = payload["docs"] if isinstance(payload, dict) else payload
    return [d["doc_id"] if isinstance(d, dict) else str(d) for d in docs]


def _triple_text(f: dict) -> str:
    return " ".join(str(f.get(k) or "").strip()
                    for k in ("subject", "predicate", "object")).strip()


def _shadow(facts: list[dict]) -> list[dict]:
    """Same facts, but `natural_language` replaced by the triple, so the
    unmodified encoder sees the triple. Local copies; nothing is written."""
    return [dict(f, natural_language=_triple_text(f)) for f in facts]


def _cos_lookup(fpa: dict[str, list[dict]], shadow: bool):
    """{(fact_id_a, fact_id_b): cosine} over every cross-annotator fact pair."""
    annots = list(fpa)
    lists = [(_shadow(fpa[a]) if shadow else fpa[a]) for a in annots]
    embs = encode_facts_joint(*lists)
    out: dict[tuple[str, str], float] = {}
    for i in range(len(annots)):
        for j in range(i + 1, len(annots)):
            cos = embs[i] @ embs[j].T
            for ia, fa in enumerate(lists[i]):
                for ib, fb in enumerate(lists[j]):
                    key = (fa.get("fact_id"), fb.get("fact_id"))
                    out[key] = float(cos[ia, ib])
                    out[(key[1], key[0])] = float(cos[ia, ib])
    return out


# --- The granularity population ------------------------------------------
# CHANGELOG 2026-08-08 (correction): the frozen per-slot gates looked like they
# fixed the 1:n rate, but they did it by CUTTING the part-inside-whole links that
# granularity consists of — 192 of 1219 dev links. Any replacement criterion has
# to be checked against the same population, or it repeats the mistake.

_STOP = {"the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "be",
         "is", "are", "shall", "must", "that", "this", "which", "by", "with",
         "as", "at", "from", "it", "its", "their", "such"}


def _content_words(s: str) -> set[str]:
    return {w for w in "".join(c.lower() if c.isalnum() else " " for c in (s or "")).split()
            if w not in _STOP and len(w) > 2}


def _is_containment(fa: dict, fb: dict, cover: float = 0.80) -> bool:
    """True when one side's content is a strict sub-part of the other's — the
    asymmetric relation a granularity split actually is."""
    wa, wb = _content_words(_triple_text(fa)), _content_words(_triple_text(fb))
    if not wa or not wb or wa == wb:
        return False
    small, big = (wa, wb) if len(wa) < len(wb) else (wb, wa)
    return len(small & big) / len(small) >= cover


def analyse_doc(doc_id: str, facts_root: Path, tau: float) -> dict | None:
    fpa = {k: v for k, v in discover_annotators(facts_root, doc_id).items()
           if not k.startswith("cloud")}
    if len(fpa) < 2:
        return None

    pairs = align_all_pairs(fpa, threshold=tau, one_to_many=True)
    groups = build_groups(pairs)

    # Baseline (NL) links, tagged by the category of the group they belong to.
    # "schema" = every partner of this 1:n group sits in ONE annotation group.
    links: dict[str, list[tuple[str, str]]] = defaultdict(list)
    shapes: Counter = Counter()
    for g in groups:
        d = group_to_dict(g)
        shapes[g.shape] += 1
        if g.shape == "orphan":
            continue
        cat = ("one_to_one" if g.shape == "one_to_one"
               else "schema_1n" if d["partners_one_annotation"] else "real_1n")
        anchor_id = g.anchor.get("fact_id")
        for p in g.partners:
            links[cat].append((anchor_id, p.get("fact_id")))

    tri = _cos_lookup(fpa, shadow=True)
    nl = _cos_lookup(fpa, shadow=False)

    out = {"doc_id": doc_id, "shapes": dict(shapes), "categories": {}}
    for cat, ls in links.items():
        tri_c = [tri.get(k, 0.0) for k in ls]
        nl_c = [nl.get(k, 0.0) for k in ls]
        if not ls:
            continue
        out["categories"][cat] = {
            "n_links": len(ls),
            "survive_at_tau": sum(1 for c in tri_c if c >= tau),
            "triple_cos_median": round(statistics.median(tri_c), 4),
            "nl_cos_median": round(statistics.median(nl_c), 4),
        }

    # Does the triple representation keep the part-inside-whole links, or does it
    # delete them the way the symmetric slot gates did?
    by_id = {f.get("fact_id"): f for fs in fpa.values() for f in fs}
    n_cont = n_cont_kept = 0
    for ls in links.values():
        for a_id, b_id in ls:
            fa, fb = by_id.get(a_id), by_id.get(b_id)
            if fa is None or fb is None or not _is_containment(fa, fb):
                continue
            n_cont += 1
            if tri.get((a_id, b_id), 0.0) >= tau:
                n_cont_kept += 1
    out["containment"] = {"n_links": n_cont, "kept_by_triple": n_cont_kept}
    return out


def shape_comparison(doc_ids: list[str], facts_root: Path, tau: float) -> list[dict]:
    """Group-shape distribution under each embedding text, everything else equal."""
    rows = []
    for encoding in ("natural_language", "triple"):
        shapes: Counter = Counter()
        fans: list[int] = []
        n_1n = n_schema = 0
        for doc_id in doc_ids:
            fpa = {k: v for k, v in discover_annotators(facts_root, doc_id).items()
                   if not k.startswith("cloud")}
            if len(fpa) < 2:
                continue
            if encoding == "triple":
                fpa = {k: _shadow(v) for k, v in fpa.items()}
            for g in build_groups(align_all_pairs(fpa, threshold=tau, one_to_many=True)):
                shapes[g.shape] += 1
                if g.fan_out:
                    fans.append(g.fan_out)
                if g.shape == "one_to_many":
                    n_1n += 1
                    if group_to_dict(g)["partners_one_annotation"]:
                        n_schema += 1
        total = sum(shapes.values()) or 1
        rows.append({
            "encoding": "NL" if encoding == "natural_language" else "triple",
            "n_groups": total,
            "orphan_pct": 100 * shapes["orphan"] / total,
            "one_to_one_pct": 100 * shapes["one_to_one"] / total,
            "one_to_many_pct": 100 * shapes["one_to_many"] / total,
            "fan_out_median": statistics.median(fans) if fans else 0,
            "fan_out_p90": (statistics.quantiles(fans, n=10)[8]
                            if len(fans) >= 10 else max(fans or [0])),
            "schema_pct_of_1n": 100 * n_schema / n_1n if n_1n else 0,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc-list", type=Path, default=DEV_LIST)
    ap.add_argument("--facts-root", type=Path, default=ROOT / "data" / "facts")
    ap.add_argument("--tau", type=float, default=0.78)
    ap.add_argument("--n-docs", type=int, default=37)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    doc_ids = _doc_ids(args.doc_list)[:args.n_docs]
    leaked = sorted(set(doc_ids) & set(_doc_ids(EVAL_LIST)))
    if leaked:
        raise SystemExit(
            "refusing to run: LOCKED EVAL documents must not be inspected "
            f"(discipline rule 1): {', '.join(leaked)}")

    print(f"backend={backend_name()}  tau={args.tau}  docs<={len(doc_ids)} (dev)\n")
    per_doc = []
    for doc_id in doc_ids:
        r = analyse_doc(doc_id, args.facts_root, args.tau)
        if r:
            per_doc.append(r)

    agg: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    tri_all: dict[str, list[float]] = defaultdict(list)
    for r in per_doc:
        for cat, c in r["categories"].items():
            agg[cat]["n_links"] += c["n_links"]
            agg[cat]["survive"] += c["survive_at_tau"]
            tri_all[cat].append(c["triple_cos_median"])

    order = ["one_to_one", "real_1n", "schema_1n"]
    print(f"{'link category':<14} {'links':>7} {'survive':>8} {'survival%':>10} "
          f"{'median triple cos (per-doc median)':>36}")
    for cat in order:
        if cat not in agg:
            continue
        a = agg[cat]
        pct = 100 * a["survive"] / a["n_links"] if a["n_links"] else 0
        med = statistics.median(tri_all[cat]) if tri_all[cat] else 0
        print(f"{cat:<14} {int(a['n_links']):>7} {int(a['survive']):>8} "
              f"{pct:>9.1f}% {med:>36.3f}")

    print("\nReading: `schema_1n` links were formed between facts whose "
          "natural_language is\nIDENTICAL by construction. If their survival "
          "rate under triple encoding is close to\n`real_1n`, the fan-out is "
          "content-bearing and the artefact claim is wrong.")

    # --- What the whole structure looks like under each representation -------
    print("\n\nGroup shape under each embedding text (same tau, same facts):\n")
    shape_rows = shape_comparison(doc_ids, args.facts_root, args.tau)
    print(f"{'embedding':<10} {'groups':>7} {'orphan%':>8} {'1:1%':>7} {'1:n%':>7} "
          f"{'fan med':>8} {'p90':>5} {'schema% of 1:n':>15}")
    for r in shape_rows:
        print(f"{r['encoding']:<10} {r['n_groups']:>7} {r['orphan_pct']:>7.1f}% "
              f"{r['one_to_one_pct']:>6.1f}% {r['one_to_many_pct']:>6.1f}% "
              f"{r['fan_out_median']:>8.0f} {r['fan_out_p90']:>5.0f} "
              f"{r['schema_pct_of_1n']:>14.1f}%")

    # --- The slot-gate trap: does the triple keep part-inside-whole links? ----
    cont = sum(r["containment"]["n_links"] for r in per_doc)
    kept = sum(r["containment"]["kept_by_triple"] for r in per_doc)
    print(f"\nContainment links (part inside whole — the granularity population "
          f"the frozen\nslot gates were shown to DELETE): {cont} in the baseline, "
          f"{kept} kept by the triple\nencoding ({100 * kept / cont if cont else 0:.1f}%). "
          f"Compare with the overall 1:n link survival\nabove — if containment "
          f"links survive at least as well, the triple representation does\nNOT "
          f"repeat the slot-gate mistake.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"backend": backend_name(), "tau": args.tau, "split": "dev",
             "n_docs": len(per_doc),
             "aggregate": {k: dict(v) for k, v in agg.items()},
             "shape_comparison": shape_rows,
             "containment": {"n_links": cont, "kept_by_triple": kept},
             "per_doc": per_doc}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
