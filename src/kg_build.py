"""
Phase-2/3 — Knowledge graph builder.

Reads the Phase-2 output (`data/conflicts/<doc_id>.json`) and produces a
Cytoscape.js-shaped JSON: { "nodes": [...], "edges": [...] }.

Conventions
-----------
* One **node** per entity cluster (subject-or-object surface forms that
  the SBERT/TF-IDF clusterer collapsed together at the configured
  threshold). Node `id` = cluster_id; node `label` = the shortest member
  surface form; node `data.surface_forms` lists every member with its
  annotator(s) so the UI can show "Council / Council of the EU / EC
  Council — 3 surface forms, 3 annotators".
* One **edge** per (s-cluster, predicate-string, o-cluster) triple.
  Multiple annotators producing the same triple collapse into one edge;
  their fact_ids accumulate in `edge.data.fact_ids` and their annotators
  in `edge.data.annotators`. This is what surfaces REDUNDANCY visually:
  one edge with three annotator chips, all agreeing.
* Conflict label on an edge = the **strongest** conflict observed on any
  AlignedPair whose two facts both ride this edge. Severity order:
        CONTRADICTION > GRANULARITY > REDUNDANCY > NO_CONFLICT > UNLABELED
  This is the rule the proposal calls "three-level node colouring" but
  applied at the edge — colouring the *relation* is what tells the
  analyst where to look. Node colour is derived from incident edges.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Iterable


_SEVERITY = {
    "contradiction": 4,
    "granularity": 3,
    "redundancy": 2,
    "no_conflict": 1,
    "escalate": 1.5,   # pending arbitration - visible, outranks no_conflict
    "unlabeled": 0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Leading function words stripped before edge-keying so surface/modal variants
# of the SAME relation collapse to one edge instead of N parallel edges
# ("may fix"/"fix"/"shall fix", "shall be determined"/"is determined"). Only
# LEADING words are removed — trailing prepositions carry meaning, so
# "are held by" vs "are held" stay distinct. Tunable: edit this set.
_PRED_STOP_LEAD = {
    "shall", "must", "may", "should", "will", "would", "can", "could",
    "is", "are", "be", "been", "being", "was", "were", "am",
    "has", "have", "had", "does", "do", "did",
    "hereby", "to", "the", "a", "an",
}


def _norm_predicate(pred: str) -> str:
    """Normalise a predicate for edge-keying. Lowercases, collapses whitespace,
    and strips LEADING modals/auxiliaries/determiners so "may fix" and "fix"
    key to the same edge. Keeps >=1 token. Conservative on purpose — a fuller
    lemmatiser belongs in a later iteration with proper evaluation."""
    toks = (pred or "").lower().split()
    while len(toks) > 1 and toks[0] in _PRED_STOP_LEAD:
        toks.pop(0)
    return " ".join(toks)


_COND_RE = re.compile(
    r"\b(if|when|unless|provided that|subject to|in the event|conditional upon)\b",
    re.IGNORECASE)


def _strongest_label(labels: Iterable[str]) -> str:
    best, best_sev = "unlabeled", -1
    for lbl in labels:
        sev = _SEVERITY.get(lbl, -1)
        if sev > best_sev:
            best, best_sev = lbl, sev
    return best


def _disp_label(s: str, n: int = 34) -> str:
    """Truncate a node's display label so a whole-clause surface form (the
    common small-model failure: a relative/scope clause dumped into the object)
    doesn't render as a giant unreadable node. Full text stays in
    data.full_label / data.surface_forms for the tooltip."""
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "\u2026"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_graph(conflicts_doc: dict) -> dict:
    """Translate a Phase-2 output document into Cytoscape elements."""
    surface_to_cluster: dict[str, str] = conflicts_doc["entity_clusters"]
    facts_per_ann: dict[str, list[dict]] = conflicts_doc["facts_per_annotator"]
    aligned_pairs = conflicts_doc.get("aligned_pairs", [])

    # 1. Build a fact-id -> fact lookup, and accumulate which annotators
    #    each cluster sees.
    fact_by_id: dict[str, dict] = {}
    cluster_members: dict[str, dict[str, set[str]]] = {}
    # cluster_id -> {"surface_forms": set, "annotators": set}

    def _bump_cluster(cluster: str, surface: str, annotator: str) -> None:
        slot = cluster_members.setdefault(cluster, {"surface_forms": set(), "annotators": set()})
        slot["surface_forms"].add(surface)
        slot["annotators"].add(annotator)

    for annotator, facts in facts_per_ann.items():
        for f in facts:
            fact_by_id[f["fact_id"]] = f
            for k in ("subject", "object"):
                surface = (f.get(k) or "").strip()
                if not surface or surface.lower() == "null":
                    continue
                cluster = surface_to_cluster.get(surface)
                if cluster is None:
                    continue
                _bump_cluster(cluster, surface, annotator)

    # 2. Build the edge index keyed by (s_cluster, predicate, o_cluster).
    edges: dict[tuple[str, str, str], dict] = {}

    def _edge_for(f: dict) -> tuple[str, str, str] | None:
        subj = (f.get("subject") or "").strip()
        obj = (f.get("object") or "").strip()
        if not subj or not obj or obj.lower() == "null":
            return None
        s_c = surface_to_cluster.get(subj)
        o_c = surface_to_cluster.get(obj)
        if not s_c or not o_c:
            return None
        return (s_c, _norm_predicate(f.get("predicate", "")), o_c)

    for annotator, facts in facts_per_ann.items():
        for f in facts:
            key = _edge_for(f)
            if key is None:
                continue
            slot = edges.setdefault(
                key,
                {
                    "fact_ids": set(),
                    "annotators": set(),
                    "conflict_labels": set(),
                    "section_paths": set(),
                    "qualifiers": set(),
                    "group_ids": set(),
                },
            )
            slot["fact_ids"].add(f["fact_id"])
            slot["annotators"].add(annotator)
            _gid = (f.get("group_id") or "").strip()
            if _gid:
                slot["group_ids"].add(_gid)
            # Wikidata-style: modifiers stripped from S/O (dates, clause tails)
            # belong to the STATEMENT, i.e. this edge — not to the entity node.
            from src.entity_norm import split_entity
            for k in ("subject", "object"):
                for q in split_entity(f.get(k) or "")[1]:
                    if q:
                        slot["qualifiers"].add(q[:60])
            # Conditional statements (poster's If/Or concept, edge-level):
            # v1 §6 keeps the antecedent inside source_quote, so detect it
            # there and mark the edge — rendered dashed in the UI.
            _txt = f"{(f.get('source_locator') or {}).get('quote', '')} "                    f"{f.get('natural_language', '')}"
            cond_field = (f.get("condition") or "").strip()
            if cond_field or _COND_RE.search(_txt):
                slot["conditional"] = True
            if cond_field:
                slot.setdefault("conditions", set()).add(cond_field[:80])
            _tctx = (f.get("temporal_context") or "").strip()
            if _tctx:
                slot.setdefault("temporal", set()).add(_tctx[:60])
            loc = f.get("source_locator") or {}
            sp = loc.get("section_path")
            if sp:
                slot["section_paths"].add(sp)

    # 3. Attach conflict labels: walk aligned_pairs, find each pair's
    #    edge (both sides should hit the *same* edge under REDUNDANCY /
    #    CONTRADICTION; under GRANULARITY they may diverge — in that case
    #    we mark the A-edge with GRANULARITY because the divergence is
    #    visible there).
    for pair in aligned_pairs:
        label = pair.get("conflict_label", "unlabeled")
        if label == "unlabeled":
            continue   # 'escalate' stays visible: Layer-2 pending/failed is signal
        fa = fact_by_id.get(pair.get("fact_a_id") or "")
        fb = fact_by_id.get(pair.get("fact_b_id") or "")
        for f in (fa, fb):
            if not f:
                continue
            key = _edge_for(f)
            if key in edges:
                edges[key]["conflict_labels"].add(label)

    # 4. Serialize. Strongest label dictates colour; counts decide width.
    cy_nodes: list[dict] = []
    node_label_severity: dict[str, str] = {}
    for cid, info in cluster_members.items():
        cy_nodes.append(
            {
                "data": {
                    "id": cid,
                    "label": _disp_label(min(info["surface_forms"], key=lambda s: (len(s), s))),
                    "full_label": min(info["surface_forms"], key=lambda s: (len(s), s)),
                    "surface_forms": sorted(info["surface_forms"]),
                    "annotators": sorted(info["annotators"]),
                    "n_surface": len(info["surface_forms"]),
                    "n_annotators": len(info["annotators"]),
                    "conflict_label": "unlabeled",  # filled in below
                }
            }
        )

    cy_edges: list[dict] = []
    for (s_c, pred, o_c), info in edges.items():
        label = _strongest_label(info["conflict_labels"]) if info["conflict_labels"] else "unlabeled"
        cy_edges.append(
            {
                "data": {
                    "id": f"e_{s_c}__{pred[:24]}__{o_c}".replace(" ", "_"),
                    "source": s_c,
                    "target": o_c,
                    "label": pred,
                    "annotators": sorted(info["annotators"]),
                    "fact_ids": sorted(info["fact_ids"]),
                    "group_ids": sorted(info.get("group_ids", set()))[:8],
                    "section_paths": sorted(info["section_paths"]),
                    "qualifiers": sorted(info.get("qualifiers", set()))[:6],
                    "conditional": bool(info.get("conditional", False)),
                    "conditions": sorted(info.get("conditions", set()))[:4],
                    "temporal": sorted(info.get("temporal", set()))[:4],
                    "conflict_label": label,
                    "n_annotators": len(info["annotators"]),
                }
            }
        )
        # Propagate severity to incident nodes
        sev = _SEVERITY.get(label, 0)
        for endpoint in (s_c, o_c):
            cur = node_label_severity.get(endpoint, "unlabeled")
            if sev > _SEVERITY.get(cur, 0):
                node_label_severity[endpoint] = label

    for node in cy_nodes:
        nid = node["data"]["id"]
        node["data"]["conflict_label"] = node_label_severity.get(nid, "unlabeled")

    return {
        "doc_id": conflicts_doc["doc_id"],
        "params": conflicts_doc.get("params", {}),
        "nodes": cy_nodes,
        "edges": cy_edges,
        "summary": {
            "n_nodes": len(cy_nodes),
            "n_edges": len(cy_edges),
            "edge_label_counts": _count_by_label(cy_edges),
        },
    }


def _count_by_label(edges: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in edges:
        lbl = e["data"]["conflict_label"]
        out[lbl] = out.get(lbl, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Reified (statement-node) graph — mirrors the Neo4j model in neo4j_export.py.
# Each fact-group becomes a :statement node between its subject/object entity
# nodes, carrying predicate + condition + temporal + logic flags. Keying the
# statement on (subject, predicate, object, condition, temporal) keeps facts
# that share an S-P-O but differ in their CONDITION as *separate* statement
# nodes — so conditional / ordering structure stays visible instead of being
# flattened into one edge (the failure mode of the flat build_graph above).
# ---------------------------------------------------------------------------

import hashlib

_NEG_RE = re.compile(
    r"\b(not|no longer|never|without|fails? to|shall not|may not|cannot)\b",
    re.IGNORECASE)
_MODAL_RE = re.compile(
    r"\b(alleged(?:ly)?|reportedly|according to|claims? that|is said to)\b",
    re.IGNORECASE)
_COMPLEX_RE = re.compile(
    r"\b(either|or else|whether|all of|each of|none of|any of|"
    r"if and only if|implies)\b",
    re.IGNORECASE)


def _norm_text(s: str) -> str:
    return " ".join((s or "").lower().split())


def _reified_fact_text(f: dict) -> str:
    loc = f.get("source_locator") or {}
    quote = loc.get("quote") or loc.get("linearized_sentence") or ""
    return f"{quote} {f.get('natural_language', '') or ''}"


def _stmt_id(s_c, pred_norm, o_c, cond_key, temp_key) -> str:
    raw = "|".join([s_c or "", pred_norm or "", o_c or "", cond_key or "", temp_key or ""])
    return "st_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:14]


def _resolve_reference(text, surface_to_cluster, exclude):
    """Find an entity cluster *referenced inside* a condition/temporal phrase.

    Returns the cluster_id of the longest surface form that occurs as a
    substring of `text`, skipping clusters in `exclude` (the statement's own
    subject/object, to avoid trivial self-loops). This is what turns a bare
    `condition` string into a first-class edge so the reader can see WHICH
    entity a precondition / ordering depends on.
    """
    if not text:
        return None
    low = text.lower()
    best, best_len = None, 0
    for surface, cid in surface_to_cluster.items():
        if cid in exclude:
            continue
        s = (surface or "").strip()
        if len(s) < 5:
            continue
        if s.lower() in low and len(s) > best_len:
            best, best_len = cid, len(s)
    return best


def build_reified_graph(conflicts_doc: dict) -> dict:
    """Statement-reified Cytoscape graph. See module note above.

    Node kinds (data.type):
        entity     — one per resolved cluster (same identity as build_graph);
                     `ref_only` entities are referenced ONLY by a condition /
                     temporal clause, never asserted as a subject/object.
        statement  — one per (s,p,o,condition,temporal) fact-group
    Edge roles (data.role):
        subject    entity --> statement
        object     statement --> entity
        condition  statement --> referenced entity   (dashed; the precondition)
        temporal   statement --> referenced entity   (ordering / deadline)
    The conflict label lives on the STATEMENT node (strongest of its facts'
    own labels plus any aligned-pair label touching its fact_ids), and is
    mirrored onto incident edges so the existing annotator/conflict filters
    keep working unchanged.
    """
    surface_to_cluster: dict = conflicts_doc.get("entity_clusters", {})
    facts_per_ann: dict = conflicts_doc.get("facts_per_annotator", {})
    aligned_pairs = conflicts_doc.get("aligned_pairs", [])

    # invert clusters so a condition-referenced cluster (one that never appears
    # as a subject/object) can still be reconstructed into a labelled node.
    cluster_surfaces: dict = {}
    for surface, cid in surface_to_cluster.items():
        cluster_surfaces.setdefault(cid, set()).add(surface)

    def cluster_of(surface):
        s = (surface or "").strip()
        if not s or s.lower() == "null":
            return None
        return surface_to_cluster.get(s)

    cluster_members: dict = {}
    fact_by_id: dict = {}

    def _bump(cid, surface, annot):
        slot = cluster_members.setdefault(cid, {"surface_forms": set(), "annotators": set()})
        slot["surface_forms"].add(surface)
        slot["annotators"].add(annot)

    stmts: dict = {}
    stmt_of_fact: dict = {}
    spo_variants: dict = {}
    logic_groups: dict = {}   # (annotator, logic_group) -> {stmts:set, ops:{}, annot}

    for annot, facts in facts_per_ann.items():
        for f in facts:
            fid = f["fact_id"]
            fact_by_id[fid] = f
            subj = (f.get("subject") or "").strip()
            obj = (f.get("object") or "").strip()
            s_c = cluster_of(subj)
            o_c = cluster_of(obj)
            if s_c:
                _bump(s_c, subj, annot)
            if o_c:
                _bump(o_c, obj, annot)
            pred = f.get("predicate", "") or ""
            pred_norm = _norm_predicate(pred)
            text = _reified_fact_text(f)
            cond = (f.get("condition") or "").strip()
            temp = (f.get("temporal_context") or "").strip()
            key = _stmt_id(s_c, pred_norm, o_c, _norm_text(cond), _norm_text(temp))
            stmt_of_fact[fid] = key
            st = stmts.get(key)
            if st is None:
                st = stmts[key] = {
                    "id": key, "s_c": s_c, "o_c": o_c,
                    "predicate": pred, "predicate_norm": pred_norm,
                    "fact_ids": set(), "annotators": set(), "group_ids": set(),
                    "conditions": set(), "temporal": set(),
                    "conflict_labels": set(), "section_paths": set(),
                    "natural_language": f.get("natural_language") or "",
                    "subject_surface": subj, "object_surface": obj,
                    "conditional": False, "negated": False,
                    "complex": False, "modality": "asserted",
                }
            st["fact_ids"].add(fid)
            st["annotators"].add(annot)
            _lg = (f.get("logic_group") or "").strip()
            _lop = (f.get("logic_op") or "").strip().upper()
            if _lg and _lop in ("AND", "OR", "XOR"):
                lgslot = logic_groups.setdefault((annot, _lg),
                                                 {"stmts": set(), "ops": {}, "annot": annot})
                lgslot["stmts"].add(key)
                lgslot["ops"][_lop] = lgslot["ops"].get(_lop, 0) + 1
            _gid = (f.get("group_id") or "").strip()
            if _gid:
                st["group_ids"].add(_gid)
            if not st["predicate"] and pred:
                st["predicate"], st["predicate_norm"] = pred, pred_norm
            if not st["natural_language"]:
                st["natural_language"] = f.get("natural_language") or ""
            if cond:
                st["conditions"].add(cond[:120])
            if temp:
                st["temporal"].add(temp[:80])
            lbl = f.get("conflict_label")
            if lbl and lbl not in ("unlabeled", "escalate"):
                st["conflict_labels"].add(lbl)
            loc = f.get("source_locator") or {}
            if loc.get("section_path"):
                st["section_paths"].add(loc["section_path"])
            if cond or _COND_RE.search(text):
                st["conditional"] = True
            if _NEG_RE.search(text):
                st["negated"] = True
            if _COMPLEX_RE.search(text):
                st["complex"] = True
            if _MODAL_RE.search(text):
                st["modality"] = "alleged"
            spo_variants.setdefault((s_c, pred_norm, o_c), set()).add(key)

    # Same (s,p,o) realized with >1 distinct condition/temporal scope: a
    # "conditional-scope" signal the flat graph cannot show (it would collapse
    # them to one edge and call it redundancy).
    for (s_c, _pn, o_c), keys in spo_variants.items():
        if s_c and o_c and len(keys) > 1:
            for k in keys:
                stmts[k]["scope_variant"] = True

    for p in aligned_pairs:
        label = p.get("conflict_label", "unlabeled")
        if label in ("unlabeled", "escalate", None):
            continue
        for fk in (p.get("fact_a_id"), p.get("fact_b_id")):
            k = stmt_of_fact.get(fk or "")
            if k:
                stmts[k]["conflict_labels"].add(label)

    # ----- statement nodes (also computes conflict severity for entities) -----
    node_label_sev: dict = {}
    stmt_nodes = []
    for key, st in stmts.items():
        label = _strongest_label(st["conflict_labels"]) if st["conflict_labels"] else "unlabeled"
        stmt_nodes.append({"data": {
            "id": key, "type": "statement",
            "label": st["predicate"] or "—",
            "predicate": st["predicate"],
            "fact_ids": sorted(st["fact_ids"]),
            "annotators": sorted(st["annotators"]),
            "n_annotators": len(st["annotators"]),
            "group_ids": sorted(st["group_ids"]),
            "n_groups": len(st["group_ids"]),
            "conditions": sorted(st["conditions"])[:4],
            "temporal": sorted(st["temporal"])[:4],
            "conditional": bool(st["conditional"]) or bool(st["conditions"]),
            "negated": st["negated"], "complex": st["complex"],
            "modality": st["modality"],
            "scope_variant": bool(st.get("scope_variant", False)),
            "conflict_label": label,
            "natural_language": st["natural_language"],
            "section_paths": sorted(st["section_paths"]),
            "subject_label": st["subject_surface"],
            "object_label": st["object_surface"],
        }})
        sev = _SEVERITY.get(label, 0)
        for c in (st["s_c"], st["o_c"]):
            if c and sev > _SEVERITY.get(node_label_sev.get(c, "unlabeled"), 0):
                node_label_sev[c] = label

    # ----- edges (condition/temporal may PROMOTE a referenced entity) -----
    referenced = set()
    for st in stmts.values():
        if st["s_c"]:
            referenced.add(st["s_c"])
        if st["o_c"]:
            referenced.add(st["o_c"])

    edges = []
    cond_referenced = set()

    def _edge(eid, src, tgt, role, st, label=""):
        edges.append({"data": {
            "id": eid, "source": src, "target": tgt, "role": role, "label": label,
            "annotators": sorted(st["annotators"]),
            "conflict_label": _strongest_label(st["conflict_labels"]) if st["conflict_labels"] else "unlabeled",
            "fact_ids": sorted(st["fact_ids"]),
            "conditional": role == "condition",
        }})

    for key, st in stmts.items():
        if st["s_c"]:
            _edge(f"se_{key}", st["s_c"], key, "subject", st)
        if st["o_c"]:
            _edge(f"oe_{key}", key, st["o_c"], "object", st)
        if st["conditions"]:
            ref = _resolve_reference(" ".join(st["conditions"]), surface_to_cluster,
                                     exclude={st["s_c"], st["o_c"]})
            if ref:
                cond_referenced.add(ref)
                _edge(f"ce_{key}", key, ref, "condition", st, label="if")
        if st["temporal"]:
            ref = _resolve_reference(" ".join(st["temporal"]), surface_to_cluster,
                                     exclude={st["s_c"], st["o_c"]})
            if ref:
                cond_referenced.add(ref)
                _edge(f"te_{key}", key, ref, "temporal", st, label="then")

    # ----- entity nodes for every referenced cluster (asserted or promoted) ---
    all_referenced = referenced | cond_referenced
    entity_nodes = []
    for cid in all_referenced:
        info = cluster_members.get(cid)
        sforms = (info["surface_forms"] if info else cluster_surfaces.get(cid)) or {cid}
        annots = sorted(info["annotators"]) if info else []
        entity_nodes.append({"data": {
            "id": cid, "type": "entity",
            "label": _disp_label(min(sforms, key=lambda s: (len(s), s))),
            "full_label": min(sforms, key=lambda s: (len(s), s)),
            "surface_forms": sorted(sforms),
            "annotators": annots,
            "n_surface": len(sforms),
            "n_annotators": max(1, len(annots)),
            "ref_only": cid not in referenced,
            "conflict_label": node_label_sev.get(cid, "unlabeled"),
        }})

    # ----- logical-operator nodes (AND/OR/XOR among branch statements) -----
    # Facts the annotator marked as branches of one construction ("either A or
    # B") share a (annotator, logic_group); render a small operator node joining
    # their statements so the KG SHOWS the disjunction instead of burying
    # "A or B" inside one object (the v1-v4 failure).
    logic_nodes = []
    for (annot, lg), slot in logic_groups.items():
        member_keys = {k for k in slot["stmts"] if k in stmts}
        if len(member_keys) < 2:
            continue
        op = max(slot["ops"].items(), key=lambda kv: kv[1])[0] if slot["ops"] else "OR"
        lid = "lo_" + hashlib.sha1(f"{annot}|{lg}".encode("utf-8")).hexdigest()[:12]
        mlabels = set()
        for k in member_keys:
            mlabels |= stmts[k]["conflict_labels"]
        logic_nodes.append({"data": {
            "id": lid, "type": "logic", "op": op, "label": op,
            "annotators": [annot], "n_branches": len(member_keys),
            "conflict_label": _strongest_label(mlabels) if mlabels else "unlabeled",
        }})
        for k in member_keys:
            edges.append({"data": {
                "id": f"loe_{lid}_{k}", "source": lid, "target": k,
                "role": "branch", "label": "", "op": op,
                "annotators": [annot], "conflict_label": "unlabeled",
            }})

    nodes = entity_nodes + stmt_nodes + logic_nodes
    return {
        "doc_id": conflicts_doc["doc_id"],
        "params": conflicts_doc.get("params", {}),
        "reified": True,
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "n_nodes": len(nodes),
            "n_entities": len(entity_nodes),
            "n_statements": len(stmt_nodes),
            "n_logic": len(logic_nodes),
            "n_edges": len(edges),
            "edge_label_counts": _count_by_label(stmt_nodes),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", type=Path, default=Path("data/conflicts/train-000000.json"))
    p.add_argument("--out", type=Path, default=None,
                   help="Default: data/graphs/<doc_id>.json")
    args = p.parse_args()

    doc = json.loads(args.inp.read_text(encoding="utf-8"))
    graph = build_graph(doc)

    out = args.out or Path("data/graphs") / f"{doc['doc_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[kg_build] {graph['summary']}")
    print(f"[kg_build] wrote {out}")


if __name__ == "__main__":
    main()
