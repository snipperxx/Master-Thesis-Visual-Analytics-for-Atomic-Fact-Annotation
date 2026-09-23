"""
Phase-2/3 — Per-annotator coverage metrics.

What this measures
------------------
For a given (doc, annotator) pair, three numbers:

  sections_hit_frac    fraction of doc sections (recitals + articles) that
                       have at least one fact landing in them. Tells you
                       "did annotator X look at every section, or did it
                       give up after recital (3)?".

  char_coverage_frac   total characters spanned by the annotator's
                       source_quote ranges, divided by the total
                       (preamble + enacting) character length. Spans are
                       merged before summing so overlapping facts don't
                       double-count.

  mean_fact_chars      average char_end - char_start across the
                       annotator's facts; a sanity check for whether the
                       model is producing tiny "fragment" facts vs.
                       full-clause facts.

Usage
-----
    from src.coverage import compute_coverage

    cov = compute_coverage(
        facts_per_annotator,   # dict from data/conflicts/<doc>.json
        parsed_doc_json,        # the data/parsed/<doc>.json document
    )
    # cov = {"qwen3.5:4b": {"sections_hit_frac": 0.78, ...}, ...}

The UI calls this through GET /api/coverage/<doc_id> (Flask route added
separately). Keep this module pure — no I/O, no Flask deps — so it stays
unit-testable.
"""

from __future__ import annotations

from typing import Iterable


_PROSE = "prose"


def _container_for_section(section_path: str) -> str | None:
    if not section_path:
        return None
    if section_path.startswith("preamble"):
        return "preamble"
    if section_path.startswith("enacting"):
        return "enacting"
    return None


def _enumerate_sections(parsed_doc: dict) -> list[tuple[str, str]]:
    """Reproduce the section enumeration used by src/extractor.enumerate_sections.

    Returns a list of (container, section_path) pairs. We don't need the
    actual text here — we only need to know which paths exist so
    sections_hit_frac has a denominator.
    """
    doc = parsed_doc.get("document", parsed_doc)
    sections: list[tuple[str, str]] = []
    for i, _ in enumerate(doc.get("recitals", [])):
        sections.append(("preamble", f"preamble.recitals[{i}]"))
    for art in doc.get("articles", []):
        sections.append(("enacting", f"enacting.article_{art.get('number','?')}"))
    return sections


def _merge_intervals(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Classic interval merge so overlaps don't double-count chars."""
    sorted_spans = sorted((s for s in spans if s[1] > s[0]), key=lambda x: x[0])
    out: list[tuple[int, int]] = []
    for start, end in sorted_spans:
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def _container_lengths(parsed_doc: dict) -> dict[str, int]:
    """Char length of each container (preamble / enacting)."""
    doc = parsed_doc.get("document", parsed_doc)
    return {
        "preamble": len(doc.get("preamble_text", "") or ""),
        "enacting": len(doc.get("enacting_text", "") or ""),
    }


def compute_coverage(
    facts_per_annotator: dict[str, list[dict]],
    parsed_doc: dict,
) -> dict[str, dict]:
    """Per-annotator coverage. See module docstring."""
    sections = _enumerate_sections(parsed_doc)
    section_paths = [sp for _, sp in sections]
    n_sections = len(section_paths) or 1  # avoid div-by-0 on empty docs
    lengths = _container_lengths(parsed_doc)

    out: dict[str, dict] = {}
    for annotator, facts in facts_per_annotator.items():
        spans_per_container: dict[str, list[tuple[int, int]]] = {
            "preamble": [], "enacting": [],
        }
        sections_hit: set[str] = set()
        fact_lengths: list[int] = []

        for f in facts:
            loc = f.get("source_locator") or {}
            if loc.get("source_type") != _PROSE:
                continue
            section_path = loc.get("section_path", "")
            sections_hit.add(section_path)

            container = _container_for_section(section_path)
            cs = loc.get("char_start")
            ce = loc.get("char_end")
            if container is None or cs is None or ce is None:
                continue
            spans_per_container[container].append((int(cs), int(ce)))
            fact_lengths.append(max(0, int(ce) - int(cs)))

        # Merged chars across both containers
        total_chars_covered = 0
        total_chars_available = sum(lengths.values()) or 1
        for c in ("preamble", "enacting"):
            for start, end in _merge_intervals(spans_per_container[c]):
                total_chars_covered += max(0, min(end, lengths[c]) - max(start, 0))

        out[annotator] = {
            "n_facts": len(facts),
            "sections_hit": sorted(sections_hit),
            "sections_hit_frac": round(
                sum(1 for sp in section_paths if sp in sections_hit) / n_sections, 3
            ),
            "char_coverage_frac": round(total_chars_covered / total_chars_available, 3),
            "mean_fact_chars": round(
                sum(fact_lengths) / len(fact_lengths) if fact_lengths else 0.0, 1
            ),
            "total_chars_covered": total_chars_covered,
            "total_chars_available": total_chars_available,
        }
    return out


def aggregate_across_docs(per_doc_coverage: dict[str, dict[str, dict]]) -> dict[str, dict]:
    """Roll up many docs' coverage into a single per-annotator summary.

    Input shape:   {doc_id: {annotator: {metrics}}}
    Output shape:  {annotator: {mean_sections_hit_frac, mean_char_coverage_frac,
                                total_facts, n_docs}}

    Used by the future doc-list overview view; not by the per-doc UI.
    """
    by_ann: dict[str, list[dict]] = {}
    for cov in per_doc_coverage.values():
        for ann, metrics in cov.items():
            by_ann.setdefault(ann, []).append(metrics)
    rolled: dict[str, dict] = {}
    for ann, ms in by_ann.items():
        rolled[ann] = {
            "n_docs": len(ms),
            "total_facts": sum(m["n_facts"] for m in ms),
            "mean_sections_hit_frac": round(
                sum(m["sections_hit_frac"] for m in ms) / len(ms), 3
            ),
            "mean_char_coverage_frac": round(
                sum(m["char_coverage_frac"] for m in ms) / len(ms), 3
            ),
            "mean_fact_chars": round(
                sum(m["mean_fact_chars"] for m in ms) / len(ms), 1
            ),
        }
    return rolled
