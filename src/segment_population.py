"""
The segment population of an RQ1 comparison (method.tex, sec:missing-segments).

The rule, pre-registered 2026-09-22 before any g_v2 conflict file existed: for
a comparison between guideline versions v and w, Phi_v and Phi_w are computed
over the segments (doc_id + section_path) for which EVERY annotator of record
returned a validated output under BOTH versions, and triples from any other
segment are removed BEFORE alignment. Reported alongside: the excluded segments
per annotator and per version, and the shift over all segments as a
sensitivity check. The restricted figure is primary.

Why the filter has to sit before alignment: alignment is pairwise over a whole
document (`alignment.align_all_pairs`, `build_groups`), so every triple from a
segment one annotator did not return has nothing to align with in that pair
and becomes an orphan, and orphans are part of Phi (`distribution_shift`).
Filtering the conflict file afterwards cannot undo that: the orphan groups are
already there, and so are the matches the missing annotator's absence did not
create.

"Returned" versus "not returned"
--------------------------------
A segment for which the model answered `{"facts": []}` (a bare header, a
citation, a closing formula) RETURNED a validated output. It has no triples,
and it is in the population. Only a segment the run itself recorded as lost is
out: every correction turn failed to parse (`SectionUnparsed`, "unparsed"), or
the call failed on the network ("network"). The explicit record is used, never
the absence of triples, because absence cannot tell those two apart
(`train-000085 preamble.recitals[4]`, qwen, v1: "0 facts from 0 group(s)").

Where the record comes from, in order of preference:

  1. `failed_sections` in the per-document facts file (written with the facts,
     since 2026-09-23, by `extractor.write_facts`);
  2. `failed_sections` in the manifest record (same date, `run_extraction`);
  3. the manifest record's `error` string, which since BUG-2026-09-04 has named
     every skipped section as `<path> (unparsed)` / `<path> (network)`; the
     number parsed must equal `incomplete_sections`, or this module refuses.
     This is how every existing run (v1 dev, the running v2 dev pass) is read.

A record with none of the three is a LEGACY record (written before
2026-09-04), when a skipped section could be silently recorded `ok`. For those
only, a section absent from the facts file cannot be verified as returned and
is excluded with reason `legacy_unverified`. That is the one inference this
module makes, and on the v1 dev pass it touches nothing: the two legacy
records (qwen, train-000006 and train-000061, 2026-08-31) cover every section.

A document the annotator never produced a facts file for is out entirely:
`document_failed` (the run failed as a whole) or `no_record` (never run). The
second means the extraction pass is not finished, so building a population
over it is refused unless explicitly allowed.

Identity
--------
A population is a JSON artefact with a content hash over its versions,
annotators, segments and the sha256 of every facts file it was derived from.
Phase-2 verifies those file hashes before it filters (a re-extraction after
the population was built is refused, not silently mixed in) and records the
population hash in the conflicts file's `params`; `distribution_shift` refuses
to compare two files computed over different populations.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent

SCHEMA = "segment_population/1"

RULE = ("method.tex sec:missing-segments: Phi_v and Phi_w are computed over the "
        "segments for which every annotator returned a validated output under "
        "every listed version; triples from any other segment are removed "
        "before alignment.")

#: Why a (version, annotator, segment) is not returned.
REASONS = ("failed_section", "document_failed", "no_record", "legacy_unverified")

#: Causes of a failed section, as `extractor.extract` records them.
CAUSES = ("unparsed", "network")

_ENTRY = re.compile(r"^\s*(\S+)\s+\((unparsed|network)\)\s*$")
_ENTRY_IN_ERROR = re.compile(r"([A-Za-z_][\w.\[\]]*) \((unparsed|network)\)")


class PopulationError(ValueError):
    """The records cannot support a population, or a population does not fit."""


# ---------------------------------------------------------------------------
# Failed-section records
# ---------------------------------------------------------------------------


def failed_section_records(incomplete: Iterable[str], *, strict: bool = True) -> list[dict]:
    """Structured form of `extractor.extract(..., incomplete_out=...)` entries.

    `"enacting.article_1 (unparsed)"` -> `{"section_path": "enacting.article_1",
    "cause": "unparsed"}`. An entry of any other shape raises, because a record
    this module cannot read is a record it would silently drop. With
    `strict=False` (the extraction driver, which must never lose finished
    facts over bookkeeping) it is kept verbatim with cause `unknown`, and the
    population builder refuses it later, loudly, because the path is not a
    section of the document.
    """
    out = []
    for entry in incomplete:
        m = _ENTRY.match(entry)
        if m:
            out.append({"section_path": m.group(1), "cause": m.group(2)})
        elif strict:
            raise PopulationError(f"unrecognised incomplete-section entry: {entry!r}")
        else:
            out.append({"section_path": entry, "cause": "unknown"})
    return out


def failed_sections_from_error(error: str | None, expected: int) -> list[dict]:
    """Recover the failed sections a pre-2026-09-23 manifest record names.

    `run_extraction` has written `INCOMPLETE: N section(s) skipped ...
    (<path> (<cause>), ...); ...` since BUG-2026-09-04. The count must match
    `incomplete_sections`, or the string is not the record it claims to be.
    """
    found = [{"section_path": p, "cause": c}
             for p, c in _ENTRY_IN_ERROR.findall(error or "")]
    if len(found) != expected:
        raise PopulationError(
            f"manifest record says incomplete_sections={expected} but its error "
            f"string names {len(found)} section(s): {error!r}")
    return found


def _explicit_failed(record: dict | None, payload: dict | None) -> list[dict] | None:
    """The run's own record of failed sections, or None for a legacy record."""
    from_file = payload.get("failed_sections") if payload else None
    from_record = record.get("failed_sections") if record else None
    if from_file is not None and from_record is not None:
        if _paths(from_file) != _paths(from_record):
            raise PopulationError(
                f"facts file and manifest disagree on failed sections: "
                f"{sorted(_paths(from_file))} vs {sorted(_paths(from_record))}")
    if from_file is not None:
        return list(from_file)
    if from_record is not None:
        return list(from_record)
    if record is not None and "incomplete_sections" in record:
        n = int(record.get("incomplete_sections") or 0)
        return failed_sections_from_error(record.get("error"), n) if n else []
    return None


def _paths(entries: Iterable[dict]) -> set[str]:
    return {e["section_path"] for e in entries}


# ---------------------------------------------------------------------------
# Paths and hashing
# ---------------------------------------------------------------------------


def safe_dir(model_name: str) -> str:
    """Directory `extractor.write_facts` writes a model's facts to."""
    return model_name.replace(":", "_").replace("/", "_")


def _stored(p: Path) -> str:
    """A path as recorded in a population: repo-relative when possible."""
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _resolve_stored(s: str) -> Path:
    p = Path(s)
    return (p if p.is_absolute() else ROOT / p).resolve()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_hash(population: dict) -> str:
    core = {
        "schema": population["schema"],
        "versions": [(v["label"], v.get("guideline_version"))
                     for v in population["versions"]],
        "annotators": sorted(population["annotators"]),
        "segments": {d: sorted(p) for d, p in sorted(population["segments"].items())},
        "doc_sections": {d: sorted(p) for d, p in
                         sorted(population["doc_sections"].items())},
        "sources": population["sources"],
    }
    blob = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def fact_segment(fact: dict) -> str | None:
    """The section_path a fact was extracted from (None for a table locator)."""
    loc = fact.get("source_locator") or {}
    return loc.get("section_path")


# ---------------------------------------------------------------------------
# Building a population
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VersionSource:
    """Where one guideline version's extraction output lives."""

    label: str
    facts_root: Path
    manifest: Path | None = None   # default: <facts_root>/manifest.json

    @property
    def manifest_path(self) -> Path:
        return Path(self.manifest) if self.manifest else Path(self.facts_root) / "manifest.json"


def sections_from_parsed(parsed_root: Path, doc_ids: Iterable[str]) -> dict[str, list[str]]:
    """Every segment of every document, exactly as extraction enumerated them."""
    from src.extractor import enumerate_sections, load_parsed_doc
    out = {}
    for d in doc_ids:
        fp = Path(parsed_root) / f"{d}.json"
        if not fp.exists():
            raise PopulationError(f"parsed document missing: {fp}")
        out[d] = [s.section_path for s in enumerate_sections(load_parsed_doc(fp))]
    return out


def classify_returns(record: dict | None, payload: dict | None, sections: list[str],
                     *, facts_file: Path | None = None) -> dict[str, tuple[str, str]]:
    """Which sections of one (version, annotator, doc) were NOT returned.

    Returns `{section_path: (reason, detail)}`; every section not in the result
    was returned (possibly with zero triples). `payload` is the parsed facts
    file, or None when there is none on disk.
    """
    if record is None:
        if payload is None:
            return {s: ("no_record", "no manifest record and no facts file")
                    for s in sections}
        # A facts file nobody recorded running: provenance unknown. Refuse
        # rather than guess; this is what an interactive re-extraction that
        # bypassed the manifest looks like.
        raise PopulationError(f"facts file with no manifest record: {facts_file}")
    if record.get("status") not in ("ok", "failed"):
        raise PopulationError(f"unknown manifest status {record.get('status')!r}")
    if not record.get("output_path"):
        # The run failed before it wrote anything. A facts file on disk is
        # then left over from an earlier run and describes nothing current.
        why = record.get("error") or "run failed before writing facts"
        return {s: ("document_failed", str(why)[:200]) for s in sections}
    if payload is None:
        raise PopulationError(
            f"manifest record points at {record['output_path']}, which is not on "
            f"disk (expected {facts_file})")

    facts = payload.get("facts", [])
    if len(facts) != int(record.get("fact_count", -1)):
        raise PopulationError(
            f"manifest fact_count={record.get('fact_count')} but {facts_file} holds "
            f"{len(facts)} facts: the record does not describe the file on disk")
    if payload.get("annotator") and payload["annotator"] != record.get("model"):
        raise PopulationError(
            f"{facts_file} is by {payload['annotator']!r}, manifest says "
            f"{record.get('model')!r}")
    if facts_file is not None and record.get("output_path"):
        described = _resolve_stored(record["output_path"])
        if described != Path(facts_file).resolve():
            raise PopulationError(
                f"manifest output_path {record['output_path']} is not {facts_file}")

    present = {fact_segment(f) for f in facts}
    failed = _explicit_failed(record, payload)
    if failed is None:          # legacy record: the only inference in this module
        return {s: ("legacy_unverified",
                    "record predates failed-section bookkeeping and the section "
                    "has no triples") for s in sections if s not in present}
    out: dict[str, tuple[str, str]] = {}
    for e in failed:
        path, cause = e["section_path"], e.get("cause", "")
        if path not in sections:
            raise PopulationError(f"{facts_file}: failed section {path!r} is not a "
                                  f"section of the document")
        if path in present:
            raise PopulationError(f"{facts_file}: section {path!r} is recorded as "
                                  f"failed but has triples in the facts file")
        out[path] = ("failed_section", cause)
    return out


def build_population(versions: list[VersionSource],
                     doc_sections: dict[str, list[str]],
                     annotators: Iterable[str],
                     *,
                     allow_missing_records: bool = False) -> dict:
    """The segments every annotator returned under every version.

    `doc_sections` is the universe (`sections_from_parsed`). Raises
    `PopulationError` on any record it cannot trust, and on a (version,
    annotator, doc) with no record at all unless `allow_missing_records`.
    """
    annotators = list(annotators)
    if not versions:
        raise PopulationError("at least one version is required")
    if len({v.label for v in versions}) != len(versions):
        raise PopulationError("version labels must be distinct")
    if len({Path(v.facts_root).resolve() for v in versions}) != len(versions):
        raise PopulationError("two versions share one facts root")
    if not annotators:
        raise PopulationError("at least one annotator is required")

    version_meta, excluded, sources = [], [], {}
    out_by_segment: dict[tuple[str, str], list[dict]] = {}
    for v in versions:
        mpath = v.manifest_path
        if not mpath.exists():
            raise PopulationError(f"no manifest for version {v.label}: {mpath}")
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        runs = manifest.get("runs", {})
        # The label IS the guideline version: pinning it here is what stops a
        # population labelled v2 from being built over the control's facts.
        seen_gv = ({r.get("guideline_version") for r in runs.values()}
                   | {manifest.get("guideline_version")}) - {None}
        if seen_gv != {v.label}:
            raise PopulationError(f"version label {v.label!r} but {mpath} records "
                                  f"guideline_version(s) {sorted(seen_gv)}")
        gv = v.label
        version_meta.append({"label": v.label, "guideline_version": gv,
                             "facts_root": _stored(v.facts_root),
                             "manifest": _stored(mpath)})
        sources[v.label] = {}
        for ann in annotators:
            sources[v.label][ann] = {}
            for doc_id, sections in doc_sections.items():
                record = runs.get(f"{ann}|{doc_id}")
                fp = Path(v.facts_root) / safe_dir(ann) / f"{doc_id}.json"
                payload = None
                if fp.exists():
                    try:
                        payload = json.loads(fp.read_text(encoding="utf-8"))
                    except json.JSONDecodeError as exc:
                        raise PopulationError(f"unreadable facts file {fp}: {exc}") from exc
                    if payload.get("guideline_version") not in (None, v.label):
                        raise PopulationError(
                            f"{fp} is guideline {payload.get('guideline_version')!r}, "
                            f"not {v.label!r}")
                sources[v.label][ann][doc_id] = file_sha256(fp) if fp.exists() else None
                if record is None and payload is None and not allow_missing_records:
                    raise PopulationError(
                        f"no record for {ann} x {doc_id} under {v.label}: the "
                        f"extraction pass is not finished (pass "
                        f"allow_missing_records to build over it anyway)")
                lost = classify_returns(record, payload, sections, facts_file=fp)
                for path, (reason, detail) in sorted(lost.items()):
                    e = {"doc_id": doc_id, "section_path": path, "version": v.label,
                         "annotator": ann, "reason": reason, "detail": detail}
                    excluded.append(e)
                    out_by_segment.setdefault((doc_id, path), []).append(e)

    segments = {d: [s for s in secs if (d, s) not in out_by_segment]
                for d, secs in doc_sections.items()}
    counts = {v.label: {a: 0 for a in annotators} for v in versions}
    by_reason: dict[str, int] = {}
    by_cause: dict[str, int] = {}
    for e in excluded:
        counts[e["version"]][e["annotator"]] += 1
        by_reason[e["reason"]] = by_reason.get(e["reason"], 0) + 1
        if e["reason"] == "failed_section":
            by_cause[e["detail"]] = by_cause.get(e["detail"], 0) + 1
    n_universe = sum(len(s) for s in doc_sections.values())
    n_kept = sum(len(s) for s in segments.values())
    population = {
        "schema": SCHEMA,
        "rule": RULE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "versions": version_meta,
        "annotators": annotators,
        "doc_ids": list(doc_sections),
        "n_segments_universe": n_universe,
        "n_segments": n_kept,
        "n_segments_excluded": n_universe - n_kept,
        # per (version, annotator): segments that annotator did not return
        "excluded_counts": counts,
        "excluded_by_reason": by_reason,
        "excluded_by_cause": by_cause,
        "excluded": excluded,
        "segments": segments,
        "doc_sections": {d: list(s) for d, s in doc_sections.items()},
        "sources": sources,
        "complete_passes": "no_record" not in by_reason,
    }
    population["hash"] = _canonical_hash(population)
    return population


def write_population(population: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(population, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_population(path: Path) -> dict:
    """Read a population and verify its hash (a hand-edited file is refused)."""
    population = json.loads(Path(path).read_text(encoding="utf-8"))
    if population.get("schema") != SCHEMA:
        raise PopulationError(f"{path} is not a {SCHEMA} file")
    if _canonical_hash(population) != population.get("hash"):
        raise PopulationError(f"{path}: content does not match its hash; rebuild it")
    population["_path"] = _stored(Path(path))
    return population


# ---------------------------------------------------------------------------
# Applying a population (Phase-2)
# ---------------------------------------------------------------------------


def version_for_root(population: dict, facts_root: Path) -> str:
    """The population version whose facts live under `facts_root`."""
    want = Path(facts_root).resolve()
    for v in population["versions"]:
        if _resolve_stored(v["facts_root"]) == want:
            return v["label"]
    raise PopulationError(
        f"facts root {facts_root} is none of the population's versions "
        f"({', '.join(v['label'] + '=' + v['facts_root'] for v in population['versions'])})")


def restrict_to_population(facts_per_annotator: dict[str, list[dict]], doc_id: str,
                           population: dict, *, facts_root: Path,
                           ) -> tuple[dict[str, list[dict]], dict]:
    """Drop every fact outside the population, BEFORE alignment.

    Verifies first that the facts files on disk are the ones the population was
    built from. Returns the filtered facts and the summary Phase-2 records in
    its `params.segment_population`.
    """
    version = version_for_root(population, facts_root)
    if doc_id not in population["segments"]:
        raise PopulationError(f"{doc_id} is not in the population")
    pop_anns = list(population["annotators"])
    for ann in pop_anns:
        fp = Path(facts_root) / safe_dir(ann) / f"{doc_id}.json"
        now = file_sha256(fp) if fp.exists() else None
        was = population["sources"][version][ann].get(doc_id)
        if now != was:
            raise PopulationError(
                f"{fp} changed since the population was built "
                f"({(was or 'absent')[:12]} -> {(now or 'absent')[:12]}); rebuild it")
    extra = set(facts_per_annotator) - set(pop_anns)
    if extra:
        raise PopulationError(f"annotators outside the population: {sorted(extra)}")

    keep = set(population["segments"][doc_id])
    missing = set(pop_anns) - set(facts_per_annotator)
    if missing and keep:
        raise PopulationError(
            f"{doc_id}: the population keeps {len(keep)} segment(s) but "
            f"{sorted(missing)} have no facts on disk")

    sections = set(population["doc_sections"][doc_id])
    filtered, removed, kept, outside = {}, {}, {}, {}
    for ann, facts in facts_per_annotator.items():
        filtered[ann] = [f for f in facts if fact_segment(f) in keep]
        removed[ann] = len(facts) - len(filtered[ann])
        kept[ann] = len(filtered[ann])
        outside[ann] = sum(1 for f in facts if fact_segment(f) not in sections)

    doc_excl = [e for e in population["excluded"] if e["doc_id"] == doc_id]
    summary = {
        "hash": population["hash"],
        "path": population.get("_path"),
        "version": version,
        "versions": [v["label"] for v in population["versions"]],
        "annotators": pop_anns,
        "n_segments": population["n_segments"],
        "n_segments_universe": population["n_segments_universe"],
        "excluded_counts": population["excluded_counts"],
        "doc": {
            "n_segments": len(population["doc_sections"][doc_id]),
            "n_kept": len(keep),
            "excluded_sections": sorted({e["section_path"] for e in doc_excl}),
            "excluded": [{k: e[k] for k in ("section_path", "version", "annotator",
                                             "reason", "detail")} for e in doc_excl],
            "facts_kept": kept,
            "facts_removed": removed,
            "facts_outside_universe": outside,
        },
    }
    return filtered, summary


def summary_of(conflicts_doc: dict) -> dict | None:
    """The population a conflicts file was computed over, or None (unrestricted)."""
    return (conflicts_doc.get("params") or {}).get("segment_population")
