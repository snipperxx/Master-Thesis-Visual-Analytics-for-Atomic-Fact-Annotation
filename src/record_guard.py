"""What an interactive run may not overwrite: the output of a batch run of record.

The runs the thesis reports are batch runs of `scripts/run_extraction.py`
(Phase 1) and `scripts/run_phase2.py` (Phase 2). The interface starts the same
code — a job, the per-segment probe, "Build KG" — and every one of those paths
used to write wherever its guideline version pointed. For v1 that is
`data/facts/` and `data/conflicts/<doc>.json`: the baseline of record itself.
A probe of one clause, or a stub Phase-2 behind a toolbar button, would have
replaced a file of the run every later comparison is measured against, and
nothing on disk would have said so (invariant 3; BUG-2026-09-23-D, -E).

What counts as "of record" is read off the data, not configured, so a new run
of record is protected the moment it exists:

  * a facts root is a **batch root** when it holds the `manifest.json` that
    `run_extraction.py` keeps (today `data/facts/` for v1 and `data/facts_v2/`
    for the running v2 pass); a (model, document) cell is of record when that
    manifest lists it, whatever its status — a failed cell is one the batch
    will retry and overwrite, and a file written into it meanwhile would be
    read by Phase-2 as if the batch had produced it;
  * a conflicts file is of record when it exists and was computed from a batch
    root (`params.facts_root`, which `run_phase2.run` records).

The interface's own roots (`data/facts__<version>/`, see
`reextract_worker.facts_root_for_version`) carry no manifest, so iterating on a
draft version stays free. Unreadable evidence is read as "of record": a guard
that opens when it cannot parse a file is not a guard.
"""

from __future__ import annotations

import json
from pathlib import Path

MANIFEST = "manifest.json"


def facts_file(root: Path | str, model: str, doc_id: str) -> Path:
    """Where `extractor.write_facts` puts (model, doc) under `root`.

    The directory keeps the model's dots (`cloud_qwen3.6_35b`). `ui/app.py`
    once rebuilt this path with a helper that also replaced dots, so for every
    annotator of record it looked in a directory that does not exist, found no
    file, and wrote a fresh one — replacing a whole document's facts with the
    probed clause's (BUG-2026-09-23-D). One formula, pinned to `write_facts`
    by a test.
    """
    safe = model.replace(":", "_").replace("/", "_")
    return Path(root) / safe / f"{doc_id}.json"


def is_batch_root(root: Path | str) -> bool:
    """True when `root` holds a batch manifest, i.e. is a run of record."""
    return (Path(root) / MANIFEST).exists()


def facts_cell_of_record(root: Path | str, model: str, doc_id: str) -> bool:
    """Would writing (model, doc) under `root` modify a batch run of record?"""
    path = Path(root) / MANIFEST
    if not path.exists():
        return False
    try:
        runs = json.loads(path.read_text(encoding="utf-8")).get("runs") or {}
    except (OSError, ValueError, AttributeError):
        # a manifest we cannot read still marks a batch root; without its list
        # of cells, every existing file in it has to be taken as of record
        return facts_file(root, model, doc_id).exists()
    return f"{model}|{doc_id}" in runs


def _resolve(path_str: str, repo_root: Path) -> Path:
    # run_phase2 records the path as given on the command line, and on this
    # machine that is `data\facts` — a Windows separator a POSIX Path would
    # read as part of one file name
    p = Path(str(path_str).replace("\\", "/"))
    return p if p.is_absolute() else Path(repo_root) / p


def conflicts_file_of_record(path: Path | str, repo_root: Path | str) -> bool:
    """Would overwriting this conflicts file replace a Phase-2 run of record?"""
    path = Path(path)
    if not path.exists():
        return False
    try:
        params = json.loads(path.read_text(encoding="utf-8")).get("params") or {}
    except (OSError, ValueError, AttributeError):
        return True
    facts_root = params.get("facts_root")
    if not facts_root:
        return True             # provenance unknown: not ours to overwrite
    return is_batch_root(_resolve(facts_root, Path(repo_root)))
