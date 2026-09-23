"""One-off helper for the 2026-09-02 cloud dev run.

The DBvis box is flaky under load: individual sections fail with
`RemoteDisconnected` (BUG-2026-08-31-D), and `extract()` skips just that
section instead of failing the whole document — so a document that lost
every section to network failure still lands in the manifest as
`status: ok, fact_count: 0`, indistinguishable from a real "the model found
nothing" result unless you read the log.

This scans a run's stderr log for "SKIPPED after a persistent network
failure" lines, matches them by model + time window against
data/facts/manifest.json entries, and deletes any contaminated entry so
`run_extraction run` (resumable) picks it up as pending again on the next
invocation. Never removes a clean entry. Prints what it removed.

Not part of the permanent pipeline; delete once this run is fully clean.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "facts" / "manifest.json"

SKIP_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \[ERROR\] \[(?P<model>[^\]]+)\] "
)


def parse_log_time(ts: str, local_offset: timedelta) -> datetime:
    naive = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    return naive.replace(tzinfo=timezone(local_offset))


def find_skip_windows(err_log_paths: list[Path], local_offset: timedelta) -> dict[str, list[datetime]]:
    """model -> sorted list of UTC timestamps where a section was skipped."""
    events: dict[str, list[datetime]] = {}
    for p in err_log_paths:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if "SKIPPED after a persistent network failure" not in line:
                continue
            m = SKIP_RE.match(line)
            if not m:
                continue
            ts = parse_log_time(m.group("ts"), local_offset).astimezone(timezone.utc)
            events.setdefault(m.group("model"), []).append(ts)
    for v in events.values():
        v.sort()
    return events


def main() -> None:
    err_logs = [ROOT / a for a in sys.argv[1:]]
    if not err_logs:
        print("usage: requeue_incomplete.py <err_log> [<err_log> ...]")
        raise SystemExit(1)

    # This machine's local clock runs 2h ahead of UTC (verified 2026-09-02:
    # `date -u` vs log timestamps). Logging uses local time, manifest uses
    # UTC ISO8601 with explicit offset.
    local_offset = timedelta(hours=2)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    runs = manifest["runs"]

    skip_events = find_skip_windows(err_logs, local_offset)

    removed = []
    for key, record in list(runs.items()):
        model = record["model"]
        events = skip_events.get(model)
        if not events:
            continue
        started = datetime.fromisoformat(record["started_at"])
        finished = datetime.fromisoformat(record["finished_at"]) if record.get("finished_at") else started
        # small margin: the SKIPPED line is logged slightly before the
        # document's own "Done" line, well inside [started, finished].
        if any(started <= ev <= finished for ev in events):
            removed.append((key, record.get("fact_count", 0)))
            del runs[key]

    if removed:
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"requeued {len(removed)} contaminated (model, doc) pair(s):")
    for key, fc in removed:
        print(f"  {key}  (was fact_count={fc})")


if __name__ == "__main__":
    main()
