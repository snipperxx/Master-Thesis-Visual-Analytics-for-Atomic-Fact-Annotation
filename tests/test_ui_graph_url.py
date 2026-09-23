"""The KG URL must be built in exactly one place.

`switchDoc` and `refreshGraph` each assembled their own `/api/graph/...` query,
and only `refreshGraph` put `reify` on it. The server defaults `reify=0`, so a
document *opened* on a flat graph while the "Reify (statement nodes)" box was
checked — and since the conflict-detection modal (the Layer-1 / Layer-2
rationale) only opens for a statement node or a double-tapped edge, the
arbitrator's reasons were unreachable until the checkbox was toggled, which
called `refreshGraph` and silently fixed the URL (BUG-2026-09-20-B).

There is no JS test runner in this project, so the guard is on the source: two
builders for one resource is the defect, and a second one would bring it back.
"""

from __future__ import annotations

from pathlib import Path

APP_JS = Path(__file__).resolve().parent.parent / "ui" / "static" / "app.js"


def test_the_graph_url_has_exactly_one_builder():
    src = APP_JS.read_text(encoding="utf-8")
    # Every knob-carrying /api/graph URL must come from graphUrl(); a second
    # place that spells out `merge_threshold=` is the drift this guards against.
    spellings = [ln.strip() for ln in src.splitlines() if "merge_threshold=" in ln]
    assert len(spellings) == 1, (
        "more than one place builds the graph URL with knobs: %r" % spellings)
    assert "function graphUrl(" in src
    builder = src[src.index("function graphUrl("):]
    builder = builder[:builder.index("\n}")]
    for knob in ("merge_threshold", "core", "reify"):
        assert knob in builder, f"graphUrl() dropped {knob!r}"


def test_both_graph_loads_go_through_it():
    src = APP_JS.read_text(encoding="utf-8")
    # the initial/document-change load and the knob refresh
    assert src.count("fetch(graphUrl(") == 2
