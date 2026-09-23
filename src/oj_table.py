"""Reconstruct tables that the OJ plain-text export flattened into `//` runs.

Background (2026-08-06)
-----------------------
The HuggingFace `coastalcph/lex_glue` eurlex text carries no `<table>`
structure, but it does not *drop* tables either: the Official Journal
plain-text export encodes them inline on a single line, like

    1.2.3.4 // // // // // Order No // Category (unit) // CN code // ...

The leading dotted token is a **column marker**: `1.2.3.4` declares four
columns, `1.2.3.4.5.6` declares six. Everything after it is the grid in
row-major order, one cell per `//`, *including empty cells* — the empties
carry the geometry, so they must not be discarded.

Two documents in the frozen EUR-Lex corpus contain such tables, both in the
locked eval split: `train-000019` (articles 1 and 2) and `train-000121`
(article 1). Gold decision log D-01 records that the pre-annotation treated
the raw `//` run as opaque; this module is the parser-side fix so the run
becomes readable sentences instead.

Honesty constraint
------------------
A grid is only reconstructed when the cell count divides evenly by the
declared column count. When it does not — which happens when the source
table had merged or spanning cells — the table is marked `ragged` and the
row structure is NOT fabricated. Emitting a confidently wrong grid ("row 1,
column 6 = '- 66/402'") is worse than emitting nothing, because a downstream
annotator cannot tell it is wrong. Ragged tables are rendered as an explicit
cell list instead, and callers can branch on the flag.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .schema import AnnexTable
from .table_linearize import linearize_table

# A column marker sitting at the start of a line and followed by the first
# cell separator. Anchored to line start so a dotted cross-reference inside
# prose ("point 1.1", "Article 18(5)") can never match.
_MARKER_RE = re.compile(r"^[ \t]*(?P<marker>\d+(?:\.\d+)+)[ \t]*(?=//)", re.MULTILINE)


@dataclass
class FlattenedTable:
    """One reconstructed table plus the span it occupied in the source text."""

    char_start: int          # offset of the marker in the text passed to the finder
    char_end: int            # offset just past the end of the table's line
    marker: str              # "1.2.3.4"
    n_cols: int
    headers: List[str]       # [] when the source used positional (numeric) labels
    rows: List[List[str]]    # data rows, all-empty rows removed; [] when ragged
    cells: List[str] = field(default_factory=list)   # every cell, empties kept
    ragged: bool = False     # True => row geometry could not be recovered
    merged_fill: bool = False  # True => a vertical cell span was carried down
    raw: str = ""            # the original line, for provenance

    @property
    def n_data_cells(self) -> int:
        return sum(1 for c in self.cells if c)


def _marker_cols(marker: str) -> Optional[int]:
    """Column count from a dotted marker, or None if it is not 1..n."""
    try:
        parts = [int(p) for p in marker.split(".")]
    except ValueError:
        return None
    if parts != list(range(1, len(parts) + 1)):
        return None
    if len(parts) < 2:
        return None
    return len(parts)


def _looks_positional(header_row: List[str]) -> bool:
    """True when the header row is just column numbers ("1 // 2 // 3 // 4").

    Those are placeholders in the source layout, not labels, so they must not
    be reattached as "1 = 'U" — that reads as a fact about the number 1.
    """
    present = [c for c in header_row if c]
    return bool(present) and all(c.isdigit() for c in present)


def _forward_fill_leading(rows: List[List[str]]) -> tuple[List[List[str]], bool]:
    """Carry a vertically merged grouping cell down the rows it spans.

    The OJ export emits a merged cell once and leaves the spanned rows empty,
    so row 2 of a table whose first two columns say "Uruguay / Ministerio ..."
    arrives as ``["", "", "- 66/402", "- Basic seed", "B", ""]``. Rendered as
    is, that row has no subject to attach a fact to — the very
    decontextualization failure the fact schema forbids.

    Only the *leading* run of empty cells is filled. A merge in the left-hand
    grouping columns is the pattern these tables actually use; filling
    interior or trailing gaps would invent data (in `train-000019` the
    6th-column value "(d)" belongs to one row only, and forward-filling it
    downward would be simply wrong).
    """
    filled: List[List[str]] = []
    prev: List[str] = []
    touched = False

    for row in rows:
        new = list(row)
        if prev:
            for i, cell in enumerate(new):
                if cell:
                    break                      # leading run is over
                if i < len(prev) and prev[i]:
                    new[i] = prev[i]
                    touched = True
        filled.append(new)
        prev = new

    return filled, touched


def find_flattened_tables(text: str) -> List[FlattenedTable]:
    """Locate every flattened OJ table in `text`, in document order."""
    out: List[FlattenedTable] = []

    for m in _MARKER_RE.finditer(text):
        n_cols = _marker_cols(m.group("marker"))
        if n_cols is None:
            continue

        # The table occupies the remainder of the marker's line.
        line_end = text.find("\n", m.end())
        if line_end == -1:
            line_end = len(text)
        rest = text[m.end():line_end]

        # Each `//` introduces one cell, so the fragment before the first
        # separator is an artefact of str.split and is dropped.
        parts = rest.split("//")
        if len(parts) < 2:
            continue
        cells = [c.strip() for c in parts[1:]]

        ragged = (len(cells) % n_cols) != 0

        headers: List[str] = []
        rows: List[List[str]] = []
        merged_fill = False
        if not ragged:
            grid = [cells[i:i + n_cols] for i in range(0, len(cells), n_cols)]
            grid = [r for r in grid if any(c for c in r)]      # drop blank rows
            if grid:
                first, body = grid[0], grid[1:]
                if _looks_positional(first):
                    headers, rows = [], body
                else:
                    headers, rows = first, body
                rows, merged_fill = _forward_fill_leading(rows)

        out.append(
            FlattenedTable(
                char_start=m.start(),
                char_end=line_end,
                marker=m.group("marker"),
                n_cols=n_cols,
                headers=headers,
                rows=rows,
                cells=cells,
                ragged=ragged,
                merged_fill=merged_fill,
                raw=text[m.start():line_end],
            )
        )

    return out


def to_annex_table(ft: FlattenedTable, *, annex_id: str = "", table_id: int = 0) -> AnnexTable:
    """Adapt a reconstructed table to the schema `table_linearize` consumes.

    `AnnexTable.rows` requires `len(row) == len(headers)`, so a positional
    table (no usable labels) is given empty headers and `linearize_table`
    falls back to its `" | ".join(...)` positional rendering.
    """
    headers = list(ft.headers)
    rows = [list(r) for r in ft.rows]
    if headers:
        width = len(headers)
        rows = [(r + [""] * width)[:width] for r in rows]
    return AnnexTable(annex_id=annex_id, table_id=table_id, headers=headers, rows=rows)


def render(ft: FlattenedTable, *, annex_id: str = "", table_id: int = 0,
           label: str = "table") -> str:
    """Readable replacement text for one flattened table.

    Well-formed tables become one linearized sentence per row (the form
    `table_linearize` was written for). Ragged tables get an explicit,
    clearly-labelled cell dump instead of an invented grid.
    """
    if ft.ragged:
        present = [c for c in ft.cells if c]
        return (
            f"The following {ft.n_cols}-column {label} could not be reconstructed "
            f"into rows because the source layout uses merged or spanning cells; "
            f"its cells are, in order: " + "; ".join(present) + "."
        )

    at = to_annex_table(ft, annex_id=annex_id, table_id=table_id)
    # These tables sit inline in the article, not in an annex, so the default
    # "In Annex ..." opener would put a location in the text that is not true.
    lines = [lr.sentence for lr in linearize_table(
        at, prefix_template=f"In the {label} set out in this Article, row {{row_idx}}: ")]
    if not lines:
        return f"The following {ft.n_cols}-column {label} contained no data rows."
    return " ".join(lines)


def normalize_flattened_tables(text: str, *, annex_id: str = "",
                               label: str = "table") -> tuple[str, List[FlattenedTable]]:
    """Replace every flattened table in `text` with its readable rendering.

    Returns the rewritten text and the tables that were found. Replacement is
    done back-to-front so earlier spans keep their offsets while iterating.
    Callers must treat the returned text as the new offset base: everything
    downstream (sections, `source_quote` anchoring) is computed against it.
    """
    tables = find_flattened_tables(text)
    if not tables:
        return text, []

    out = text
    for table_id, ft in reversed(list(enumerate(tables))):
        replacement = render(ft, annex_id=annex_id, table_id=table_id, label=label)
        out = out[:ft.char_start] + replacement + out[ft.char_end:]

    return out, tables
