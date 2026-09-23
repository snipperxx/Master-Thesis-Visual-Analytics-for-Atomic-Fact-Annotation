"""
Linearize annex table rows into self-contained sentences.

Why:
    A 4B model cannot reliably extract atomic facts from a flat
    "L-lysine 2b04 5 mg/kg poultry" string — it has no anchor for
    the column semantics. Re-attaching column headers as a sentence
    template restores that context.

Output unit:
    LinearizedRow = (sentence, table_locator_template)

The downstream extractor receives `sentence` as the LLM input, and
upon producing each fact, attaches the matching `TableLocator` (with
`col_indices` filled in based on which columns the fact references).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .schema import AnnexTable, TableLocator


@dataclass
class LinearizedRow:
    """One row's worth of LLM-ready input plus a locator stub."""

    sentence: str
    locator: TableLocator  # `col_indices` is filled in later by the extractor


#: Default sentence opener. Correct for the HTML/annex path, which is where
#: tables were first found. Callers handling tables that sit *inline* in an
#: article must pass their own template — claiming "In Annex ..." for an
#: article-internal table puts a false location into every fact derived from
#: it. Available fields: {annex_id}, {table_id}, {row_idx}.
DEFAULT_PREFIX_TEMPLATE = "In Annex {annex_id}, Table {table_id}, row {row_idx}: "


def linearize_table(table: AnnexTable, *,
                    prefix_template: str | None = None) -> List[LinearizedRow]:
    """Convert each row into a sentence using the column headers as labels."""
    out: List[LinearizedRow] = []
    template = prefix_template or DEFAULT_PREFIX_TEMPLATE

    has_headers = bool(table.headers) and any(h.strip() for h in table.headers)

    for row_idx, row in enumerate(table.rows):
        if has_headers:
            parts = []
            for header, cell in zip(table.headers, row):
                if not cell:
                    continue
                # Format: "<header> = <cell>"; chosen over "<header>: <cell>"
                # because colons appear inside legal text and confuse the LLM.
                parts.append(f"{header.rstrip(':')} = {cell}")
            body = "; ".join(parts) if parts else " | ".join(c for c in row if c)
        else:
            # No headers — fall back to a positional dump. Less helpful, but
            # at least the LLM sees structured separators.
            body = " | ".join(c for c in row if c)

        prefix = template.format(
            annex_id=table.annex_id or "(unnumbered)",
            table_id=table.table_id,
            row_idx=row_idx,
        )
        sentence = prefix + body + "."

        locator = TableLocator(
            annex_id=table.annex_id,
            table_id=table.table_id,
            row_idx=row_idx,
            col_indices=[],  # extractor fills this when emitting facts
            linearized_sentence=sentence,
        )
        out.append(LinearizedRow(sentence=sentence, locator=locator))

    return out
