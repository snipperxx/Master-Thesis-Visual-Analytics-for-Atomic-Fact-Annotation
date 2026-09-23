"""
JSON schema for atomic facts and parsed EUR-Lex documents.

Design notes (Phase 1):
- `AtomicFact` is the universal contract between LLM extractors and the KG.
- It supports two `source_locator` flavours so prose-derived and table-derived
  facts can co-exist in the same store without ad-hoc fields:
    * ProseLocator   — character offsets in flat text (RapidFuzz traceback)
    * TableLocator   — (annex_id, table_id, row_idx, col_indices)
- Adding new locator types later (e.g. for footnotes) only requires extending
  the Union; the rest of the pipeline does not need to change.
- Human-annotated facts can be ingested through this same schema, satisfying
  the "source-agnostic JSON schema" requirement in the proposal.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ConflictLabel(str, Enum):
    """Label assigned by Layer-1 filter or Layer-2 LLM arbitrator."""

    NO_CONFLICT = "no_conflict"
    REDUNDANCY = "redundancy"
    GRANULARITY = "granularity"
    CONTRADICTION = "contradiction"
    UNLABELED = "unlabeled"  # default before detection runs


class SourceType(str, Enum):
    PROSE = "prose"
    TABLE_ROW = "table_row"


# ---------------------------------------------------------------------------
# Source locators (discriminated union on `source_type`)
# ---------------------------------------------------------------------------


class ProseLocator(BaseModel):
    """Character-offset locator for facts extracted from running text."""

    source_type: Literal[SourceType.PROSE] = SourceType.PROSE
    section_path: str  # e.g. "preamble.recitals[3]" or "enacting.article_1.paragraph_2"
    char_start: int
    char_end: int
    quote: str  # the literal source span; redundancy is intentional for audit


class TableLocator(BaseModel):
    """Cell-coordinate locator for facts derived from an annex table row."""

    source_type: Literal[SourceType.TABLE_ROW] = SourceType.TABLE_ROW
    annex_id: str  # e.g. "I", "II"
    table_id: int  # 0-based index within the annex
    row_idx: int  # 0-based row index (excludes header)
    col_indices: List[int] = Field(default_factory=list)  # which columns the fact references
    linearized_sentence: str  # the template-generated sentence given to the LLM


SourceLocator = Union[ProseLocator, TableLocator]


# ---------------------------------------------------------------------------
# Atomic fact
# ---------------------------------------------------------------------------


class AtomicFact(BaseModel):
    """One atomic fact extracted by an annotator (LLM or human)."""

    fact_id: str  # stable hash, e.g. sha1(model + doc_id + index)
    doc_id: str  # CELEX number
    annotator: str  # model name (e.g. "qwen3.5-4b") or "human:initials"
    guideline_version: str  # e.g. "v1", "v2"

    # Triple form — minimum representation for KG ingestion.
    subject: str
    predicate: str
    object: str

    # Statement-level qualifiers (Property-Graph style), emitted by the annotator.
    # condition = logical precondition (if/when/under); temporal_context = ordering/deadline.
    condition: str = ""
    temporal_context: str = ""

    # Reification grouping: when one atomic annotation (a single natural_language /
    # source span) decomposes into several SPO triples, every triple shares this
    # group_id so the KG / Neo4j layer can group co-asserted statements.
    group_id: str = ""

    # Logical structure: facts that are branches/conjuncts of ONE logical
    # construction ("either A or B", "A and B", "if X then A or B") share a
    # logic_group id; logic_op names the relation among them. Lets the KG render
    # an AND/OR/XOR operator node instead of cramming "A or B" into one object
    # (the v1-v4 failure mode). Empty when the fact stands alone.
    logic_group: str = ""
    logic_op: str = ""  # "" | "AND" | "OR" | "XOR"

    # Free-text restatement (optional but recommended for LLM consumption
    # downstream, e.g. by the Layer-2 arbitrator).
    natural_language: Optional[str] = None

    # Where this fact came from in the source document.
    source_locator: SourceLocator

    # Conflict label assigned later in Phase 2; default UNLABELED.
    conflict_label: ConflictLabel = ConflictLabel.UNLABELED

    # Free-form metadata bag for downstream tooling.
    extra: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parsed-document containers (intermediate representation, pre-LLM)
# ---------------------------------------------------------------------------


class Recital(BaseModel):
    number: str  # "(1)", "(2)" — kept as string because EUR-Lex sometimes uses
    # roman numerals or letter suffixes.
    text: str
    char_start: int  # offsets into the flattened "preamble" text
    char_end: int


class Article(BaseModel):
    number: str  # "1", "2" — string for "1a" amendments etc.
    title: Optional[str]
    paragraphs: List[str]  # each numbered paragraph kept separately
    char_start: int  # offsets into the flattened "enacting" text
    char_end: int


class AnnexTable(BaseModel):
    annex_id: str
    table_id: int
    headers: List[str]  # column headers (possibly empty if not detected)
    rows: List[List[str]]  # each row's cell strings; row length == len(headers)


class Annex(BaseModel):
    annex_id: str  # "I", "II", or "" if unnumbered
    title: Optional[str]
    prose: str  # flattened prose surrounding the tables
    tables: List[AnnexTable]


class FlattenedTableRecord(BaseModel):
    """One OJ `1.2.3.4 // cell // cell` table that the parser rewrote.

    Kept so a later reader can tell which segment text is verbatim source and
    which was normalised, without re-running the detector. `ragged` marks a
    table whose row geometry could not be recovered; `merged_fill` marks one
    where a vertical cell span was carried down into the rows it covers.
    """

    field: str                  # "preamble_text" | "enacting_text"
    marker: str                 # "1.2.3.4"
    n_cols: int
    n_cells: int
    n_rows: int
    headers: List[str]
    ragged: bool
    merged_fill: bool
    raw: str                    # the original flattened line


class ParsedDocument(BaseModel):
    """Structural decomposition of a single EUR-Lex act."""

    celex: str
    title: str
    citations: List[str]  # raw "Having regard to..." paragraphs
    recitals: List[Recital]
    articles: List[Article]
    annexes: List[Annex]
    concluding_formulas: str  # "Done at Brussels..." trailer

    # Original cleaned text per section, to support char-offset locators.
    preamble_text: str
    enacting_text: str

    # Flattened-table normalisations applied to the two text fields above.
    # Defaults to empty so parsed JSON written before 2026-08-06 still loads.
    flattened_tables: List[FlattenedTableRecord] = []
