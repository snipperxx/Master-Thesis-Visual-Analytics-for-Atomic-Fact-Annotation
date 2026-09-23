"""
Structural parser for EUR-Lex HTML.

Splits a Commission act into:
    title / citations / recitals / articles / annexes / concluding_formulas

EUR-Lex HTML quirks we have to handle:
- Recitals are laid out as a 2-column `<table>` (number cell + text cell),
  not as a list. We must detect this layout pattern and NOT treat it as a
  data table.
- Articles are flagged by `<p class="oj-ti-art">` headings, but some older
  acts use `<div class="eli-subdivision">` wrappers instead. We try both.
- The boundary "HAS ADOPTED THIS REGULATION" / "HAS ADOPTED THIS DECISION"
  separates preamble from enacting terms in many acts; we use it as a
  fallback when class-based detection misses.
- Annex starts at a heading whose text begins with "ANNEX". The annex
  title may be on the next line.

The parser is intentionally defensive: if a region cannot be detected
cleanly, it falls back to "unparsed" lumps rather than crashing. This is
fine for Phase-1 dry runs — the goal is to surface parsing failure modes
on real documents, not to claim 100% structural fidelity.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from bs4 import BeautifulSoup, Tag, NavigableString

from .oj_table import normalize_flattened_tables
from .schema import (
    Annex,
    AnnexTable,
    Article,
    FlattenedTableRecord,
    ParsedDocument,
    Recital,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_RECITAL_NUM_RE = re.compile(r"^\(\s*([0-9]+|[ivxIVX]+|[a-zA-Z])\s*\)\s*$")
_ARTICLE_HEADING_RE = re.compile(r"^Article\s+([0-9]+[a-zA-Z]?)\b", re.IGNORECASE)
_ANNEX_HEADING_RE = re.compile(r"^ANNEX(?:\s+([IVX0-9]+))?\b", re.IGNORECASE)
_ENACTING_MARKER_RE = re.compile(
    # Trailing [\s:]* eats the colon and whitespace after "...REGULATION:"
    # so that `match.end()` lands cleanly on Article 1 / first paragraph.
    r"HAS\s+ADOPTED\s+THIS\s+(?:REGULATION|DECISION|DIRECTIVE|RECOMMENDATION)[\s:]*",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _is_recital_layout_table(table: Tag) -> bool:
    """
    True if the <table> is the recital layout pattern: 2 columns, first
    column matches `(N)` style, no <th>.
    """
    if table.find("th"):
        return False
    rows = table.find_all("tr", recursive=False) or table.find_all("tr")
    if not rows:
        return False
    matched = 0
    for tr in rows[:5]:  # only sample the first few
        cells = tr.find_all("td")
        if len(cells) != 2:
            return False
        if _RECITAL_NUM_RE.match(_norm(cells[0].get_text())):
            matched += 1
    return matched >= 1


# ---------------------------------------------------------------------------
# Region splitting
# ---------------------------------------------------------------------------


def _extract_title(soup: BeautifulSoup) -> str:
    # Try CSS class first; fall back to <title> tag.
    el = soup.find(class_="oj-doc-ti")
    if el:
        return _norm(el.get_text(" "))
    el = soup.find("title")
    return _norm(el.get_text()) if el else ""


def _split_preamble_enacting_annex(soup: BeautifulSoup) -> tuple[List[Tag], List[Tag], List[Tag], List[Tag]]:
    """
    Walk the body's top-level descendants and bucket them into
    (preamble, enacting, annex, trailer).

    Boundaries:
        preamble  → up to "HAS ADOPTED THIS ..." marker
        enacting  → from marker up to first "ANNEX" heading
        annex     → from first ANNEX heading until end-of-document or "Done at"
        trailer   → "Done at Brussels, ..." and any signature lines
    """
    body = soup.body or soup
    nodes = [n for n in body.descendants if isinstance(n, Tag)]

    preamble: List[Tag] = []
    enacting: List[Tag] = []
    annex: List[Tag] = []
    trailer: List[Tag] = []

    bucket = preamble
    seen_enacting_marker = False
    seen_annex = False
    seen_trailer = False

    for tag in nodes:
        text = _norm(tag.get_text(" ")) if tag.name not in ("script", "style") else ""
        if not seen_annex and not seen_trailer and not seen_enacting_marker:
            if _ENACTING_MARKER_RE.search(text):
                seen_enacting_marker = True
                bucket = enacting
                continue
        if not seen_annex and _ANNEX_HEADING_RE.match(text):
            seen_annex = True
            bucket = annex
        if not seen_trailer and re.match(r"^Done\s+at\b", text, re.IGNORECASE):
            seen_trailer = True
            bucket = trailer
        bucket.append(tag)

    return preamble, enacting, annex, trailer


# ---------------------------------------------------------------------------
# Preamble (citations + recitals)
# ---------------------------------------------------------------------------


def _parse_citations(preamble_tags: List[Tag]) -> List[str]:
    """
    Citations = the "Having regard to ..." paragraphs that appear before
    the first recital number.
    """
    out: List[str] = []
    for tag in preamble_tags:
        if tag.name != "p":
            continue
        text = _norm(tag.get_text(" "))
        if text.lower().startswith("having regard to"):
            out.append(text)
        elif text.lower().startswith("whereas"):
            break
    return out


def _parse_recitals(preamble_tags: List[Tag]) -> tuple[List[Recital], str]:
    """
    Returns (recitals, flattened_preamble_text).

    Recitals come from the 2-column layout table whose first cell is "(N)".
    """
    recitals: List[Recital] = []
    flat = []  # (text_chunk,) — accumulated to compute char offsets

    for tag in preamble_tags:
        if tag.name == "table" and _is_recital_layout_table(tag):
            for tr in tag.find_all("tr"):
                cells = tr.find_all("td")
                if len(cells) != 2:
                    continue
                num_match = _RECITAL_NUM_RE.match(_norm(cells[0].get_text()))
                if not num_match:
                    continue
                number = num_match.group(0).strip()
                text = _norm(cells[1].get_text(" "))
                if not text:
                    continue
                start = sum(len(c) + 1 for c in flat)
                flat.append(text)
                end = start + len(text)
                recitals.append(
                    Recital(number=number, text=text, char_start=start, char_end=end)
                )
        # Skip non-table preamble nodes for the recital list (they're
        # citations or filler), but DON'T contribute to recital flat text.

    flat_text = "\n".join(flat)
    return recitals, flat_text


# ---------------------------------------------------------------------------
# Articles
# ---------------------------------------------------------------------------


def _parse_articles(enacting_tags: List[Tag]) -> tuple[List[Article], str]:
    """
    Walk enacting tags, group paragraphs under their preceding article
    heading. Heading detection prefers `oj-ti-art` class and falls back
    to "Article N" text matching.
    """
    articles: List[Article] = []
    current_number: Optional[str] = None
    current_title: Optional[str] = None
    current_paras: List[str] = []
    flat_chunks: list[str] = []
    flat_offsets: list[tuple[int, int]] = []  # per-article (start, end) into flat text

    def flush():
        if current_number is None:
            return
        joined = "\n".join(current_paras)
        start = sum(len(c) + 1 for c in flat_chunks)
        flat_chunks.append(joined)
        end = start + len(joined)
        flat_offsets.append((start, end))
        articles.append(
            Article(
                number=current_number,
                title=current_title,
                paragraphs=list(current_paras),
                char_start=start,
                char_end=end,
            )
        )

    for tag in enacting_tags:
        if tag.name not in ("p", "div"):
            continue
        text = _norm(tag.get_text(" "))
        if not text:
            continue
        classes = tag.get("class") or []

        is_heading = False
        if "oj-ti-art" in classes:
            is_heading = True
        else:
            m = _ARTICLE_HEADING_RE.match(text)
            if m and len(text) < 60:
                is_heading = True

        if is_heading:
            flush()
            m = _ARTICLE_HEADING_RE.match(text)
            current_number = m.group(1) if m else text
            current_title = None
            current_paras = []
            continue

        # The line right after the heading is often a sub-title (e.g. "Subject matter")
        if current_number is not None and current_title is None and len(text) < 80 and not text.endswith("."):
            current_title = text
            continue

        if current_number is not None:
            current_paras.append(text)

    flush()
    flat_text = "\n".join(flat_chunks)
    return articles, flat_text


# ---------------------------------------------------------------------------
# Annex (prose + data tables)
# ---------------------------------------------------------------------------


def _parse_annexes(annex_tags: List[Tag]) -> List[Annex]:
    """
    Group annex content by ANNEX heading. Within each annex, separate
    the prose nodes from the data tables.
    """
    if not annex_tags:
        return []

    annexes: List[Annex] = []
    current_id = ""
    current_title: Optional[str] = None
    current_prose: List[str] = []
    current_tables: List[AnnexTable] = []
    table_counter = 0

    def flush():
        nonlocal table_counter
        if not (current_id or current_title or current_prose or current_tables):
            return
        annexes.append(
            Annex(
                annex_id=current_id,
                title=current_title,
                prose="\n".join(current_prose),
                tables=list(current_tables),
            )
        )
        table_counter = 0

    for tag in annex_tags:
        if tag.name not in ("p", "div", "table"):
            continue
        text = _norm(tag.get_text(" "))

        if tag.name in ("p", "div") and _ANNEX_HEADING_RE.match(text):
            flush()
            m = _ANNEX_HEADING_RE.match(text)
            current_id = (m.group(1) or "") if m else ""
            current_title = text
            current_prose = []
            current_tables = []
            continue

        if tag.name == "table":
            if _is_recital_layout_table(tag):
                # Layout table accidentally inside annex — treat as prose.
                if text:
                    current_prose.append(text)
                continue
            headers, rows = _extract_table(tag)
            if not rows:
                continue
            current_tables.append(
                AnnexTable(
                    annex_id=current_id,
                    table_id=table_counter,
                    headers=headers,
                    rows=rows,
                )
            )
            table_counter += 1
            continue

        if text:
            current_prose.append(text)

    flush()
    return annexes


def _extract_table(table: Tag) -> tuple[List[str], List[List[str]]]:
    """
    Pull (headers, rows). Header detection rules:
      1. <thead><tr><th>... if present.
      2. First <tr> whose cells are all <th>.
      3. First <tr> if it visually looks like a header (short cells, no numbers).
      4. Empty headers list otherwise.
    """
    # Rule 1 + 2
    thead = table.find("thead")
    headers: List[str] = []
    rows: List[List[str]] = []

    all_trs = table.find_all("tr")
    if not all_trs:
        return [], []

    if thead:
        header_tr = thead.find("tr")
        if header_tr:
            headers = [_norm(c.get_text(" ")) for c in header_tr.find_all(["th", "td"])]
        body_trs = [tr for tr in all_trs if tr is not header_tr]
    else:
        first = all_trs[0]
        first_cells = first.find_all(["th", "td"])
        if all(c.name == "th" for c in first_cells) and first_cells:
            headers = [_norm(c.get_text(" ")) for c in first_cells]
            body_trs = all_trs[1:]
        else:
            # Rule 3: heuristic header detection — short cells, no digits.
            candidate = [_norm(c.get_text(" ")) for c in first_cells]
            looks_like_header = (
                candidate
                and all(len(c) <= 40 for c in candidate)
                and all(not re.search(r"\d{2,}", c) for c in candidate)
            )
            if looks_like_header:
                headers = candidate
                body_trs = all_trs[1:]
            else:
                headers = []
                body_trs = all_trs

    n_cols = len(headers) if headers else None
    for tr in body_trs:
        cells = [_norm(c.get_text(" ")) for c in tr.find_all(["th", "td"])]
        if not cells:
            continue
        if n_cols is None:
            n_cols = len(cells)
        # Pad/truncate to n_cols so downstream code can index safely.
        if len(cells) < n_cols:
            cells = cells + [""] * (n_cols - len(cells))
        elif len(cells) > n_cols:
            cells = cells[:n_cols]
        rows.append(cells)

    return headers, rows


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def parse_document(html: str, celex: str) -> ParsedDocument:
    """
    Parse one EUR-Lex HTML document into structured form.

    The CELEX number is passed in (rather than scraped) because the URL
    we used to fetch is the authoritative source.
    """
    soup = BeautifulSoup(html, "lxml")
    title = _extract_title(soup)

    preamble_tags, enacting_tags, annex_tags, trailer_tags = _split_preamble_enacting_annex(soup)

    citations = _parse_citations(preamble_tags)
    recitals, preamble_text = _parse_recitals(preamble_tags)
    articles, enacting_text = _parse_articles(enacting_tags)
    annexes = _parse_annexes(annex_tags)
    concluding = _norm(" ".join(_norm(t.get_text(" ")) for t in trailer_tags if t.name == "p"))

    return ParsedDocument(
        celex=celex,
        title=title,
        citations=citations,
        recitals=recitals,
        articles=articles,
        annexes=annexes,
        concluding_formulas=concluding,
        preamble_text=preamble_text,
        enacting_text=enacting_text,
    )


# ---------------------------------------------------------------------------
# Plain-text path (HuggingFace lex_glue input)
# ---------------------------------------------------------------------------

# Recital lines in plain-text dumps look like "(1) Whereas..." or "(1)\nWhereas".
_RECITAL_INLINE_RE = re.compile(
    r"^\s*\(\s*(?P<num>[0-9]+|[ivxIVX]+)\s*\)\s+(?P<text>.+?)(?=^\s*\(\s*[0-9ivxIVX]+\s*\)\s+|\Z)",
    re.MULTILINE | re.DOTALL,
)
# Standalone article heading on its own line: "Article 1", "Article 2a".
_ARTICLE_LINE_RE = re.compile(
    r"^\s*Article\s+(?P<num>[0-9]+[a-zA-Z]?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
# Annex heading at start of a line: "ANNEX I", "ANNEX", "ANNEX II".
_ANNEX_LINE_RE = re.compile(
    r"^\s*ANNEX(?:\s+(?P<id>[IVX0-9]+))?\b.*$",
    re.MULTILINE,
)


def _slice(text: str, start: int, end: Optional[int]) -> str:
    return text[start: end if end is not None else len(text)]


_TITLE_END_RE = re.compile(
    # Boundary between the title block and the preamble body. Matches the
    # institutional banner ("THE COUNCIL...", "THE EUROPEAN COMMISSION,",
    # "THE EUROPEAN PARLIAMENT AND THE COUNCIL...") or the first
    # "Having regard to" as a fallback.
    r"^\s*(?:THE\s+(?:COUNCIL|EUROPEAN|PARLIAMENT)|Having\s+regard\s+to)\b",
    re.MULTILINE | re.IGNORECASE,
)


def slice_articles(enacting_text: str) -> List[Article]:
    """Slice `enacting_text` between consecutive "Article N" headings.

    Exposed (rather than inlined in `parse_plain_text`) so a data migration
    that rewrites `enacting_text` in place can recompute article offsets with
    the *same* code the parser uses. Duplicating the logic in a script is how
    offsets and text drift apart.
    """
    article_headings = list(_ARTICLE_LINE_RE.finditer(enacting_text))
    articles: List[Article] = []
    for i, m in enumerate(article_headings):
        start = m.end()
        end = article_headings[i + 1].start() if i + 1 < len(article_headings) else len(enacting_text)
        body = enacting_text[start:end].strip()
        body_lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        # First short line after the heading is typically a sub-title (e.g. "Subject matter").
        title: Optional[str] = None
        if body_lines and len(body_lines[0]) < 80 and not body_lines[0].endswith("."):
            title = body_lines[0]
            body_lines = body_lines[1:]
        articles.append(
            Article(
                number=m.group("num"),
                title=title,
                paragraphs=body_lines,
                char_start=m.start(),
                char_end=end,
            )
        )
    return articles


def parse_plain_text(text: str, doc_id: str) -> ParsedDocument:
    """
    Parse a pre-cleaned plain-text EUR-Lex document (e.g. from
    HuggingFace `lex_glue` eurlex) into a `ParsedDocument`.

    Strategy: locate the same landmarks the HTML parser uses
    (HAS ADOPTED THIS / Article N / ANNEX) as regex matches on the
    flat text, slice between them, and populate `ParsedDocument`.

    Trade-offs vs the HTML parser:
        - No annex tables (no `<table>` structure in plain text).
        - Recital detection relies on "(N)" markers being preserved by
          the HF preprocessing. If the dataset normalised them away,
          `recitals` will be empty and the content stays in
          `preamble_text` as a single blob.
    """
    # 1) Title = all non-blank lines up to the first
    #    "THE COUNCIL/COMMISSION/PARLIAMENT..." or "Having regard to" line.
    title_end_match = _TITLE_END_RE.search(text)
    title_zone = text[: title_end_match.start()] if title_end_match else text[:200]
    title_lines = [ln.strip() for ln in title_zone.splitlines() if ln.strip()]
    first_line = " — ".join(title_lines) if title_lines else ""

    # 2) Find region boundaries.
    enacting_match = _ENACTING_MARKER_RE.search(text)
    annex_match = _ANNEX_LINE_RE.search(text)
    done_match = re.search(r"^\s*Done\s+at\b", text, re.MULTILINE | re.IGNORECASE)

    preamble_end = enacting_match.start() if enacting_match else (
        annex_match.start() if annex_match else (
            done_match.start() if done_match else len(text)
        )
    )
    enacting_start = enacting_match.end() if enacting_match else preamble_end
    enacting_end = annex_match.start() if annex_match else (
        done_match.start() if done_match else len(text)
    )
    annex_start = annex_match.start() if annex_match else None
    annex_end = done_match.start() if done_match else len(text)

    preamble_text = _slice(text, 0, preamble_end).strip()
    # _ENACTING_MARKER_RE already swallows trailing whitespace/colons,
    # so the slice starts cleanly on the first article.
    enacting_text = _slice(text, enacting_start, enacting_end).strip()
    annex_text = _slice(text, annex_start, annex_end).strip() if annex_start is not None else ""
    trailer_text = _slice(text, done_match.start() if done_match else len(text), len(text)).strip()

    # 2b) Rebuild tables that the OJ plain-text export flattened into `//`
    #     runs, so a segment carries readable rows instead of
    #     "1.2.3.4 // // // Order No // ...". This MUST happen before the
    #     recital and article slicing below: every downstream offset, and
    #     therefore every `source_quote` anchor, is measured against these two
    #     strings. See src/oj_table.py and gold decision log D-01.
    flattened: List[FlattenedTableRecord] = []
    preamble_text, _pre_tables = normalize_flattened_tables(preamble_text)
    enacting_text, _enact_tables = normalize_flattened_tables(enacting_text)
    for _field, _tables in (("preamble_text", _pre_tables),
                            ("enacting_text", _enact_tables)):
        for _ft in _tables:
            flattened.append(
                FlattenedTableRecord(
                    field=_field,
                    marker=_ft.marker,
                    n_cols=_ft.n_cols,
                    n_cells=len(_ft.cells),
                    n_rows=len(_ft.rows),
                    headers=list(_ft.headers),
                    ragged=_ft.ragged,
                    merged_fill=_ft.merged_fill,
                    raw=_ft.raw,
                )
            )

    # 3) Citations = "Having regard to ..." lines in the preamble.
    citations: List[str] = []
    for line in preamble_text.splitlines():
        s = line.strip()
        if s.lower().startswith("having regard to"):
            citations.append(s.rstrip(","))

    # 4) Recitals — match "(N) ..." blocks in the preamble.
    recitals: List[Recital] = []
    for m in _RECITAL_INLINE_RE.finditer(preamble_text):
        rtext = _norm(m.group("text"))
        if not rtext:
            continue
        recitals.append(
            Recital(
                number=f"({m.group('num')})",
                text=rtext,
                char_start=m.start(),
                char_end=m.end(),
            )
        )

    # 4b) Fallback for pre-1999 acts whose recitals are unnumbered run-on
    #     "Whereas ...; whereas ..." prose (no "(N)" markers). Without this the
    #     entire preamble stays an un-extracted blob (observed on train-000014,
    #     CELEX 31996R1629: 0 recitals, ~40% of the doc never sent to the LLM).
    #     Split on line-initial "Whereas" so enumerate_sections yields one
    #     section per recital. Offsets are into preamble_text, same convention
    #     as the numbered path above.
    if not recitals:
        wmarks = [w.start() for w in re.finditer(r"(?m)^\s*Whereas\b", preamble_text)]
        for wi, cs in enumerate(wmarks):
            ce = wmarks[wi + 1] if wi + 1 < len(wmarks) else len(preamble_text)
            rtext = _norm(preamble_text[cs:ce])
            if rtext:
                recitals.append(
                    Recital(number=f"({wi + 1})", text=rtext,
                            char_start=cs, char_end=ce)
                )

    # 5) Articles — slice enacting_text between consecutive "Article N" headings.
    articles = slice_articles(enacting_text)

    # 6) Annexes — collected as prose only (no table structure in plain text).
    annexes: List[Annex] = []
    if annex_text:
        # Find all ANNEX X headings within annex_text and split.
        annex_heads = list(_ANNEX_LINE_RE.finditer(annex_text))
        if not annex_heads:
            # Treat the whole block as a single unnumbered annex.
            annexes.append(Annex(annex_id="", title=None, prose=annex_text, tables=[]))
        else:
            for i, m in enumerate(annex_heads):
                head_line = m.group(0).strip()
                start = m.end()
                end = annex_heads[i + 1].start() if i + 1 < len(annex_heads) else len(annex_text)
                body = annex_text[start:end].strip()
                annexes.append(
                    Annex(
                        annex_id=m.group("id") or "",
                        title=head_line,
                        prose=body,
                        tables=[],
                    )
                )

    return ParsedDocument(
        celex=doc_id,
        title=first_line,
        citations=citations,
        recitals=recitals,
        articles=articles,
        annexes=annexes,
        concluding_formulas=trailer_text,
        preamble_text=preamble_text,
        enacting_text=enacting_text,
        flattened_tables=flattened,
    )
