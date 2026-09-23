"""
External-text ingestion — wrap arbitrary plain text into the same
ParsedDocument JSON shape the rest of the pipeline expects.

Why this exists
---------------
Phase-1's `src.eurlex_parse.parse_plain_text` assumes EUR-Lex structure
(recitals, articles, preamble/enacting boundary). Most user-pasted text
won't have that structure. Rather than try to extend the legal parser
to handle arbitrary prose, we provide a tiny "generic parser" that:

  * preserves the existing schema (so extractor / Phase-2 / UI all
    keep working unchanged);
  * splits the input into "sections" the user can choose between —
    paragraph-level (one section per blank-line block) or
    sentence-level (one section per terminal punctuation);
  * **maps every section into `preamble.recitals[i]`** so the existing
    text-pane offsets in the UI still resolve correctly.

The pipeline's other half (`enacting.article_*`) stays empty for user
docs. That asymmetry is fine for prototyping — the extractor and Phase-2
treat the two halves symmetrically.

Conventions
-----------
- Output `celex` = the doc_id the caller passes in (we use
  `user_<short_hash>` from the Flask route).
- Output `preamble_text` is the raw input text (joined paragraphs).
  Char offsets inside the section objects point back into this string.
- `enacting_text` is empty.
- Annexes / citations / concluding_formulas are empty arrays / strings.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal


SplitMode = Literal["paragraph", "sentence", "blank_line"]


# ---------------------------------------------------------------------------
# Splitters
# ---------------------------------------------------------------------------


_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(“])")
_BLANK_RE = re.compile(r"\n\s*\n+")


def _split_paragraphs(text: str) -> list[tuple[int, int, str]]:
    """One chunk per blank-line-separated paragraph. Returns (start, end, body)."""
    out: list[tuple[int, int, str]] = []
    cursor = 0
    for m in _BLANK_RE.finditer(text):
        para = text[cursor:m.start()].strip()
        if para:
            # Find the actual content offsets (skip leading whitespace).
            start = cursor + (len(text[cursor:m.start()]) - len(text[cursor:m.start()].lstrip()))
            end = start + len(para)
            out.append((start, end, para))
        cursor = m.end()
    if cursor < len(text):
        tail = text[cursor:].strip()
        if tail:
            start = cursor + (len(text[cursor:]) - len(text[cursor:].lstrip()))
            end = start + len(tail)
            out.append((start, end, tail))
    if not out and text.strip():
        # No blank lines at all — the whole input is one paragraph.
        body = text.strip()
        start = text.find(body)
        out.append((start, start + len(body), body))
    return out


def _split_sentences(text: str) -> list[tuple[int, int, str]]:
    """One chunk per sentence (terminal . ! ? followed by a capital)."""
    if not text.strip():
        return []
    # Use finditer over splits to keep offsets.
    offsets: list[int] = [0]
    for m in _SENTENCE_END_RE.finditer(text):
        offsets.append(m.end())
    offsets.append(len(text))

    out: list[tuple[int, int, str]] = []
    for i in range(len(offsets) - 1):
        s, e = offsets[i], offsets[i + 1]
        body = text[s:e].strip()
        if not body:
            continue
        # tighten offsets to the stripped body
        actual_s = s + (len(text[s:e]) - len(text[s:e].lstrip()))
        actual_e = actual_s + len(body)
        out.append((actual_s, actual_e, body))
    return out


def split_into_sections(text: str, mode: SplitMode = "paragraph") -> list[tuple[int, int, str]]:
    """Public splitter. Returns (char_start, char_end, body) triples."""
    if mode in ("paragraph", "blank_line"):
        return _split_paragraphs(text)
    if mode == "sentence":
        return _split_sentences(text)
    raise ValueError(f"unknown split mode {mode!r}")


# ---------------------------------------------------------------------------
# Build ParsedDocument JSON
# ---------------------------------------------------------------------------


def _doc_id_for_text(title: str, text: str, prefix: str = "user") -> str:
    h = hashlib.sha1((title + "\0" + text).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{h}"


def build_parsed_doc(
    *,
    title: str,
    text: str,
    split_mode: SplitMode = "paragraph",
    doc_id: str | None = None,
) -> dict:
    """Return the JSON dict that `src.extractor.load_parsed_doc` can read.

    Sections become preamble.recitals[*]. enacting_text is empty so the
    extractor skips that half cleanly.
    """
    if doc_id is None:
        doc_id = _doc_id_for_text(title, text)

    sections = split_into_sections(text, mode=split_mode)
    recitals = []
    for i, (s, e, body) in enumerate(sections, start=1):
        recitals.append({
            "number": f"({i})",
            "text": body,
            "char_start": s,
            "char_end": e,
        })

    return {
        "document": {
            "celex": doc_id,
            "title": title or doc_id,
            "citations": [],
            "recitals": recitals,
            "articles": [],
            "annexes": [],
            "concluding_formulas": "",
            "preamble_text": text,
            "enacting_text": "",
        }
    }


def write_parsed_doc(
    parsed_root: Path, parsed: dict
) -> Path:
    """Write the JSON to data/parsed/<doc_id>.json. Returns the path."""
    doc_id = parsed["document"]["celex"]
    parsed_root.mkdir(parents=True, exist_ok=True)
    out_path = parsed_root / f"{doc_id}.json"
    out_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# CLI for ad-hoc testing
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--title", required=True)
    p.add_argument("--text", help="literal text. If omitted, read stdin.")
    p.add_argument("--split-mode", default="paragraph", choices=["paragraph", "sentence", "blank_line"])
    p.add_argument("--parsed-root", type=Path, default=Path("data/parsed"))
    args = p.parse_args()
    text = args.text if args.text is not None else sys.stdin.read()
    parsed = build_parsed_doc(title=args.title, text=text, split_mode=args.split_mode)
    out = write_parsed_doc(args.parsed_root, parsed)
    print(f"wrote {out}")
    print(f"doc_id={parsed['document']['celex']}  sections={len(parsed['document']['recitals'])}")
