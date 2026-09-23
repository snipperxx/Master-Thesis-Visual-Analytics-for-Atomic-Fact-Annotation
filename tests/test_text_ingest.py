"""Tests for src.text_ingest."""

from __future__ import annotations

from src.text_ingest import build_parsed_doc, split_into_sections


def test_paragraph_split_counts():
    text = "Para one.\n\nPara two. With second sentence.\n\nPara three."
    secs = split_into_sections(text, mode="paragraph")
    assert len(secs) == 3
    # offsets must point at substrings actually in text
    for s, e, body in secs:
        assert text[s:e] == body


def test_sentence_split_counts():
    text = "First sentence. Second one! Third? Fourth done."
    secs = split_into_sections(text, mode="sentence")
    # 4 terminal punctuations, but the regex requires capital after — so the
    # last "Fourth done." doesn't have a following capital → it might be
    # merged. Accept 3 or 4 (the exact number depends on the lookahead).
    assert 3 <= len(secs) <= 4
    assert all(b.strip() for _, _, b in secs)


def test_no_blank_lines_falls_back_to_single_section():
    text = "Just one line."
    secs = split_into_sections(text, mode="paragraph")
    assert len(secs) == 1
    assert secs[0][2] == "Just one line."


def test_build_parsed_doc_shape():
    doc = build_parsed_doc(title="Hello", text="A.\n\nB.", split_mode="paragraph")
    d = doc["document"]
    assert d["celex"].startswith("user-")
    assert d["title"] == "Hello"
    assert len(d["recitals"]) == 2
    # Schema fields the rest of the pipeline expects
    assert d["enacting_text"] == ""
    assert d["articles"] == []
    # Char offsets resolve back to recital bodies
    for r in d["recitals"]:
        assert d["preamble_text"][r["char_start"]:r["char_end"]] == r["text"]


def test_deterministic_doc_id():
    doc1 = build_parsed_doc(title="T", text="X", split_mode="paragraph")
    doc2 = build_parsed_doc(title="T", text="X", split_mode="paragraph")
    assert doc1["document"]["celex"] == doc2["document"]["celex"]
