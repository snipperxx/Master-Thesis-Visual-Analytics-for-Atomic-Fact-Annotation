# -*- coding: utf-8 -*-
"""Condensed annotation-guideline deck. Dark palette matched to the previous deck."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

BG    = RGBColor(0x1B, 0x2A, 0x41)
CARD  = RGBColor(0x23, 0x36, 0x4F)
BODY  = RGBColor(0xC9, 0xD6, 0xE3)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ACC   = RGBColor(0xC9, 0x77, 0x2E)
MUTE  = RGBColor(0x7C, 0x8D, 0xA3)

HEAD, TEXT, MONO = "Cambria", "Calibri", "Courier New"
W, H = Inches(13.333), Inches(7.5)
M = Inches(0.62)

prs = Presentation()
prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]


def slide():
    s = prs.slides.add_slide(BLANK)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    bg.fill.solid(); bg.fill.fore_color.rgb = BG; bg.line.fill.background()
    bg.shadow.inherit = False
    return s


def tb(s, x, y, w, h, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    t = s.shapes.add_textbox(x, y, w, h)
    tf = t.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    tf.paragraphs[0].alignment = align
    return tf


def put(tf, text, size, color=BODY, font=TEXT, bold=False, first=False,
        space_before=0, space_after=0, align=None, italic=False, spacing=None):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.space_before = Pt(space_before); p.space_after = Pt(space_after)
    if align is not None:
        p.alignment = align
    if spacing:
        p.line_spacing = spacing
    r = p.add_run(); r.text = text
    f = r.font
    f.size, f.name, f.bold, f.italic = Pt(size), font, bold, italic
    f.color.rgb = color
    return p


def card(s, x, y, w, h, fill=CARD):
    c = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    c.fill.solid(); c.fill.fore_color.rgb = fill
    c.line.fill.background()
    c.shadow.inherit = False
    c.adjustments[0] = 0.06
    return c


def header(s, kicker, title):
    tf = tb(s, M, Inches(0.5), W - 2 * M, Inches(0.3))
    put(tf, kicker.upper(), 11.5, ACC, TEXT, bold=True, first=True)
    tf2 = tb(s, M, Inches(0.85), W - 2 * M, Inches(0.6))
    put(tf2, title, 29, WHITE, HEAD, first=True)


def footnote(s, text, y=Inches(6.62)):
    tf = tb(s, M, y, W - 2 * M, Inches(0.5))
    put(tf, text, 11, MUTE, TEXT, first=True, spacing=1.25)


# ── 1 ── title ────────────────────────────────────────────────────────────
s = slide()
tf = tb(s, M, Inches(2.55), Inches(10.2), Inches(2.2))
put(tf, "Annotating atomic facts in EU legislation", 44, WHITE, HEAD, first=True)
put(tf, "Two independent passes over the same 59 segments — "
        "what the rules are, and where the two still disagree",
    19, BODY, TEXT, space_before=14, spacing=1.25)
put(tf, "Master's thesis  ·  EUR-Lex gold sample  ·  27 August 2026",
    12.5, MUTE, TEXT, space_before=22)
bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, M, Inches(2.28), Inches(1.5), Pt(3.5))
bar.fill.solid(); bar.fill.fore_color.rgb = ACC; bar.line.fill.background()
bar.shadow.inherit = False

# ── 2 ── the two passes: the numbers ──────────────────────────────────────
s = slide()
header(s, "the data", "Two passes, same 59 segments, different packaging")

rows = [
    ("facts",              "312",  "272"),
    ("triples",            "556",  "693"),
    ("triples per fact",   "1.78", "2.55"),
    ("median span",        "56 ch", "193 ch"),
]
x0, y0 = M, Inches(1.78)
colw = Inches(4.0)
gap = Inches(0.28)
for i, (label, a, b) in enumerate(rows):
    y = y0 + Emu(int((Inches(0.80) + Inches(0.12)) * i))
    c = card(s, x0, y, Inches(11.4), Inches(0.80))
    t = tb(s, x0 + Inches(0.32), y + Inches(0.22), Inches(3.4), Inches(0.4))
    put(t, label, 15, BODY, TEXT, first=True)
    for j, v in enumerate((a, b)):
        t = tb(s, x0 + Inches(4.0) + Emu(int((colw + gap) * j)), y + Inches(0.2),
               colw, Inches(0.45), align=PP_ALIGN.CENTER)
        put(t, v, 19, WHITE if j == 0 else ACC, MONO, bold=True, first=True)

t = tb(s, x0 + Inches(4.0), Inches(1.40), colw, Inches(0.3), align=PP_ALIGN.CENTER)
put(t, "PASS A", 11.5, MUTE, TEXT, bold=True, first=True)
t = tb(s, x0 + Inches(4.0) + colw + gap, Inches(1.40), colw, Inches(0.3), align=PP_ALIGN.CENTER)
put(t, "PASS B", 11.5, MUTE, TEXT, bold=True, first=True)

tf = tb(s, M, Inches(5.62), Inches(11.4), Inches(0.7))
put(tf, "Opposite packaging, not opposite readings.", 17, WHITE, HEAD, bold=True, first=True)
put(tf, "A cuts clause-sized spans into many shallow facts. "
        "B takes a whole sentence or row as one fact and externalises more relations inside it.",
    14, BODY, TEXT, space_before=5, spacing=1.2)
footnote(s, "Pass A: claude-opus-5 pre-annotation, re-decomposed under the rules.  "
            "Pass B: gemini-3-pro, blind — segment text and rules only, never A's output.")

# ── 3 ── agreement ────────────────────────────────────────────────────────
s = slide()
header(s, "agreement", "0.545 — and why the subsample said 0.769")

for i, (num, lab, sub) in enumerate([
    ("0.545", "all 59 segments", "the number to report"),
    ("0.769", "9-segment subsample", "overstates by 0.22"),
]):
    x = M + Emu(int((Inches(5.6) + Inches(0.32)) * i))
    card(s, x, Inches(1.95), Inches(5.6), Inches(2.0))
    tf = tb(s, x + Inches(0.45), Inches(2.28), Inches(4.7), Inches(1.5))
    put(tf, num, 46, WHITE if i == 0 else MUTE, MONO, bold=True, first=True)
    put(tf, lab, 16, BODY, TEXT, space_before=6)
    put(tf, sub, 12.5, ACC if i == 0 else MUTE, TEXT, space_before=3)

tf = tb(s, M, Inches(4.35), Inches(11.4), Inches(1.6))
put(tf, "The 9 were drawn uniformly over segment ids and landed on the short ones — "
        "32 words each against the sample's 63, and 7.7 % of the annotation mass. "
        "Jaccard, one-to-many: a one-to-one matcher structurally penalises whichever "
        "pass packages coarser, so 0.545 is a lower bound.",
    15, BODY, TEXT, first=True, spacing=1.3)

# ── 4 ── the rules ────────────────────────────────────────────────────────
s = slide()
header(s, "the guideline", "Eleven rules")

RULES = [
    "Predicate carries the relation only",
    "Relations inside a noun phrase come out",
    "One fact = one proposition and its chain",
    "Quantities are objects; scoping dates are context",
    "Never split an entity, measurement or code list",
    "A coined label must name exactly one thing",
    "Two propositions in one span: declare it",
    "A coordination expands into several facts",
    "One entity, one label — character for character",
    "“According to X” is a triple, not a reification",
    "A flattened table row: annotate cells, not rows",
]
colw, rowh = Inches(5.6), Inches(0.62)
for i, r in enumerate(RULES):
    col, row = i // 6, i % 6
    x = M + Emu(int((colw + Inches(0.32)) * col))
    y = Inches(1.8) + Emu(int((rowh + Inches(0.14)) * row))
    card(s, x, y, colw, rowh)
    t = tb(s, x + Inches(0.3), y + Inches(0.16), Inches(0.55), Inches(0.35))
    put(t, str(i + 1), 15, ACC, MONO, bold=True, first=True)
    t = tb(s, x + Inches(0.85), y + Inches(0.17), colw - Inches(1.1), Inches(0.4))
    put(t, r, 14.5, BODY, TEXT, first=True)

footnote(s, "Rules 1–10 were in force for both passes. Rule 11 was added on 27 August, "
            "after a flattened annex table arrived that no rule described.")

# ── 5 ── worked example, pass A ──────────────────────────────
H1 = RGBColor(0xE0, 0x8B, 0x3C)   # span 1
H2 = RGBColor(0x6F, 0xA8, 0xC7)   # span 2
H3 = RGBColor(0x8C, 0xB5, 0x7A)   # span 3

SEG = "train-000041::preamble.recitals[3]"
Q1 = ("The information must be disregarded if the goods concerned are not of "
      "sound and fair marketable quality")
Q2 = ("if the price quoted in the offer relates only to a small quantity that is "
      "not representative of the market")
Q3 = ("Offer prices which can be regarded as not representative of actual market "
      "trends must also be disregarded")

s = slide()
header(s, "worked example", "One segment, annotated")

card(s, M, Inches(1.72), Inches(11.4), Inches(1.30))
t = tb(s, M + Inches(0.34), Inches(1.88), Inches(10.7), Inches(1.0))
p = t.paragraphs[0]
p.line_spacing = 1.28
for txt, col in [(Q1, H1), (" or ", BODY), (Q2, H2), (". ", BODY),
                 (Q3, H3), (".", BODY)]:
    r = p.add_run(); r.text = txt
    r.font.size, r.font.name, r.font.color.rgb = Pt(14), HEAD, col
    r.font.bold = col is not BODY
t = tb(s, M + Inches(0.34), Inches(2.72), Inches(10.7), Inches(0.3))
put(t, SEG, 10, MUTE, MONO, first=True)

FACTS = [
    (H1, "f01", "if the goods concerned are not of sound and fair marketable quality",
     "OR  ·  t41r4-disregard",
     [("the information", "must be", "disregarded"),
      ("the goods concerned", "are not of", "sound and fair marketable quality")]),
    (H2, "f02", "if the price quoted in the offer relates only to a small quantity "
                "that is not representative of the market",
     "OR  ·  t41r4-disregard",
     [("the information", "must be", "disregarded"),
      ("the price quoted in the offer", "relates only to", "a small quantity"),
      ("a small quantity", "is not representative of", "the market")]),
    (H3, "f03", "", "",
     [("offer prices", "must also be", "disregarded"),
      ("offer prices", "can be regarded as not representative of",
       "actual market trends")]),
]
cw = Inches(3.613)
for i, (col, fid, cond, lg, triples) in enumerate(FACTS):
    x = M + Emu(int((cw + Inches(0.28)) * i))
    card(s, x, Inches(3.22), cw, Inches(3.14))
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x + Inches(0.28), Inches(3.44),
                             Inches(0.42), Pt(3))
    bar.fill.solid(); bar.fill.fore_color.rgb = col
    bar.line.fill.background(); bar.shadow.inherit = False
    t = tb(s, x + Inches(0.84), Inches(3.36), Inches(2.4), Inches(0.3))
    put(t, fid, 11, col, MONO, bold=True, first=True)

    tf = tb(s, x + Inches(0.28), Inches(3.72), cw - Inches(0.56), Inches(2.6))
    put(tf, "TRIPLES", 9, MUTE, TEXT, bold=True, first=True)
    for a, b, c in triples:
        pp = put(tf, "", 9, BODY, MONO, space_before=3, spacing=1.08)
        for seg_txt, seg_col in [(a, BODY), (" \u2192 ", MUTE), (b, WHITE),
                                 (" \u2192 ", MUTE), (c, BODY)]:
            r = pp.add_run(); r.text = seg_txt
            r.font.size, r.font.name = Pt(9), MONO
            r.font.color.rgb = seg_col
    if cond:
        put(tf, "CONDITION", 9, MUTE, TEXT, bold=True, space_before=9)
        put(tf, cond, 9.5, BODY, TEXT, space_before=3, spacing=1.1)
    if lg:
        put(tf, "LOGIC GROUP", 9, MUTE, TEXT, bold=True, space_before=9)
        put(tf, lg, 9.5, ACC, MONO, space_before=3)

footnote(s, "Colour marks the source_quote each fact is anchored to \u2014 verbatim, "
            "contiguous, non-overlapping. Rule 8 splits the coordination into f01 and "
            "f02; rule 2 pulls the relation out of \u201ca small quantity that is not "
            "representative of the market\u201d; rule 3 keeps each proposition and its "
            "chain in one fact.")

# \u2500\u2500 6 \u2500\u2500 same segment, other pass \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
s = slide()
header(s, "the same segment", "Where the two passes part company")

rows = [("facts", "3", "3"),
        ("coordination", "OR, two branches", "OR, two branches"),
        ("anchor for f01 / f02", "the clause \u2014 102 / 105 ch", "the whole recital \u2014 215 ch, both"),
        ("triples", "7", "4")]
y0 = Inches(1.78)
for i, (lab, a, b) in enumerate(rows):
    y = y0 + Emu(int(Inches(0.72) * i))
    card(s, M, y, Inches(11.4), Inches(0.62))
    t = tb(s, M + Inches(0.32), y + Inches(0.16), Inches(3.2), Inches(0.35))
    put(t, lab, 13, BODY, TEXT, first=True)
    for j, v in enumerate((a, b)):
        t = tb(s, M + Inches(3.8) + Emu(int(Inches(3.9) * j)), y + Inches(0.15),
               Inches(3.7), Inches(0.4), align=PP_ALIGN.CENTER)
        put(t, v, 13, WHITE if j == 0 else ACC, MONO, bold=True, first=True)
for j, lab in enumerate(("PASS A", "PASS B")):
    t = tb(s, M + Inches(3.8) + Emu(int(Inches(3.9) * j)), Inches(1.42),
           Inches(3.7), Inches(0.3), align=PP_ALIGN.CENTER)
    put(t, lab, 11, MUTE, TEXT, bold=True, first=True)

card(s, M, Inches(4.78), Inches(11.4), Inches(1.28))
tf = tb(s, M + Inches(0.34), Inches(4.96), Inches(10.7), Inches(1.0))
put(tf, "f01, side by side", 12, MUTE, TEXT, bold=True, first=True)
for lab, col, lines in [("A", WHITE, ["the information \u2192 must be \u2192 disregarded",
                                      "the goods concerned \u2192 are not of \u2192 sound and fair marketable quality"]),
                        ("B", ACC, ["the information \u2192 must be \u2192 disregarded"])]:
    pp = put(tf, "", 10.5, BODY, MONO, space_before=7, spacing=1.15)
    r = pp.add_run(); r.text = lab + "   "
    r.font.size, r.font.name, r.font.bold = Pt(10.5), MONO, True
    r.font.color.rgb = col
    r = pp.add_run(); r.text = "   ".join(lines)
    r.font.size, r.font.name = Pt(10.5), MONO
    r.font.color.rgb = BODY

tf = tb(s, M, Inches(6.28), Inches(11.4), Inches(0.8))
put(tf, "They agree on how many facts the segment holds and on the logic that joins "
        "them. They differ on how much text a fact is anchored to, and on whether a "
        "condition is also externalised as a triple \u2014 rules 2 and 3 settle neither.",
    13.5, BODY, TEXT, first=True, spacing=1.25)

# ── 7 ── where the rules under-determine ──────────────────────────────────
s = slide()
header(s, "open", "Three places the rules still don't decide")

items = [
    ("Span size",
     "Rule 3 fixes one fact per span but never says how large a span is. "
     "One text: 1 fact against 3."),
    ("Code lists",
     "Rule 5 says never split a code list; rule 11 sends a coordination inside a cell "
     "to rule 8, which expands it. A CN-code cell is both."),
    ("Labels",
     "32 labels fail rule 9 — 25 in single-modifier pairs, 7 unresolved demonstratives. "
     "Kept as a check, not an invariant."),
]
for i, (title, body) in enumerate(items):
    y = Inches(1.85) + Emu(int(Inches(1.28) * i))
    card(s, M, y, Inches(11.4), Inches(1.12))
    t = tb(s, M + Inches(0.34), y + Inches(0.15), Inches(0.5), Inches(0.4))
    put(t, str(i + 1), 17, ACC, MONO, bold=True, first=True)
    tf = tb(s, M + Inches(0.95), y + Inches(0.16), Inches(10.1), Inches(0.85))
    put(tf, title, 16, WHITE, HEAD, bold=True, first=True)
    put(tf, body, 13.5, BODY, TEXT, space_before=3, spacing=1.2)

tf = tb(s, M, Inches(5.85), Inches(11.4), Inches(0.8))
put(tf, "Both passes bend in the same places.", 16, WHITE, HEAD, bold=True, first=True)
put(tf, "Three broken chains out of 312, all declared — but each species its own fact in "
        "two seed tables and eight CN codes compressed into two ranges in a third. "
        "Same construction, two treatments, one pass.",
    13.5, BODY, TEXT, space_before=4, spacing=1.2)

# ── 8 ── the experiment ───────────────────────────────────────────────────
s = slide()
header(s, "what this feeds", "Closed-loop guideline refinement")

steps = [("Baseline", "extract with v1, align, label conflicts"),
         ("Control", "written before any diagnosis"),
         ("Diagnose", "dev set only — which rule under-determines what"),
         ("Revise", "v2, one change per diagnosed cluster"),
         ("Re-run", "v2 vs control on the held-out set")]
cw = Inches(2.13)
for i, (t1, t2) in enumerate(steps):
    x = M + Emu(int((cw + Inches(0.2)) * i))
    card(s, x, Inches(1.9), cw, Inches(1.85))
    tf = tb(s, x + Inches(0.24), Inches(2.12), cw - Inches(0.48), Inches(1.5))
    put(tf, str(i + 1), 14, ACC, MONO, bold=True, first=True)
    put(tf, t1, 15.5, WHITE, HEAD, bold=True, space_before=4)
    put(tf, t2, 12, BODY, TEXT, space_before=4, spacing=1.15)

for i, (q, txt) in enumerate([
    ("RQ1 — agreement", "Does the revision shift the conflict-type distribution "
                        "in the direction it was written to?"),
    ("RQ2 — correctness", "Does it beat the control on soft micro-F1 against the gold?"),
]):
    x = M + Emu(int((Inches(5.6) + Inches(0.32)) * i))
    card(s, x, Inches(4.1), Inches(5.6), Inches(1.25))
    tf = tb(s, x + Inches(0.34), Inches(4.32), Inches(4.95), Inches(0.9))
    put(tf, q, 15.5, ACC, HEAD, bold=True, first=True)
    put(tf, txt, 13, BODY, TEXT, space_before=4, spacing=1.2)

tf = tb(s, M, Inches(5.62), Inches(11.4), Inches(0.8))
put(tf, "The control is written before the diagnosis so a gain cannot be explained away as "
        "“a longer guideline helps”. A rule can make every annotator do the same wrong "
        "thing — RQ1 would reward that, only RQ2 catches it.",
    13.5, BODY, TEXT, first=True, spacing=1.25)

# ── 9 ── decisions ────────────────────────────────────────────────────────
s = slide()
header(s, "for discussion", "Four things I need settled")

asks = [
    ("Freeze the annotator set", "The cloud key has been dead since 8 Aug. "
     "Cut-off date, then freeze on the local models and carry it as a limitation?",
     "BLOCKS EVERYTHING"),
    ("One-to-many scoring", "gold_compare still matches one-to-one, which penalises the "
     "coarser pass. 0.545 is a lower bound until this is refactored.", "BLOCKS THE HEADLINE"),
    ("Rule 5 vs rule 11", "Decide what a code list is before the next pass, not during it. "
     "Then re-ask only the affected segments.", "BLOCKS THE NEXT PASS"),
    ("Adjudicating the gold", "312 facts are unreviewed. Full manual sweep, or stratified "
     "confirmation with the rest declared unreviewed?", "BLOCKS RQ2"),
]
for i, (t1, t2, tag) in enumerate(asks):
    y = Inches(1.8) + Emu(int(Inches(1.16) * i))
    card(s, M, y, Inches(11.4), Inches(1.0))
    t = tb(s, M + Inches(0.34), y + Inches(0.13), Inches(0.5), Inches(0.4))
    put(t, str(i + 1), 17, ACC, MONO, bold=True, first=True)
    tf = tb(s, M + Inches(0.95), y + Inches(0.14), Inches(7.7), Inches(0.8))
    put(tf, t1, 16, WHITE, HEAD, bold=True, first=True)
    put(tf, t2, 13, BODY, TEXT, space_before=3, spacing=1.15)
    t = tb(s, M + Inches(8.85), y + Inches(0.36), Inches(2.3), Inches(0.35),
           align=PP_ALIGN.RIGHT)
    put(t, tag, 10.5, ACC, TEXT, bold=True, first=True)

footnote(s, "Target submission 31 Oct 2026, hard deadline 30 Nov. "
            "Results, Case Studies, Discussion and Conclusion are still empty.")

out = r"C:\Users\Administrator\Desktop\Coding\Python\MasterThesis\doc\annotation_guideline_deck_v2.pptx"
prs.save(out)
print("saved", out, len(prs.slides.__iter__.__self__._sldIdLst))
