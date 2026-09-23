"""How much of each sampled segment did the gold actually annotate?

The gold anchors every fact to a verbatim `source_quote`. Laying those quotes back
over the segment text gives a character-level coverage figure and, more useful,
the **gaps**: stretches of source text no fact was written from. A gap is where a
missed proposition would hide.

Coverage is measured in whitespace-folded coordinates (the same folding
`gold_store.is_verbatim` uses), because the parsed text carries line breaks the
quotes do not. Every occurrence of a distinct quote counts as covered: a string
repeated inside one segment is the same content, and one fact may legitimately
speak for both.

The gap list is the point of the script — a coverage percentage alone cannot tell
a missed clause from a citation footnote. Classify each gap by reading it.

Usage:
    python scripts/gold_coverage_audit.py                       # summary + gaps
    python scripts/gold_coverage_audit.py --facts-root data/gold/facts_hires
    python scripts/gold_coverage_audit.py --min-words 6 --quiet  # bigger gaps only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import gold_store as gs  # noqa: E402

GOLD = ROOT / "data" / "gold"

# A gap made only of these is structure, not content: list bullets, recital
# numbers, OJ footnote markers, table pipes, page furniture.
_STRUCTURAL = re.compile(r"^[\s\-–—•|/()\[\]\.,;:'\"]*$")
_ONLY_MARKERS = re.compile(r"^[\s\-–—•|/]*(\(\d+\)|\d+|[A-Z]|'[A-Z]'|No)?[\s\-–—•|/.,;:]*$")

# Frames that carry no proposition of their own. Kept as an explicit, auditable
# list and *removed* before the content-word test, rather than padded into the
# stopword set — "account" is glue in "taking into account that" and content in
# "account must be taken of the situation", and a stopword cannot tell them apart.
_GLUE = [
    r"\bwhereas\b",
    r"\bit (?:appears|is stated|should be stated|has since been established) that\b",
    r"\btaking into account that\b",
    r"\bon the (?:one|other) hand\b",
    r"\bparticularly as regards\b",
    r"\bin respect of\b",
    r"\b(?:the )?data relating\b",
    r"\balthough\b",
    # heading of an enacting article, e.g. "Article 3 Collection of samples ..."
    r"^\W*Article\s+\d+\s+[A-Z][^.]*",
    # table-linearizer scaffolding from src/table_linearize.py
    r"In the table set out in this Article, row \d+:?",
    r"\b(?:Order No|Category \(unit\)|CN code)\s*=",
]
_GLUE_RE = [re.compile(p, re.I) for p in _GLUE]


def strip_glue(s: str) -> str:
    for rx in _GLUE_RE:
        s = rx.sub(" ", s)
    return s


def fold(s: str) -> str:
    return (s.replace("‘", "'").replace("’", "'")
             .replace("“", '"').replace("”", '"')
             .replace("–", "-").replace("—", "-").replace("−", "-"))


def fold_ws(s: str) -> str:
    return re.sub(r"\s+", " ", fold(s)).strip()


def covered_spans(text: str, quotes: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for q in {q for q in quotes if q.strip()}:
        qq = fold_ws(q)
        if not qq:
            continue
        start = 0
        while True:
            i = text.find(qq, start)
            if i < 0:
                break
            spans.append((i, i + len(qq)))
            start = i + 1
    return spans


def merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def gaps(text: str, spans: list[tuple[int, int]]) -> list[tuple[int, str]]:
    out, pos = [], 0
    for a, b in spans:
        if a > pos:
            out.append((pos, text[pos:a]))
        pos = max(pos, b)
    if pos < len(text):
        out.append((pos, text[pos:]))
    return out


# Function words carry no content, so their absence from a gap proves nothing.
_STOP = set("""a an the of on in to for by with from at as and or not that this these those
is are was were be been being shall should must may might can could will would have has had
its their it which who whom whose such other than into under within respectively also
whereas hereby therefore thus following about per each any all both same
article annex point paragraph subparagraph indent row column table set out
no nos ec eec eu""".split())


def content_tokens(s: str) -> list[str]:
    toks = re.findall(r"[a-z0-9][a-z0-9'/,\.%-]*", fold(s).lower())
    return [t.strip(".,'") for t in toks
            if t.strip(".,'") and t.strip(".,'") not in _STOP and len(t.strip(".,'")) > 1]


def classify(gap: str, facts: list[dict]) -> tuple[str, float]:
    """structural / echoed / unexplained.

    `echoed` = the gap's content words are all present in the facts of this
    segment — in a triple slot, a condition, a temporal_context or the
    natural_language restatement — so the proposition WAS annotated and only the
    `source_quote` was cut short of it. That distinction is the whole point: a
    truncated quote is a provenance defect, a missing proposition is a coverage
    defect, and a character-coverage number alone conflates them.
    """
    toks = content_tokens(strip_glue(gap))
    if not toks:
        return "structural", 1.0
    pool = set()
    for f in facts:
        for t in f["triples"]:
            pool |= set(content_tokens(" ".join((t["subject"], t["predicate"], t["object"]))))
        pool |= set(content_tokens(f.get("condition", "")))
        pool |= set(content_tokens(f.get("temporal_context", "")))
        pool |= set(content_tokens(f.get("natural_language", "") or ""))
    frac = sum(1 for t in toks if t in pool) / len(toks)
    return ("echoed" if frac >= 0.8 else "unexplained"), frac


def audit(facts_root: Path, min_words: int) -> dict:
    sample = gs.load_sample()
    idx = {s["segment_id"]: s for s in sample["segments"]}
    rows, unmatched = [], []
    for p in sorted(facts_root.glob("*.json")):
        payload = json.loads(p.read_text(encoding="utf-8"))
        for seg in payload["segments"]:
            m = idx[seg["segment_id"]]
            text = fold_ws(gs.segment_text(payload["doc_id"], m["section_path"],
                                           m["char_start"], m["char_end"]))
            quotes = [f["source_quote"] for f in seg["facts"]]
            for f in seg["facts"]:
                if fold_ws(f["source_quote"]) not in text:
                    unmatched.append(f["gold_id"])
            spans = merge(covered_spans(text, quotes))
            hit = sum(b - a for a, b in spans)
            seg_gaps = []
            for i, g in gaps(text, spans):
                if len(g.split()) < min_words:
                    continue
                if _STRUCTURAL.match(g) or _ONLY_MARKERS.match(g):
                    kind, frac = "structural", 1.0
                else:
                    kind, frac = classify(g, seg["facts"])
                seg_gaps.append({"text": g, "kind": kind, "echo": round(frac, 2)})
            rows.append({"segment_id": seg["segment_id"], "kind": m["kind"],
                         "n_facts": len(seg["facts"]), "chars": len(text),
                         "covered": hit, "gaps": seg_gaps})
    tot_c = sum(r["chars"] for r in rows)
    tot_h = sum(r["covered"] for r in rows)
    by_kind: dict[str, int] = {}
    kind_chars: dict[str, int] = {}
    for r in rows:
        for g in r["gaps"]:
            by_kind[g["kind"]] = by_kind.get(g["kind"], 0) + 1
            kind_chars[g["kind"]] = kind_chars.get(g["kind"], 0) + len(g["text"])
    return {"rows": rows, "chars": tot_c, "covered": tot_h,
            "frac": tot_h / tot_c if tot_c else 0.0,
            "n_gaps": sum(len(r["gaps"]) for r in rows),
            "gap_chars": sum(len(g["text"]) for r in rows for g in r["gaps"]),
            "by_kind": by_kind, "kind_chars": kind_chars,
            "unmatched": unmatched}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--facts-root", type=Path, default=GOLD / "facts_hires")
    ap.add_argument("--min-words", type=int, default=4,
                    help="ignore gaps shorter than this (default 4 words)")
    ap.add_argument("--quiet", action="store_true", help="summary and worst segments only")
    ap.add_argument("--only", choices=("structural", "echoed", "unexplained"),
                    help="list only gaps of this kind")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    args.facts_root = args.facts_root.resolve()
    res = audit(args.facts_root, args.min_words)
    print(f"{args.facts_root.relative_to(ROOT.resolve()).as_posix()}: "
          f"{len(res['rows'])} segments, {res['covered']}/{res['chars']} chars "
          f"= {100*res['frac']:.1f}% covered by a source quote")
    print(f"  {res['n_gaps']} gap(s) of >= {args.min_words} words, "
          f"{res['gap_chars']} chars ({100*res['gap_chars']/res['chars']:.1f}%)")
    for kind in ("structural", "echoed", "unexplained"):
        n = res["by_kind"].get(kind, 0)
        if n:
            print(f"    {kind:<12} {n:>3} gap(s), {res['kind_chars'][kind]:>5} chars "
                  f"({100*res['kind_chars'][kind]/res['chars']:.1f}%)")
    if res["unmatched"]:
        print(f"  !! {len(res['unmatched'])} quote(s) not found in their segment: "
              f"{res['unmatched'][:5]}")

    worst = sorted(res["rows"], key=lambda r: r["covered"] / max(r["chars"], 1))[:8]
    print("\nlowest-coverage segments")
    for r in worst:
        print(f"  {100*r['covered']/max(r['chars'],1):>5.1f}%  {r['n_facts']:>3} facts  "
              f"{r['segment_id']}")

    if not args.quiet:
        print("\ngaps")
        for r in res["rows"]:
            if not r["gaps"]:
                continue
            if args.only and all(g["kind"] != args.only for g in r["gaps"]):
                continue
            print(f"\n## {r['segment_id']}  ({r['n_facts']} facts, "
                  f"{100*r['covered']/max(r['chars'],1):.0f}% covered)")
            for g in r["gaps"]:
                if args.only and g["kind"] != args.only:
                    continue
                print(f"   [{g['kind']:<11} echo={g['echo']:.2f}] {g['text'].strip()[:300]}")

    if args.json_out:
        args.json_out.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
