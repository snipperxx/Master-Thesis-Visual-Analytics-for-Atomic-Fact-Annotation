"""Build `data/baseline_reader.html` — raw baseline output, no analysis layer.

This exists for exactly one step of the protocol: authoring the **no-feedback
control** g_v2'. `methodology_v2.tex` (Experimental protocol, step 2) says it
must be written "from the guideline text and a random sample of raw baseline
output only: no alignment, no conflict labels, no graph, no heatmap". The VA
interface cannot be used for that step by construction — everything it shows is
the analysis layer — and reading 9 781 facts out of JSON files by hand is not a
real option either. So: the same raw material the interface is built on, laid
out per segment with the three annotators side by side, and **nothing else**.
No alignment, no similarity, no conflict labels, no graph.

The page is deliberately dumb. It is not a second interface and must not grow
into one: the moment it shows which facts correspond to which, the control arm
stops being a control.

"A random sample" is not "one document". The default is every development
segment; `--sample N` draws a reproducible random subset (`--seed`, recorded in
the page) when a smaller reading pass is wanted. Either way the page states
what it drew, so the write-up can too.

    python scripts/make_baseline_reader.py
    python scripts/make_baseline_reader.py --sample 60 --seed 20260910
"""
from __future__ import annotations

import argparse
import html
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_phase2 import ANNOTATORS_OF_RECORD, _safe_dir  # noqa: E402

PARSED = ROOT / "data" / "parsed"
FACTS = ROOT / "data" / "facts"
DEV_LIST = ROOT / "data" / "doc_lists" / "eurlex_dev.json"
EVAL_LIST = ROOT / "data" / "doc_lists" / "eurlex_eval.json"
OUT = ROOT / "data" / "baseline_reader.html"

SHORT = {
    "cloud:qwen3.6:35b": "qwen3.6:35b",
    "cloud:hf.co/unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL": "gemma-4-31B",
    "cloud:hf.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF:UD-Q4_K_M":
        "Nemotron-30B",
}


def doc_ids(list_path: Path) -> list[str]:
    payload = json.loads(list_path.read_text(encoding="utf-8"))
    entries = payload["docs"] if isinstance(payload, dict) else payload
    return [d["doc_id"] if isinstance(d, dict) else str(d) for d in entries]


def sections_of(doc_id: str) -> list:
    """(section_path, text) in extraction order — mirrors enumerate_sections."""
    doc = json.loads((PARSED / f"{doc_id}.json").read_text(encoding="utf-8"))["document"]
    out = []
    pre = doc.get("preamble_text", "")
    for i, r in enumerate(doc.get("recitals", [])):
        t = pre[r["char_start"]:r["char_end"]]
        if t.strip():
            out.append((f"preamble.recitals[{i}]", t))
    ena = doc.get("enacting_text", "")
    for a in doc.get("articles", []):
        t = ena[a["char_start"]:a["char_end"]]
        if t.strip():
            out.append((f"enacting.article_{a['number']}", t))
    return out


def facts_of(doc_id: str) -> dict:
    """annotator -> section_path -> facts."""
    out = {}
    for model in ANNOTATORS_OF_RECORD:
        fp = FACTS / _safe_dir(model) / f"{doc_id}.json"
        if not fp.exists():
            continue
        by_sec = {}
        for f in json.loads(fp.read_text(encoding="utf-8")).get("facts", []):
            sec = (f.get("source_locator") or {}).get("section_path") or ""
            by_sec.setdefault(sec, []).append(f)
        out[model] = by_sec
    return out


def _annotation_blocks(facts: list) -> str:
    """Facts regrouped into the annotations the model actually emitted."""
    groups = {}
    for f in facts:
        groups.setdefault(f.get("group_id") or f["fact_id"], []).append(f)
    parts = []
    for _gid, triples in groups.items():
        nl = html.escape(triples[0].get("natural_language") or "")
        rows = "".join(
            '<tr><td>' + html.escape(t.get("subject") or "") + '</td>'
            '<td class="p">' + html.escape(t.get("predicate") or "") + '</td>'
            '<td>' + html.escape(t.get("object") or "") + '</td></tr>'
            for t in triples)
        extra = ""
        if triples[0].get("condition"):
            extra += '<div class="meta">condition: ' + \
                     html.escape(triples[0]["condition"]) + '</div>'
        if triples[0].get("temporal_context"):
            extra += '<div class="meta">temporal: ' + \
                     html.escape(triples[0]["temporal_context"]) + '</div>'
        # data-gid makes one annotation citable: the control console's change
        # log records (doc, section, annotator, group_id), which resolves back
        # into data/facts/<model>/<doc>.json. A DOM position would not.
        parts.append('<div class="ann" data-gid="' + html.escape(str(_gid))
                     + '" data-n="' + str(len(triples)) + '">'
                     '<div class="nl">' + nl + '</div>'
                     '<table class="tri">' + rows + '</table>' + extra + '</div>')
    return "".join(parts) or '<div class="none">- no annotation -</div>'


STYLE = """
 body { font: 14px/1.5 system-ui, sans-serif; margin: 0 auto; max-width: 1500px;
        padding: 16px; color: #1f2328; }
 header { position: sticky; top: 0; background: #fff; padding: 10px 0 8px;
          border-bottom: 2px solid #1f2328; z-index: 5; }
 .warn { background: #fdecea; border: 1px solid #f3b5ae; color: #7f1d1d;
         padding: 8px 12px; border-radius: 6px; margin: 8px 0; }
 .seg { border-top: 1px solid #d8dee4; padding: 14px 0; }
 h3 { font-size: 13px; color: #57606a; margin: 0 0 6px; font-weight: 600; }
 pre.src { background: #f6f8fa; padding: 10px; border-radius: 6px;
           white-space: pre-wrap; margin: 0 0 10px; font-size: 13px; }
 .cols { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
 .col { border: 1px solid #d8dee4; border-radius: 6px; padding: 8px; }
 .col h4 { margin: 0 0 6px; font-size: 12px; }
 .col h4 .n { color: #57606a; font-weight: 400; }
 .ann { border-top: 1px dashed #d8dee4; padding: 6px 0; }
 .ann:first-of-type { border-top: 0; }
 .nl { font-size: 12.5px; margin-bottom: 4px; }
 table.tri { border-collapse: collapse; width: 100%; font-size: 11.5px; }
 table.tri td { border: 1px solid #eaeef2; padding: 2px 4px; vertical-align: top; }
 table.tri td.p { color: #0550ae; }
 .meta { font-size: 11px; color: #57606a; }
 .none { color: #8c959f; font-size: 12px; }
 input { font: inherit; padding: 4px 8px; }
"""

SCRIPT = """
 document.getElementById("q").addEventListener("input", function (e) {
   var v = e.target.value.trim().toLowerCase();
   var segs = document.querySelectorAll(".seg");
   for (var i = 0; i < segs.length; i++)
     segs[i].style.display =
       (!v || segs[i].dataset.doc.toLowerCase().indexOf(v) >= 0) ? "" : "none";
 });
"""

WARNING = """
  <b>No analysis layer, on purpose.</b> This page shows what each annotator
  wrote and nothing about how the three relate - no alignment, no similarity,
  no conflict labels, no graph. It is the reading material the
  <b>no-feedback control g_v2'</b> is allowed to be written from
  (methodology, Experimental protocol, step 2). Once g_v2' is committed, use
  the VA interface instead; it shows strictly more.
"""


def build(sample, seed: int, list_path: Path) -> str:
    ids = doc_ids(list_path)
    units = []                                  # (doc_id, section_path, text)
    for d in ids:
        if not (PARSED / (d + ".json")).exists():
            continue
        for path, text in sections_of(d):
            units.append((d, path, text))
    total = len(units)
    drawn = "every segment of the %d-document development set" % len(ids)
    if sample and sample < total:
        rng = random.Random(seed)
        units = sorted(rng.sample(units, sample), key=lambda u: (u[0], u[1]))
        drawn = "a random sample of %d of %d segments (seed %d)" % (
            sample, total, seed)

    cache = {}
    blocks = []
    for doc_id, path, text in units:
        if doc_id not in cache:
            cache[doc_id] = facts_of(doc_id)
        cols = []
        for model in ANNOTATORS_OF_RECORD:
            fs = cache[doc_id].get(model, {}).get(path, [])
            cols.append('<div class="col" data-model="'
                        + html.escape(SHORT.get(model, model)) + '"><h4>'
                        + html.escape(SHORT.get(model, model))
                        + ' <span class="n">' + str(len(fs)) + ' triples</span></h4>'
                        + _annotation_blocks(fs) + '</div>')
        blocks.append('<section class="seg" data-doc="' + html.escape(doc_id)
                      + '" data-sec="' + html.escape(path) + '">'
                      '<h3>' + html.escape(doc_id) + ' &middot; '
                      + html.escape(path) + '</h3>'
                      '<pre class="src">' + html.escape(text.strip()) + '</pre>'
                      '<div class="cols">' + "".join(cols) + '</div></section>')

    built = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return ('<!doctype html>\n<meta charset="utf-8">'
            '<title>Baseline reader - raw g_v1 output</title>\n'
            '<style>' + STYLE + '</style>\n'
            '<header>\n <strong>Baseline reader - raw output under g_v1</strong>'
            ' &middot; ' + html.escape(drawn) + ' &middot; built ' + built +
            ' &nbsp; <input id="q" placeholder="filter by doc id..." size="18">'
            '\n <div class="warn">' + WARNING + '</div>\n</header>\n'
            + "".join(blocks) +
            '\n<script>' + SCRIPT + '</script>\n')


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=None,
                    help="draw this many segments at random (default: all)")
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--doc-list", type=Path, default=DEV_LIST,
                    help="development set only - the eval set is locked")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    # Discipline rule 1: the locked eval set is touched once per guideline
    # version, only to record results. Reading it here would be a fourth way in.
    if set(doc_ids(args.doc_list)) & set(doc_ids(EVAL_LIST)):
        raise SystemExit("refusing: that doc list contains locked evaluation "
                         "documents (discipline rule 1)")

    args.out.write_text(build(args.sample, args.seed, args.doc_list),
                        encoding="utf-8")
    print("wrote " + str(args.out))


if __name__ == "__main__":
    main()
