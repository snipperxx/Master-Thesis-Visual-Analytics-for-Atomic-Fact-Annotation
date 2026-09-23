"""Build a self-contained HTML page for adjudicating the gold pre-annotation.

The page embeds every sampled segment and every pre-annotated fact. For each
fact the reviewer accepts, edits or rejects; facts the pre-annotation missed
can be added. The result downloads as data/gold/review_decisions.json, which
scripts/apply_gold_review.py folds back into data/gold/facts/.

Usage:
    python scripts/make_gold_review_page.py
    # open data/gold/review.html, adjudicate, download the decisions file
    python scripts/apply_gold_review.py data/gold/review_decisions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import gold_store as gs  # noqa: E402

GOLD = ROOT / "data" / "gold"


def prefill_entry(fact: dict) -> dict | None:
    """The page's starting state for one fact, or None if it has no decision yet.

    `src: "machine"` means *proposed, not confirmed*: the progress bar ignores
    such an entry and `apply_gold_review` records it as machine-adjudicated
    (BUG-2026-08-09-F). A decision whose `review.mode` is already `"human"` comes
    back without the marker, so a sweep survives a partial `--mode human` apply
    followed by a regenerate — see BUG-2026-08-16-A, where the same facts were
    asked for again and could be written back as `unreviewed`.
    """
    status = fact.get("status", "unreviewed")
    if status == "unreviewed":
        return None
    rev = fact.get("review") or {}
    entry = {"status": status, "edits": {},
             "note": fact.get("reviewer_note", ""), "by": rev.get("by", "")}
    if rev.get("mode") == "machine":
        entry["src"] = "machine"
    return entry


def build_payload(facts_root: Path = GOLD / "facts", *, epoch: int = 1) -> dict:
    # A page built from a non-production facts root must not share the browser
    # copy with the real sweep: same localStorage key = one root's decisions
    # leaking into the other's progress bar and download.
    try:
        rel = facts_root.resolve().relative_to((ROOT / "data" / "gold").resolve()).as_posix()
    except ValueError:                      # a root outside data/gold (tests, scratch)
        rel = facts_root.resolve().name

    sample = gs.load_sample()
    idx = {s["segment_id"]: s for s in sample["segments"]}
    iaa = set(sample["iaa_subsample"]["segment_ids"])
    segs = []
    for p in sorted(facts_root.glob("*.json")):
        payload = json.loads(p.read_text(encoding="utf-8"))
        for seg in payload["segments"]:
            m = idx[seg["segment_id"]]
            segs.append({
                "segment_id": seg["segment_id"],
                "doc_id": payload["doc_id"],
                "doc_title": payload["doc_title"],
                "section_path": seg["section_path"],
                "kind": m["kind"],
                "iaa": seg["segment_id"] in iaa,
                "text": gs.segment_text(payload["doc_id"], m["section_path"],
                                        m["char_start"], m["char_end"]),
                "facts": seg["facts"],
            })
    segs.sort(key=lambda s: (s["doc_id"], "0" if s["kind"] == "recital" else "1",
                             s["section_path"]))

    # Machine pre-review, if it has been generated. Advisory only: it orders the
    # queue and shows what to look at, and it never sets a status.
    #
    # It must be the one computed for THIS facts root. Until 2026-08-17 the page
    # always read `machine_review.json`, so the high-resolution sweep was badged
    # with flags computed on the coarse triples — 44 of them, most either fixed
    # by the re-decomposition or already settled in the decision log
    # (BUG-2026-08-17-A).
    review_path = GOLD / ("machine_review.json" if rel == "facts"
                          else f"machine_review_{rel.replace('/', '_')}.json")
    review, settled = {}, {}
    if review_path.exists():
        blob = json.loads(review_path.read_text(encoding="utf-8"))
        review = blob.get("facts", {})
        settled = blob.get("settled_not_checked", {})
    rank = {"error": 0, "warn": 1, "info": 2}
    for seg in segs:
        worst = 3
        for fact in seg["facts"]:
            rec = review.get(fact["gold_id"])
            fact["flags"] = rec["checks"] if rec else []
            fact["flag_severity"] = rec["severity"] if rec else ""
            worst = min(worst, rank.get(fact["flag_severity"], 3))
        seg["flag_severity"] = {0: "error", 1: "warn", 2: "info"}.get(worst, "")
        seg["n_flagged"] = sum(1 for f in seg["facts"]
                               if f["flag_severity"] in ("error", "warn"))

    # Whatever adjudication has already happened becomes the page's starting
    # state: each fact arrives with its status and reason, and the slot text
    # already shows any edit. The reviewer confirms or overrides instead of
    # deciding cold. See `prefill_entry` for what counts as confirmed.
    prefill: dict[str, dict] = {}
    for seg in segs:
        for f in seg["facts"]:
            entry = prefill_entry(f)
            if entry is not None:
                prefill[f["gold_id"]] = entry

    # `epoch` is how a sweep gets thrown away on purpose. The browser copy is
    # keyed by it, so bumping it starts every fact at `unreviewed` again without
    # touching anything on disk — which is what has to happen when the facts
    # underneath the page change enough that the decisions made against the old
    # ones say nothing about the new (the same argument as build_gold_hires').
    base = "gold-review-v1" if rel == "facts" else f"gold-review-v1::{rel}"
    return {"segments": segs,
            "decision_log": json.loads((GOLD / "decision_log.json").read_text(encoding="utf-8")),
            "n_facts": sum(len(s["facts"]) for s in segs),
            "n_triples": sum(len(f["triples"]) for s in segs for f in s["facts"]),
            "n_flagged": sum(s["n_flagged"] for s in segs),
            "has_review": bool(review),
            "settled_not_checked": settled,
            "facts_root": rel,
            "epoch": epoch,
            "store_key": base if epoch <= 1 else f"{base}#{epoch}",
            "prefill": prefill}


HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>Gold adjudication &mdash; EUR-Lex atomic facts</title>
<style>
:root{--bg:#faf9f7;--panel:#fff;--ink:#1c1b19;--mut:#6b6862;--line:#e3e0da;
      --ok:#1f7a4d;--edit:#9a6b00;--no:#a33;--acc:#2b5c9b}
*{box-sizing:border-box}
body{margin:0;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
     background:var(--bg);color:var(--ink)}
header{position:sticky;top:0;z-index:9;background:var(--panel);border-bottom:1px solid var(--line);
       padding:10px 18px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
h1{font-size:15px;margin:0;font-weight:650}
.bar{flex:1;min-width:180px;height:8px;background:var(--line);border-radius:4px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--ok);width:0;transition:width .2s}
button{font:inherit;padding:5px 11px;border:1px solid var(--line);background:#fff;
       border-radius:6px;cursor:pointer}
button:hover{border-color:var(--mut)}
button.primary{background:var(--acc);color:#fff;border-color:var(--acc)}
.wrap{display:grid;grid-template-columns:1fr 1fr;gap:18px;padding:18px;align-items:start}
@media(max-width:1000px){.wrap{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}
.src{position:sticky;top:64px;max-height:calc(100vh - 96px);overflow:auto}
.meta{font-size:12px;color:var(--mut);margin-bottom:8px}
.txt{white-space:pre-wrap;font-size:14.5px;line-height:1.7}
mark{background:#ffe9a8;padding:1px 0}
.fact{border:1px solid var(--line);border-radius:8px;padding:11px 13px;margin-bottom:11px}
.fact.cur{border-color:var(--acc);box-shadow:0 0 0 2px rgba(43,92,155,.13)}
.fact.accepted{border-left:4px solid var(--ok)}
.fact.edited{border-left:4px solid var(--edit)}
.fact.rejected{border-left:4px solid var(--no);opacity:.55}
.fid{font:12px ui-monospace,monospace;color:var(--mut)}
.nl{margin:5px 0 7px;font-weight:550}
.spo{display:grid;grid-template-columns:78px 1fr;gap:3px 9px;font-size:13.5px}
.spo b{color:var(--mut);font-weight:500}
.spo [contenteditable]{border-bottom:1px dashed transparent;outline:none;padding:1px 2px}
.spo [contenteditable]:hover{border-bottom-color:var(--line)}
.spo [contenteditable]:focus{border-bottom-color:var(--acc);background:#f4f8ff}
.spo .ph{color:#b3aea4}
/* One card can hold a CHAIN of triples decomposed from the same source span.
   Each gets its own block; a link marker shows where this triple's subject is
   the previous object, i.e. where the chain actually connects in the KG. */
.tri{border-top:1px dotted var(--line);margin-top:7px;padding-top:6px}
.tri:first-of-type{border-top:0;margin-top:0;padding-top:0}
.tri .th{font:11px ui-monospace,monospace;color:var(--mut);margin-bottom:3px}
.tri .th .lk{color:var(--ok)}
.tri .th .orphan{color:#a5342c}
.qual{display:grid;grid-template-columns:78px 1fr;gap:3px 9px;font-size:13.5px;
      border-top:1px dotted var(--line);margin-top:7px;padding-top:6px}
.qual b{color:var(--mut);font-weight:500}
.qual [contenteditable]{border-bottom:1px dashed transparent;outline:none;padding:1px 2px}
.qual [contenteditable]:hover{border-bottom-color:var(--line)}
.qual [contenteditable]:focus{border-bottom-color:var(--acc);background:#f4f8ff}
.minib{font:11px inherit;padding:2px 7px}
.q{font-size:12.5px;color:var(--mut);margin-top:6px;font-style:italic}
.acts{margin-top:9px;display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.tag{font-size:11px;padding:2px 7px;border-radius:99px;background:#f0eee9;color:var(--mut)}
.tag.err{background:#fdecec;color:#a5342c}
.tag.wrn{background:#fdf4e3;color:#8a6414}
.fact.f-error{box-shadow:inset 4px 0 0 #c0392b}
.fact.f-warn{box-shadow:inset 4px 0 0 #d79a25}
.flags{margin-top:8px;font-size:12px;line-height:1.5;border-top:1px dashed var(--line);padding-top:7px}
.flags .fl{display:block;color:var(--mut)}
.flags .fl b{font-weight:600}
.flags .fl.e b{color:#a5342c}
.flags .fl.w b{color:#8a6414}
.flags .hint{color:var(--mut);font-style:italic}
.tag.iaa{background:#ffe2e2;color:#9a2b2b}
.tag.bp{background:#e8eef8;color:#2b5c9b}
.note{width:100%;font:13px inherit;border:1px solid var(--line);border-radius:6px;
      padding:5px 7px;margin-top:6px}
nav{display:flex;gap:6px;align-items:center;font-size:13px;color:var(--mut)}
.hint{font-size:12px;color:var(--mut)}
</style>
<header>
  <h1 id="title">Gold adjudication</h1>
  <nav>
    <button onclick="go(-1)">&larr;</button>
    <span id="pos"></span>
    <button onclick="go(1)">&rarr;</button>
  </nav>
  <div class="bar"><i id="prog"></i></div>
  <span id="cnt" class="hint"></span>
  <button id="ordbtn" onclick="toggleOrder()"></button>
  <button onclick="addFact()">+ missing fact</button>
  <button onclick="resetProgress()">Reset progress</button>
  <button onclick="document.getElementById('imp').click()">Load decisions&hellip;</button>
  <input id="imp" type="file" accept="application/json,.json" style="display:none"
         onchange="loadFile(this)">
  <button class="primary" onclick="save()">Download decisions</button>
</header>
<div class="wrap">
  <div class="card src">
    <div class="meta" id="smeta"></div>
    <div class="txt" id="stext"></div>
  </div>
  <div id="facts"></div>
</div>
<p class="hint" style="padding:0 18px 30px">
  Keys: <b>a</b> accept &middot; <b>r</b> reject &middot; <b>e</b> mark edited &middot;
  <b>j</b>/<b>k</b> next/prev fact &middot; <b>&larr;</b>/<b>&rarr;</b> segment.
  Edit any slot by clicking it; editing marks the fact <i>edited</i> automatically.
  A fact can hold a <b>chain of triples</b> decomposed from its one quote &mdash; each
  block is one triple, and <b>&#8627;</b> says whether its subject repeats an earlier
  object (that repeat is what connects them in the KG). <b>+ triple</b> extends the
  chain. Emptying <i>condition</i>/<i>temporal</i> is not applied &mdash; an empty edit
  is skipped by <code>apply_gold_review</code>; reject the fact and re-add it instead.
  Progress autosaves in this browser; download the decisions file when done.
  A download can be read back with <b>Load decisions</b> &mdash; that is the only
  copy that survives a cleared browser profile or a move to another machine.
  <b>Reset progress</b> discards the browser copy and reopens every fact at its
  starting state.
</p>
<p class="hint" id="settled" style="padding:0 18px 30px"></p>
<script>
const DATA = __PAYLOAD__;
const KEY = DATA.store_key || "gold-review-v1";
let si = 0, fi = 0;

// Two orderings over the same segments. "flagged" puts the machine pre-review's
// error segments first, then warnings, then the rest — so the queue starts where
// something is probably wrong. "document" is the original corpus order, which is
// the right one for a final read-through. Nothing is hidden in either.
const DOC_ORDER = DATA.segments.slice();
const SEV_RANK = {error:0, warn:1, info:2, "":3};
const FLAG_ORDER = DATA.segments.slice().sort((a,b)=>
    (SEV_RANK[a.flag_severity]-SEV_RANK[b.flag_severity])
    || (b.n_flagged-a.n_flagged)
    || a.segment_id.localeCompare(b.segment_id));
let order = DATA.has_review ? "flagged" : "document";

// `keepCursor` is false on first load: there is no cursor to preserve yet, and
// trying to preserve one parks the queue on whatever position the document-order
// first segment happens to occupy in the flagged order — which defeats the
// triage entirely. It is true only for an explicit toggle, where staying on the
// segment being read is what the reviewer expects.
function applyOrder(keepCursor){
  const cur = keepCursor && DATA.segments[si] && DATA.segments[si].segment_id;
  DATA.segments = (order==="flagged") ? FLAG_ORDER.slice() : DOC_ORDER.slice();
  const back = cur ? DATA.segments.findIndex(s=>s.segment_id===cur) : -1;
  si = back >= 0 ? back : 0;
  const b = document.getElementById("ordbtn");
  if (b) {
    b.textContent = (order==="flagged")
      ? "order: flagged first (" + DATA.n_flagged + ")"
      : "order: document";
    b.style.display = DATA.has_review ? "" : "none";
  }
}
function toggleOrder(){ order = (order==="flagged") ? "document" : "flagged";
                        applyOrder(true); fi = 0; render(); }
// First load starts from the machine pass (if there was one); after that the
// browser copy wins, so a reload never discards work in progress.
let state = JSON.parse(localStorage.getItem(KEY) || "null")
            || JSON.parse(JSON.stringify(DATA.prefill || {}));

function seg(){ return DATA.segments[si]; }
function allFacts(){ return DATA.segments.flatMap(s=>s.facts); }
function st(id){ return state[id] || (state[id] = {status:"unreviewed", note:"", edits:{}}); }
function persist(){ localStorage.setItem(KEY, JSON.stringify(state)); paint(); }

function esc(s){ return (s||"").replace(/[&<>]/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }
function fold(s){ return (s||"").replace(/[‘’]/g,"'").replace(/[“”]/g,'"')
                    .replace(/[–—−]/g,"-").replace(/\s+/g," ").trim(); }

function render(){
  const s = seg();
  document.getElementById("pos").textContent = (si+1)+" / "+DATA.segments.length;
  document.getElementById("smeta").innerHTML =
     "<b>"+esc(s.segment_id)+"</b> &middot; "+esc(s.doc_title.slice(0,90))
     + (s.iaa ? ' <span class="tag iaa">IAA subsample &mdash; second annotator works this blind</span>' : "");
  const cur = s.facts[fi];
  let t = esc(s.text);
  if (cur){
    const q = fold(cur.source_quote);
    const ft = fold(s.text), pos = ft.indexOf(q);
    if (pos >= 0){
      // map folded offsets back by walking the raw string
      let raw=0, f=0, start=-1, end=-1;
      while (raw < s.text.length){
        const ch = s.text[raw];
        const isws = /\s/.test(ch);
        if (!(isws && f>0 && /\s/.test(fold(s.text[raw-1]||"")))){
          if (f === pos && start<0) start = raw;
          if (f === pos+q.length && end<0) end = raw;
          f++;
        }
        raw++;
      }
      if (start>=0){ if(end<0) end = s.text.length;
        t = esc(s.text.slice(0,start))+"<mark>"+esc(s.text.slice(start,end))+"</mark>"+esc(s.text.slice(end)); }
    }
  }
  document.getElementById("stext").innerHTML = t;

  document.getElementById("facts").innerHTML = s.facts.map((f,i)=>{
    const d = st(f.gold_id), e = d.edits;
    const trs = triplesOf(f);
    const v = k => esc(e[k] !== undefined ? e[k] : f[k]);
    const sev = f.flag_severity||"", act = (f.flags||[]).filter(c=>c.severity!=="info");
    const notes = (f.flags||[]).length - act.length;
    const fcls = sev==="error"?" f-error":(sev==="warn"?" f-warn":"");
    // An info check is context, not a complaint. Badging it "flag" is what made a
    // clean sweep look like 44 problems, so it gets its own neutral badge.
    const badge = act.length
      ? '<span class="tag '+(sev==="error"?"err":"wrn")+'">'+act.length+' flag'+(act.length>1?'s':'')+'</span>'
      : (notes ? '<span class="tag">'+notes+' note'+(notes>1?'s':'')+'</span>' : '');
    return `<div class="card fact ${i===fi?'cur':''} ${d.status}${fcls}" onclick="pick(${i})">
      <div class="fid">${esc(f.gold_id)} ${f.boilerplate?'<span class="tag bp">boilerplate</span>':''}
        ${badge}
        ${f.logic_op?'<span class="tag">'+f.logic_op+' &middot; '+esc(f.logic_group)+'</span>':''}</div>
      <div class="nl" contenteditable oninput="ed('${f.gold_id}','natural_language',this)">${v("natural_language")}</div>
      ${trs.map((tr,ti)=>`<div class="tri">
        ${trs.length>1?'<div class="th">t'+(ti+1)+"/"+trs.length+" "+chainMark(trs,ti)+'</div>':''}
        <div class="spo">
          <b>subject</b><span contenteditable oninput="edt('${f.gold_id}',${ti},'subject',this)">${esc(tr.subject)}</span>
          <b>predicate</b><span contenteditable oninput="edt('${f.gold_id}',${ti},'predicate',this)">${esc(tr.predicate)}</span>
          <b>object</b><span contenteditable oninput="edt('${f.gold_id}',${ti},'object',this)">${esc(tr.object)}</span>
        </div></div>`).join("")}
      <div class="qual">
        <b>condition</b><span contenteditable class="${v("condition")?"":"ph"}"
          oninput="ed('${f.gold_id}','condition',this)">${v("condition")||"&mdash;"}</span>
        <b>temporal</b><span contenteditable class="${v("temporal_context")?"":"ph"}"
          oninput="ed('${f.gold_id}','temporal_context',this)">${v("temporal_context")||"&mdash;"}</span>
      </div>
      <div class="q">&ldquo;${esc(f.source_quote)}&rdquo;</div>
      ${f.calibration_demo?'<div class="q" style="font-style:normal;color:var(--acc)">'
        +esc(f.calibration_demo)+'</div>':''}
      ${(f.flags&&f.flags.length)?'<div class="flags">'+f.flags.map(c=>
          '<span class="fl '+(c.severity==="error"?"e":c.severity==="warn"?"w":"")+'">'
          + '<b>'+esc(c.code)+'</b> &mdash; '+esc(c.detail)+'</span>').join("")
        + '<span class="fl hint">machine pre-review &mdash; advisory only, the decision is yours'
        + (act.length ? '' : '. Nothing here is a complaint: these are info checks.')
        + '</span></div>':''}
      <div class="acts">
        <button onclick="event.stopPropagation();mark('${f.gold_id}','accepted')">accept</button>
        <button onclick="event.stopPropagation();mark('${f.gold_id}','edited')">edited</button>
        <button onclick="event.stopPropagation();mark('${f.gold_id}','rejected')">reject</button>
        <button class="minib" onclick="event.stopPropagation();addTriple('${f.gold_id}')">+ triple</button>
        ${trs.length>1?'<button class="minib" onclick="event.stopPropagation();delTriple(\''+f.gold_id+'\','+(trs.length-1)+')">&minus; last triple</button>':''}
        <span class="tag">${d.status}</span>
        ${d.src==="machine" ? '<span class="tag wrn">machine-proposed &mdash; confirm or override</span>' : ''}
      </div>
      <input class="note" placeholder="reviewer note (why rejected / what was wrong)"
             value="${esc(d.note)}" onclick="event.stopPropagation()"
             oninput="st('${f.gold_id}').note=this.value;persist()">
    </div>`;
  }).join("") || '<div class="card">No pre-annotated facts in this segment.</div>';
  paint();
}
function paint(){
  const fs = allFacts();
  // Confirmed = a decision this reviewer made. A machine-proposed decision still
  // carrying src:"machine" is not confirmed, however definite it looks.
  const done = fs.filter(f=>{const d=st(f.gold_id);
                             return d.status!=="unreviewed" && d.src!=="machine";}).length;
  const pre = fs.filter(f=>st(f.gold_id).src==="machine").length;
  document.getElementById("prog").style.width = (100*done/fs.length)+"%";
  document.getElementById("cnt").textContent = done+" / "+fs.length+" confirmed"
    + (pre ? "  ("+pre+" machine-proposed, awaiting you)" : "");
}
function pick(i){ fi=i; render(); }
function go(d){ si=(si+d+DATA.segments.length)%DATA.segments.length; fi=0; render(); }
function mark(id, s){ const d=st(id); d.status=s; delete d.src; persist(); render(); }
function ed(id, k, el){ const d=st(id); let t = el.textContent.trim();
                        if (t === "—" || t === "-") t = "";   // the empty-slot placeholder
                        d.edits[k]=t;
                        if(d.status!=="rejected") d.status="edited";
                        delete d.src; persist(); }

// ---- triple chains ---------------------------------------------------------
// A fact holds N triples decomposed from ONE source span (schema.py `group_id`).
// Edits are written back as a whole `triples` array, which is the shape
// apply_gold_review.py already understands; the legacy flat subject/predicate/
// object keys are dropped on the first triple edit so the applier cannot see
// both and write them in an undefined order.
function factOf(id){ return allFacts().find(f=>f.gold_id===id); }
function triplesOf(f){
  const e = st(f.gold_id).edits;
  if (e.triples) return e.triples;
  const t0 = JSON.parse(JSON.stringify(f.triples));
  ["subject","predicate","object"].forEach(k=>{ if (e[k] !== undefined) t0[0][k] = e[k]; });
  return t0;
}
function writeTriples(id, arr){
  const d = st(id);
  d.edits.triples = arr;
  ["subject","predicate","object"].forEach(k=>delete d.edits[k]);
  if (d.status!=="rejected") d.status="edited";
  delete d.src; persist();
}
function edt(id, ti, k, el){
  const arr = triplesOf(factOf(id));
  arr[ti][k] = el.textContent.trim();
  writeTriples(id, arr);
}
function addTriple(id){
  const arr = triplesOf(factOf(id));
  arr.push({subject:"", predicate:"", object:""});
  writeTriples(id, arr); render();
}
function delTriple(id, ti){
  const arr = triplesOf(factOf(id));
  if (arr.length < 2) return;
  arr.splice(ti,1); writeTriples(id, arr); render();
}
// Where does this triple attach? Three cases, and they are not the same thing:
//   * its subject repeats an earlier OBJECT  -> a chain link, the nesting itself;
//   * its subject repeats an earlier SUBJECT -> a sibling, two things said about
//     one entity, which is fine but is not a chain;
//   * neither -> nothing connects it, so the KG gets a second unconnected node.
// The match is verbatim up to a leading article and case (`entity_norm` folds
// more later, but a near-miss here still splits the node), or the head of a
// longer earlier slot ("the arrangements" inside "access to the arrangements").
function chainMark(trs, ti){
  if (ti === 0) return "";
  const key = s => (s||"").toLowerCase().replace(/^(the|a|an)\s+/,"").trim();
  const hit = (subj, list) => list.some(e => e === subj || e.includes(subj));
  const subj = key(trs[ti].subject);
  if (hit(subj, trs.slice(0,ti).map(t=>key(t.object))))
    return '<span class="lk">&#8627; links on &ldquo;'+esc(trs[ti].subject)+'&rdquo;</span>';
  if (hit(subj, trs.slice(0,ti).map(t=>key(t.subject))))
    return 'same subject';
  return '<span class="orphan">&#8627; nothing connects this triple</span>';
}
// Throw the browser copy away and reopen from the page's own starting state.
// The page is regenerated with a bumped `epoch` when the facts underneath it
// change; this is the same thing on demand, for when the reviewer wants to
// restart a sweep he no longer trusts.
function resetProgress(){
  const fs = allFacts();
  const mine = fs.filter(f=>{const d=state[f.gold_id];
                             return d && d.status && d.status!=="unreviewed" && d.src!=="machine";}).length;
  if (!confirm("Discard the sweep in this browser?\n" + mine + " decision(s) you made "
      + "will be forgotten and every fact reopens at its starting state. Nothing on "
      + "disk changes, and a decisions file you downloaded can be read back with "
      + "Load decisions.")) return;
  localStorage.removeItem(KEY);
  state = JSON.parse(JSON.stringify(DATA.prefill || {}));
  si = 0; fi = 0; persist(); render();
}
function addFact(){
  const s = seg();
  const id = s.segment_id+"::added"+(1+s.facts.filter(f=>f.gold_id.includes("added")).length);
  s.facts.push({gold_id:id, natural_language:"", source_quote:"", condition:"",
                temporal_context:"", logic_group:"", logic_op:"", boilerplate:false,
                triples:[{subject:"",predicate:"",object:""}], added_by_reviewer:true});
  st(id).status="edited"; persist(); fi=s.facts.length-1; render();
}
// The download is the only copy of a sweep that leaves this browser — progress
// otherwise lives in localStorage, which a cleared profile, a different browser
// or a second machine does not carry. Reading a decisions file back in is what
// makes the download a real backup. `src` is preserved verbatim: an entry the
// machine proposed and nobody confirmed must not become confirmed by a
// round-trip through a file (BUG-2026-08-09-F).
function loadFile(inp){
  const file = inp.files && inp.files[0];
  inp.value = "";
  if (!file) return;
  const r = new FileReader();
  r.onload = () => {
    let p;
    try { p = JSON.parse(r.result); } catch(err){ alert("Not a JSON file."); return; }
    if (!p || typeof p.decisions !== "object" || !p.decisions){
      alert("No `decisions` object in that file — pick a review_decisions.json."); return; }
    const n = Object.keys(p.decisions).length;
    const conf = Object.values(p.decisions)
      .filter(d=>d && d.status && d.status!=="unreviewed" && d.src!=="machine").length;
    if (!confirm("Replace the progress in this browser with " + n + " decision(s) from "
        + file.name + "?\n" + conf + " of them are confirmed by you; the rest stay "
        + "machine-proposed.")) return;
    state = p.decisions;
    (p.added || []).forEach(a=>{
      const s = DATA.segments.find(x=>x.segment_id===a.segment_id);
      if (s && !s.facts.some(x=>x.gold_id===a.fact.gold_id)) s.facts.push(a.fact);
    });
    fi = 0; persist(); render();
  };
  r.readAsText(file);
}
function save(){
  const left = Object.values(state).filter(d=>d.src==="machine").length;
  // Accurate as of BUG-2026-08-09-F: unconfirmed entries are applied as
  // machine-adjudicated, per fact. Downloading a partial sweep is safe — it just
  // cannot make the gold human-adjudicated yet.
  if (left && !confirm(left+" decision(s) are still machine-proposed. They will be "
      + "recorded as machine-adjudicated, so the gold stays PARTIALLY "
      + "human-adjudicated until you finish. Download anyway?"))
    return;
  // `facts_root` says which annotation these decisions belong to. Without it a
  // sweep of the second annotator's facts could be applied to the gold by
  // pointing the wrong applier at it — the appliers refuse on a mismatch.
  const out = {reviewed_on:new Date().toISOString(), mode:"human",
    facts_root:DATA.facts_root, decisions:state,
    added:DATA.segments.flatMap(s=>s.facts.filter(f=>f.added_by_reviewer)
      .map(f=>({segment_id:s.segment_id, fact:f})))};
  const b = new Blob([JSON.stringify(out,null,2)],{type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(b); a.download = "review_decisions.json"; a.click();
}
addEventListener("keydown", e=>{
  if (/^(INPUT|TEXTAREA)$/.test(e.target.tagName) || e.target.isContentEditable) return;
  const s = seg(), f = s.facts[fi];
  if (e.key==="ArrowRight") go(1);
  else if (e.key==="ArrowLeft") go(-1);
  else if (e.key==="j"){ if(fi<s.facts.length-1){fi++;render();} else go(1); }
  else if (e.key==="k"){ if(fi>0){fi--;render();} else go(-1); }
  else if (f && "are".includes(e.key)){
    mark(f.gold_id, {a:"accepted", r:"rejected", e:"edited"}[e.key]);
    if(fi<s.facts.length-1){fi++;render();}
  }
});
// Which annotation is on screen. Two pages with the same layout — the gold and
// the second annotator's blind pass — must not be confusable at a glance.
if (DATA.facts_root && DATA.facts_root !== "facts"){
  const t = document.getElementById("title");
  t.textContent = (DATA.facts_root === "annotator_b")
    ? "Second annotation — adjudication" : "Adjudication — " + DATA.facts_root;
  t.title = "data/gold/" + DATA.facts_root;
  document.title = t.textContent;
}
// A check that is NOT run is as much a part of the review as one that is. Saying
// so on the page is what stops the next reader from wondering whether the
// boilerplate flags went missing by accident (they were retired by D-13).
(function(){
  const s = DATA.settled_not_checked || {}, keys = Object.keys(s);
  const el = document.getElementById("settled");
  if (!el || !keys.length) return;
  el.innerHTML = "<b>Checks retired by the decision log, deliberately not shown:</b><br>"
    + keys.sort().map(k=>"<code>"+esc(k)+"</code> &mdash; "+esc(s[k])).join("<br>");
})();
applyOrder(false);
render();
</script>
"""


#: Which generation of the sweep each facts root is on. Lives on disk because
#: the browser copy has to be invalidated across regenerations, and a number in
#: a source file would be silently wrong the moment someone forgot to commit it.
EPOCHS = GOLD / "review_epochs.json"


def _epochs() -> dict[str, int]:
    if EPOCHS.exists():
        return json.loads(EPOCHS.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--facts-root", type=Path, default=GOLD / "facts",
                    help="directory of gold fact files to review "
                         "(default data/gold/facts). A non-default root gets its "
                         "own browser storage key, so a preview cannot overwrite "
                         "the progress of the real sweep.")
    ap.add_argument("--out", type=Path, default=None,
                    help="output HTML (default data/gold/review.html, or "
                         "review_<root>.html for a non-default root)")
    ap.add_argument("--reset-progress", action="store_true",
                    help="throw away the sweep already made in the browser: bump "
                         "this root's storage epoch so every fact reopens as "
                         "unreviewed. Use it when the facts changed under the "
                         "page — a decision made against the old ones says "
                         "nothing about the new. Nothing on disk is touched.")
    args = ap.parse_args()

    try:
        rel_probe = args.facts_root.resolve().relative_to(
            (ROOT / "data" / "gold").resolve()).as_posix()
    except ValueError:
        rel_probe = args.facts_root.resolve().name
    epochs = _epochs()
    epoch = epochs.get(rel_probe, 1)
    if args.reset_progress:
        epoch += 1
        epochs[rel_probe] = epoch
        EPOCHS.write_text(json.dumps(epochs, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")

    payload = build_payload(args.facts_root, epoch=epoch)
    rel = payload["facts_root"]
    out = args.out or (GOLD / ("review.html" if rel == "facts"
                               else f"review_{rel.replace('/', '_')}.html"))
    out.write_text(HTML.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False)),
                   encoding="utf-8")
    print(f"wrote {out}  ({payload['n_facts']} facts / {payload['n_triples']} triples "
          f"over {len(payload['segments'])} segments, {out.stat().st_size//1024} KB)")
    print(f"  facts root: data/gold/{rel}  (storage key {payload['store_key']})")
    if args.reset_progress:
        print(f"  progress RESET — epoch {epoch}: any accept/edit/reject made in the "
              f"browser against epoch {epoch - 1} is ignored, every fact reopens "
              f"unreviewed")
    if payload["has_review"]:
        n_act = sum(1 for s in payload["segments"] for f in s["facts"]
                    if f["flag_severity"] in ("error", "warn"))
        n_info = sum(1 for s in payload["segments"] for f in s["facts"]
                     if f["flag_severity"] == "info")
        print(f"  machine pre-review loaded: {n_act} fact(s) flagged error/warn, "
              f"{n_info} carrying context only; queue opens in flagged-first order")
        if payload["settled_not_checked"]:
            print(f"  {len(payload['settled_not_checked'])} check(s) retired by the "
                  f"decision log and not shown: "
                  f"{', '.join(sorted(payload['settled_not_checked']))}")
    else:
        print(f"  no machine pre-review for this root — run "
              f"`python -m scripts.machine_review_gold --facts-root "
              f"data/gold/{rel} --write` to triage the queue")


if __name__ == "__main__":
    main()
