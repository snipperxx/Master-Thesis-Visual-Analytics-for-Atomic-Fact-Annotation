/* ============================================================
 * Atomic Fact VA — front-end controller (extended for Phase-3).
 *
 * Hot keys:
 *    j / k             cycle through visible facts (down / up)
 *    Esc               close any open modal
 *
 * Modals
 *    Pair-detail modal   — opens when a Cytoscape EDGE is clicked
 *                          (shows every aligned-pair behind that edge,
 *                          both annotators side-by-side with their
 *                          natural_language + source_quote + cosine +
 *                          Layer-1/2 reason).
 *    Compare modal       — opens via the "Compare v1↔v2" top-bar button;
 *                          fetches /api/distribution_shift and renders
 *                          an inline-SVG bar chart of label counts.
 *
 * State machine (single global `state` object). All mutators go through
 * renderAll() so the panes stay coherent. The facts table is the
 * authoritative current-visible source; both text spans and the graph
 * are filtered by intersecting with `visibleFacts()`.
 * ============================================================ */

(() => {
const state = {
  doc: null,
  annotators: new Set(),
  allAnnotators: [],
  conflict: "",
  search: "",
  merge: 0.78,
  layout: "fcose",
  selectedFactId: null,
  cy: null,
  textCache: { preamble: "", enacting: "" },
  factsById: new Map(),
  factSpans: new Map(),
  alignedPairs: [],          // for the pair-detail modal
  coverage: {},
  graph: null,
  jobs: new Map(),
  // Phase-6: visual encoding + KG layout knobs
  colorMode: "annotator",    // "fact" | "annotator" | "conflict" | "none"
  edgeLength: 160,            // px — drives ideal-edge-length for fcose/cose
  // Document selector (grouped by corpus)
  docs: [],                  // cached /api/docs list
  sourceFilter: "all",       // "all" | "webnlg" | "eurlex" | "user"
};

// ============================================================
// Color helpers (Phase-6)
// ============================================================
// Deterministic hash so the same fact_id always paints the same hue.
function _hash32(s) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = (h * 16777619) >>> 0;
  }
  return h;
}
function colorForFact(factId) {
  // Pastel, readable on white. Hue spread is uniform; sat/lum keep contrast.
  const hue = _hash32(factId || "") % 360;
  return `hsl(${hue}deg 70% 80%)`;
}
// Find the first case-insensitive index of `needle` in `hay` at or after pos.
// Returns -1 when no match. Tolerates extra whitespace by collapsing both.
function _findCI(hay, needle, pos = 0) {
  if (!needle) return -1;
  const lh = hay.toLowerCase();
  const ln = needle.toLowerCase().trim();
  if (!ln) return -1;
  const i = lh.indexOf(ln, pos);
  return i >= 0 ? i : -1;
}
// Tokenize for fuzzy matching: words/numbers incl. "2,0" "0,5" "%" forms.
function _spoTokens(s) {
  return [...String(s || "").toLowerCase().matchAll(/\d+(?:[.,]\d+)?|%|[\p{L}]+/gu)];
}
// Fuzzy locate `needle` inside `hay`: best token window sharing >= 55% of the
// needle's tokens. Decontextualized facts paraphrase the source ("the
// Commission services" vs "Commission services'"), so exact substring search
// alone misses most objects; token overlap recovers them.
function _fuzzyFind(hay, needle) {
  const nT = _spoTokens(needle).map(m => m[0]);
  if (!nT.length) return null;
  const hT = _spoTokens(hay);
  if (!hT.length) return null;
  const need = new Map();
  for (const t of nT) need.set(t, (need.get(t) || 0) + 1);
  let best = null;
  const win = Math.min(nT.length + 2, hT.length);
  for (let a = 0; a < hT.length; a++) {
    if (!need.has(hT[a][0])) continue;          // window must start on a hit
    const avail = new Map(need);
    let hits = 0;
    for (let b = a; b < Math.min(a + win, hT.length); b++) {
      const t = hT[b][0];
      if ((avail.get(t) || 0) > 0) { avail.set(t, avail.get(t) - 1); hits++; }
      const overlap = hits / nT.length;
      if (overlap >= 0.55 && (!best || overlap > best.overlap ||
          (overlap === best.overlap && (b - a) < (best.span || 1e9)))) {
        best = { overlap, span: b - a,
                 start: hT[a].index, end: hT[b].index + t.length };
      }
    }
  }
  return best;
}
// Build [{start,end,role}] spans for S/P/O substrings inside a quote string.
// Pass 1: exact case-insensitive, left-to-right (keeps natural S→P→O order).
// Pass 2: fuzzy token-window for roles exact search missed.
// Finally sort + clip overlaps so the DOM slicing stays consistent.
function spoBands(quote, subject, predicate, object) {
  const out = [];
  const missed = [];
  let pos = 0;
  for (const [role, text] of [["subject", subject], ["predicate", predicate], ["object", object]]) {
    const i = _findCI(quote, text || "", pos);
    if (i < 0) { missed.push([role, text]); continue; }
    out.push({ start: i, end: i + (text || "").length, role });
    pos = i + (text || "").length;
  }
  for (const [role, text] of missed) {
    const f = _fuzzyFind(quote, text || "");
    if (f) out.push({ start: f.start, end: f.end, role, fuzzy: true });
  }
  out.sort((a, b) => a.start - b.start || a.end - b.end);
  // Clip overlaps (fuzzy windows may intersect exact bands).
  const clipped = [];
  let cursor = 0;
  for (const b of out) {
    const start = Math.max(b.start, cursor);
    if (start >= b.end) continue;
    clipped.push({ ...b, start });
    cursor = b.end;
  }
  return clipped;
}

const $ = (sel) => document.querySelector(sel);

// ============================================================
// Bootstrap
// ============================================================

// The top document selector is grouped by corpus. WebNLG is listed first so
// it is easy to find, and a separate #source-filter narrows the list to a
// single corpus (all / WebNLG / EUR-Lex / Uploaded).
const SOURCE_GROUPS = [
  ["webnlg", "WebNLG"],
  ["redocred", "Re-DocRED"],
  ["eurlex", "EUR-Lex"],
  ["user", "Uploaded"],
];
function _docSource(d) {
  if (d.source) return d.source;
  if (d.is_user_doc || (d.doc_id || "").startsWith("user-")) return "user";
  if ((d.doc_id || "").startsWith("webnlg-")) return "webnlg";
  return "eurlex";
}
function _docOptionLabel(d) {
  const src = _docSource(d);
  const icon = src === "webnlg" ? "🔗 " : (src === "redocred" ? "📚 " : (src === "user" ? "📝 " : ""));
  // A locked eval document is empty on purpose, and the reason has to be on the
  // option itself: the dropdown is where the choice is made, and an unmarked
  // empty document reads as a broken app (or as an invitation to extract it).
  const lock = d.split === "eval" ? "🔒 " : "";
  const tail = d.split === "eval" ? "  [eval — locked]"
             : (d.has_conflicts ? "" : "  [no annotations yet]");
  return `${lock}${icon}${d.doc_id} — ${truncate(d.title || "", 70)}${tail}`;
}
/** The document the app should open on: the first one that has something to
 *  show. Falling back to the first listed put the app on `train-000000`, which
 *  is in the locked eval set — so every fresh open looked empty. */
function _firstUsefulDoc(shown) {
  const by = new Map((state.docs || []).map((d) => [d.doc_id, d]));
  return shown.find((id) => by.get(id)?.has_conflicts)
      || shown.find((id) => by.get(id)?.split !== "eval")
      || shown[0];
}
// Rebuild #doc-select from state.docs as <optgroup>s, filtered by the current
// source filter. Returns the doc_ids shown, in display order.
function populateDocSelect({ select = null } = {}) {
  const sel = $("#doc-select");
  if (!sel) return [];
  const filter = state.sourceFilter || "all";
  const prev = select || sel.value;
  sel.innerHTML = "";
  const shown = [];
  for (const [key, label] of SOURCE_GROUPS) {
    if (filter !== "all" && filter !== key) continue;
    const group = (state.docs || []).filter((d) => _docSource(d) === key);
    if (!group.length) continue;
    const og = document.createElement("optgroup");
    og.label = `${label} (${group.length})`;
    for (const d of group) {
      const opt = document.createElement("option");
      opt.value = d.doc_id;
      opt.textContent = _docOptionLabel(d);
      og.appendChild(opt);
      shown.push(d.doc_id);
    }
    sel.appendChild(og);
  }
  if (prev && shown.includes(prev)) sel.value = prev;
  else if (shown.length) sel.value = _firstUsefulDoc(shown);
  return shown;
}

async function boot() {
  state.docs = (await fetch("/api/docs").then((r) => r.json())).docs;
  const sel = $("#doc-select");
  populateDocSelect();
  sel.addEventListener("change", () => switchDoc(sel.value));

  const srcSel = $("#source-filter");
  if (srcSel) {
    srcSel.value = state.sourceFilter || "all";
    srcSel.addEventListener("change", (e) => {
      state.sourceFilter = e.target.value;
      const shown = populateDocSelect();
      if (shown.length) switchDoc(sel.value);
    });
  }

  $("#cy-segment-chip")?.addEventListener("click", () => setSegmentFilter(null));

  $("#conflict-filter").addEventListener("change", (e) => {
    state.conflict = e.target.value;
    refreshText(); refreshFacts(); refreshGraphHighlights();
  });

  $("#search-box").addEventListener("input", (e) => {
    state.search = e.target.value.toLowerCase();
    refreshText(); refreshFacts();
  });

  $("#layout-select").addEventListener("change", (e) => {
    state.layout = e.target.value;
    if (state.cy) state.cy.layout(_buildLayoutOpts(state.layout, state.cy.nodes().length)).run();
  });

  // Color-by selector (paints text marks per fact / annotator / conflict / off)
  const colorSel = $("#color-mode-select");
  if (colorSel) {
    colorSel.value = state.colorMode;
    document.body.dataset.colorMode = state.colorMode;
    colorSel.addEventListener("change", (e) => {
      state.colorMode = e.target.value;
      document.body.dataset.colorMode = state.colorMode;
      refreshText();
    });
  } else {
    document.body.dataset.colorMode = state.colorMode;
  }

  // Edge-length slider in #graph-settings panel.
  const edgeLenSlider = $("#cy-edge-length");
  const edgeLenVal = $("#cy-edge-length-v");
  if (edgeLenSlider) {
    let lenTimer = null;
    edgeLenSlider.addEventListener("input", () => {
      state.edgeLength = parseInt(edgeLenSlider.value, 10);
      if (edgeLenVal) edgeLenVal.textContent = String(state.edgeLength);
      clearTimeout(lenTimer);
      lenTimer = setTimeout(() => {
        if (state.cy) state.cy.layout(_buildLayoutOpts(state.layout, state.cy.nodes().length)).run();
      }, 250);
    });
  }

  const slider = $("#merge-threshold");
  const sliderVal = $("#merge-threshold-value");
  let sliderTimer = null;
  slider.addEventListener("input", () => {
    state.merge = parseFloat(slider.value);
    sliderVal.textContent = state.merge.toFixed(2);
    clearTimeout(sliderTimer);
    sliderTimer = setTimeout(refreshGraph, 250);
  });

  $("#compare-btn").addEventListener("click", openCompareModal);
  $("#cy-hide-islands")?.addEventListener("change", refreshGraphHighlights);
  $("#cy-core-entities")?.addEventListener("change", refreshGraph);
  $("#cy-reify")?.addEventListener("change", refreshGraph);
  $("#modal-close").addEventListener("click", closeAllModals);
  $("#compare-close").addEventListener("click", closeAllModals);
  $("#modal-overlay").addEventListener("click", (e) => { if (e.target.id === "modal-overlay") closeAllModals(); });
  $("#compare-overlay").addEventListener("click", (e) => { if (e.target.id === "compare-overlay") closeAllModals(); });
  $("#gold-compare-btn")?.addEventListener("click", openGoldCompare);
  $("#gold-close")?.addEventListener("click", closeAllModals);
  $("#gold-overlay")?.addEventListener("click", (e) => { if (e.target.id === "gold-overlay") closeAllModals(); });

  $("#guideline-save")?.addEventListener("click", saveGuideline);
  $("#export-json").addEventListener("click", () => exportFacts("json"));
  $("#export-csv").addEventListener("click", () => exportFacts("csv"));

  document.addEventListener("keydown", onKeyDown);

  if (state.docs.length) switchDoc(sel.value);
}

/** The one place the KG URL is built.
 *
 * `switchDoc` used to assemble its own and left `reify` off it, so every
 * document OPENED on a flat graph while the "Reify (statement nodes)" box said
 * otherwise — and the conflict-detection modal (the L1/L2 rationale) only opens
 * for a statement node or a double-tapped edge, so the arbitrator's reasons were
 * unreachable until you toggled the checkbox, which called `refreshGraph` and
 * quietly fixed the URL. Two builders for one resource is the bug; there is now
 * one (BUG-2026-09-20-B).
 */
function graphUrl(docId) {
  const core = ($("#cy-core-entities")?.checked ?? true) ? 1 : 0;
  const reify = ($("#cy-reify")?.checked ?? true) ? 1 : 0;
  return `/api/graph/${encodeURIComponent(docId)}`
       + `?merge_threshold=${state.merge}&core=${core}&reify=${reify}`;
}

async function switchDoc(docId) {
  state.doc = docId;
  _updateGoldCompareBtn(docId);
  const [doc, factsResp, graph, coverage, pairs] = await Promise.all([
    fetch(`/api/doc/${docId}`).then(r => r.json()),
    fetch(`/api/facts/${docId}`).then(r => r.json()),
    fetch(graphUrl(docId)).then(r => r.json()),
    fetch(`/api/coverage/${docId}`).then(r => r.json()).catch(() => ({})),
    fetch(`/api/pairs/${docId}`).then(r => r.json()).catch(() => []),
  ]);

  state.textCache.preamble = doc.preamble_text;
  state.textCache.enacting = doc.enacting_text;
  state.docParsed = doc;            // recital / article offsets, for the segment view
  _segLists = null;                 // pick up a guideline version added meanwhile
  $("#doc-title").textContent = doc.title || docId;

  const annotators = new Set();
  for (const f of factsResp.facts) annotators.add(f._annotator);
  state.allAnnotators = [...annotators].sort();
  state.annotators = new Set(state.allAnnotators);
  renderChips();

  state.factsById = new Map(factsResp.facts.map(f => [f.fact_id, f]));
  state.alignedPairs = pairs;
  state.coverage = coverage;
  state.graph = graph;

  renderCoverageRow();
  renderText();
  renderFacts();
  renderGraph(graph);
}

// ============================================================
// Coverage row
// ============================================================

function renderCoverageRow() {
  const row = $("#coverage-row");
  row.innerHTML = "";
  for (const [ann, m] of Object.entries(state.coverage)) {
    const chip = document.createElement("span");
    chip.className = "cov-chip";
    const charPct = Math.round((m.char_coverage_frac || 0) * 100);
    const secPct  = Math.round((m.sections_hit_frac || 0) * 100);
    chip.innerHTML = `<span class="cov-ann">${escapeHtml(ann)}</span>
      sec ${secPct}% <span class="bar"><span style="width:${secPct}%"></span></span>
      ch ${charPct}% <span class="bar"><span style="width:${charPct}%"></span></span>
      ${m.n_facts}f`;
    row.appendChild(chip);
  }
}

// ============================================================
// Text pane
// ============================================================

function _container(sectionPath) {
  if (!sectionPath) return null;
  if (sectionPath.startsWith("preamble")) return "preamble";
  if (sectionPath.startsWith("enacting")) return "enacting";
  return null;
}

function renderText() {
  state.factSpans.clear();
  const visible = visibleFacts();
  const segments = { preamble: [], enacting: [] };
  for (const f of visible) {
    const loc = f.source_locator || {};
    const c = _container(loc.section_path);
    if (!c || loc.char_start == null || loc.char_end == null) continue;
    segments[c].push({
      start: loc.char_start, end: loc.char_end,
      fact_id: f.fact_id,
      conflict: f._edge_conflict_label || "unlabeled",
    });
  }

  const body = $("#doc-body");
  body.innerHTML = "";
  body.appendChild(renderSection("PREAMBLE", state.textCache.preamble, segments.preamble));
  if ((state.textCache.enacting || "").trim())   // WebNLG / Re-DocRED docs have no
    body.appendChild(renderSection("ENACTING TEXT", state.textCache.enacting, segments.enacting));
}

function renderSection(name, text, segs) {
  const wrap = document.createElement("div");
  const h = document.createElement("div");
  h.className = "section-head";
  h.textContent = `— ${name} —`;
  if (segs.length === 0) {
    const hint = document.createElement("span");
    hint.className = "empty-section-hint";
    // "Blind spot" is a claim about the *guideline*, and it only holds when the
    // annotators ran and returned nothing here. On a document that was never
    // extracted every section would carry it, which is the one reading that
    // must not be invented — it is the very thing the diagnosis looks for.
    hint.textContent = (state.factsById && state.factsById.size)
      ? "no atomic facts extracted in this section (v1-guideline blind spot)"
      : "this document has not been annotated";
    h.appendChild(hint);
  }
  wrap.appendChild(h);

  // Count overlapping spans at each cursor so we can stack underline layers.
  // The "overlap depth" of a fact within a section is the number of OTHER
  // facts whose span already overlaps. We render every fact as its own mark
  // (one click target per row) but bump CSS overlay layers for legibility.
  segs = segs.slice().sort((a, b) => a.start - b.start || a.end - b.end);
  // Pass 1: compute depth per seg (count of earlier segs whose end > seg.start)
  for (let i = 0; i < segs.length; i++) {
    let d = 0;
    for (let j = 0; j < i; j++) {
      if (segs[j].end > segs[i].start && segs[j].start < segs[i].end) d++;
    }
    segs[i].depth = d;
  }

  let cursor = 0;
  for (const seg of segs) {
    if (seg.start > cursor) wrap.appendChild(document.createTextNode(text.slice(cursor, seg.start)));
    const start = Math.max(seg.start, cursor);
    const end = Math.max(seg.end, start);
    if (end > start) {
      const m = document.createElement("mark");
      m.dataset.factId = seg.fact_id;
      m.className = seg.conflict + (seg.depth >= 2 ? " overlay-3" : seg.depth === 1 ? " overlay-2" : "");
      m.textContent = text.slice(start, end);
      // Phase-6: paint a distinct color per fact (HSL hash of fact_id). The
      // CSS rule body[data-color-mode="fact"] reads --fact-color from inline.
      m.style.setProperty("--fact-color", colorForFact(seg.fact_id));
      m.addEventListener("click", () => selectFact(seg.fact_id));
      wrap.appendChild(m);
      if (!state.factSpans.has(seg.fact_id)) state.factSpans.set(seg.fact_id, m);
      cursor = end;
    }
  }
  if (cursor < text.length) wrap.appendChild(document.createTextNode(text.slice(cursor)));
  return wrap;
}

function refreshText() { renderText(); }

// ============================================================
// Annotator chips
// ============================================================

function renderChips() {
  const wrap = $("#annotator-chips");
  if (!wrap) return;
  wrap.innerHTML = "";
  const local = state.allAnnotators.filter(a => !a.startsWith("cloud:"));
  const cloud = state.allAnnotators.filter(a => a.startsWith("cloud:"));
  const onChange = () => { refreshText(); refreshFacts(); refreshGraphHighlights(); };
  if (!window.renderMultiDropdown) {              // fallback: plain chips
    for (const a of state.allAnnotators) {
      const c = document.createElement("span");
      c.className = "chip" + (state.annotators.has(a) ? " on" : "");
      c.textContent = a;
      c.addEventListener("click", () => {
        if (state.annotators.has(a)) state.annotators.delete(a); else state.annotators.add(a);
        c.classList.toggle("on"); onChange();
      });
      wrap.appendChild(c);
    }
    return;
  }
  const localHost = document.createElement("span");
  const cloudHost = document.createElement("span");
  window.renderMultiDropdown(localHost, {
    groups: [{ label: "", items: local.map(a => ({ value: a, label: a })) }],
    selected: state.annotators, single: false, placeholder: "Local", onChange,
  });
  window.renderMultiDropdown(cloudHost, {
    groups: [{ label: "", items: cloud.map(a => ({ value: a, label: a.replace(/^cloud:/, "") })) }],
    selected: state.annotators, single: false, placeholder: "Cloud", onChange,
  });
  wrap.appendChild(localHost);
  wrap.appendChild(cloudHost);
}

// ============================================================
// Facts table
// ============================================================

function visibleFacts() {
  const out = [];
  const q = state.search;
  for (const f of state.factsById.values()) {
    if (!state.annotators.has(f._annotator)) continue;
    if (state.conflict && (f._edge_conflict_label || "unlabeled") !== state.conflict) continue;
    if (q) {
      const hay = `${f.subject || ""} ${f.predicate || ""} ${f.object || ""}`.toLowerCase();
      if (!hay.includes(q)) continue;
    }
    out.push(f);
  }
  return out;
}

function _makeFactItem(f) {
  const conflict = f._edge_conflict_label || "unlabeled";
  const item = document.createElement("div");
  item.className = `fact-item conf-${conflict}`;
  item.dataset.factId = f.fact_id;
  const vstat = f._verification?.status || "";
  const vmark = vstat === "verified" ? "✓" : vstat === "rejected" ? "✗" : "?";
  const cond = f.condition ? `<span class="f-cond" title="condition">if ${escapeHtml(f.condition)}</span>` : "";
  const temp = f.temporal_context ? `<span class="f-temp" title="temporal order">⏱ ${escapeHtml(f.temporal_context)}</span>` : "";
  const sec = (f.source_locator || {}).section_path || "";
  item.innerHTML = `
    <button class="verify-btn ${vstat}" data-fact-id="${f.fact_id}">${vmark}</button>
    <div class="fact-body">
      <div class="fact-nl">${escapeHtml(f.natural_language || `${f.subject || ""} ${f.predicate || ""} ${f.object || ""}`)}${cond}${temp}</div>
      <div class="fact-spo"><span class="f-subj">${escapeHtml(f.subject || "")}</span> <span class="f-pred">${escapeHtml(f.predicate || "")}</span> <span class="f-obj">${escapeHtml(f.object || "")}</span><span class="f-sec">${escapeHtml(sec)}</span></div>
    </div>`;
  item.addEventListener("click", (e) => {
    if (e.target.classList.contains("verify-btn")) return;
    selectFact(f.fact_id);
  });
  return item;
}

function _sectionLabel(path) {
  if (!path || path === "(no section)") return path;
  const m1 = path.match(/^preamble\.recitals\[(\d+)\]$/);
  if (m1) return "Recital " + (+m1[1] + 1);
  const m2 = path.match(/^enacting\.article_(.+)$/);
  if (m2) return "Article " + m2[1];
  return path;
}

/** The source text of one segment, or "" if it cannot be located.
 *
 * The comparison the baseline reader makes readable is *per segment*: this is
 * what the three annotators were given, here is what each returned. Without the
 * source above the columns the reader has to hold the clause in their head
 * while scanning three lists, which is the work the view exists to remove.
 */
function _segmentText(path) {
  const doc = state.docParsed;
  const c = _container(path);
  if (!doc || !c) return "";
  const text = state.textCache[c] || "";
  let rec = null;
  const m1 = path.match(/^preamble\.recitals\[(\d+)\]$/);
  if (m1) rec = (doc.recitals || [])[+m1[1]];
  const m2 = path.match(/^enacting\.article_(.+)$/);
  if (m2) rec = (doc.articles || []).find((a) => String(a.number) === m2[1]);
  if (rec && rec.char_start != null && rec.char_end != null)
    return text.slice(rec.char_start, rec.char_end);
  if (rec && rec.text) return rec.text;
  return "";
}

/** section_path -> (annotator -> facts), in document order. */
function _bySection(facts) {
  const order = [];
  const bySec = new Map();
  for (const f of facts) {
    const sec = (f.source_locator || {}).section_path || "(no section)";
    if (!bySec.has(sec)) { bySec.set(sec, new Map()); order.push(sec); }
    const byAnnot = bySec.get(sec);
    if (!byAnnot.has(f._annotator)) byAnnot.set(f._annotator, []);
    byAnnot.get(f._annotator).push(f);
  }
  order.sort((a, b) => {
    const aPre = a.startsWith("preamble"), bPre = b.startsWith("preamble");
    if (aPre !== bPre) return aPre ? -1 : 1;
    return (parseInt(a.match(/\d+/)?.[0] ?? "0") - parseInt(b.match(/\d+/)?.[0] ?? "0"));
  });
  return order.map((sec) => [sec, bySec.get(sec)]);
}

/** "cloud:hf.co/unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL" -> "gemma-4-31B".
 *  A column head 40 characters wide is a column of one word per line. */
function _shortAnnot(a) {
  const s = String(a).replace(/^cloud:/, "").replace(/^hf\.co\/[^/]+\//, "")
    .replace(/-(it-)?GGUF.*$/, "").replace(/:.*$/, "").replace(/^NVIDIA-/, "");
  // a 45-character head turns a column into one word per line; the full name
  // stays on the element's title
  return s.length > 22 ? s.slice(0, 21) + "…" : s;
}

/** The model and guideline lists the annotate bar offers.
 *
 * Plain state, filled once, with a single re-render when it arrives — not a
 * promise per bar. The caches for both live in a later IIFE in this file
 * (`window.__vaModels` / `__vaGuidelines`) which has not necessarily run when
 * the facts pane first paints, and chaining off them there produced ten bars
 * whose lists never filled and nothing to show for it. A value you can look at
 * is debuggable; a promise captured in ten closures is not.
 */
// `/api/models` probes local Ollama AND the cloud box, so on a cold page it can
// take 5-20 s (and the DBvis box has a known ~50 s timeout under load,
// BUG-2026-08-31-D). The bar must not wait for it: the annotators that already
// worked on THIS document are known immediately and are the ones anyone
// re-runs, so the select is usable from the first paint and grows when the full
// list lands.
let _segLists = null;            // {models: {local, cloud}, guidelines: []}
let _segListsLoading = false;

function _loadSegLists() {
  if (_segLists || _segListsLoading) return;
  _segListsLoading = true;
  const models = window.__vaModels ? window.__vaModels()
    : fetch("/api/models").then((r) => r.json());
  const gl = window.__vaGuidelines?.();
  const guidelines = (gl && gl.length) ? Promise.resolve(gl)
    : fetch("/api/guidelines").then((r) => r.json()).then((d) => d.guidelines || []);
  Promise.all([models, guidelines])
    .then(([m, g]) => { _segLists = {models: m || {}, guidelines: g || []}; })
    .catch(() => { _segLists = {models: {}, guidelines: []}; })
    .then(() => {
      _segListsLoading = false;
      // one re-render, only if the pane is still showing segment blocks
      if (document.querySelector(".seg-run")) renderFacts();
    });
}

/** The per-segment annotate bar: pick an annotator, pick a guideline, run
 *  Phase-1 on THIS clause only.
 *
 * The same thing was already possible by selecting text in the left pane, which
 * means you had to find the clause in the running text first. The unit the
 * reader is working in is the segment block on the right — so the control
 * belongs on it. Both routes post to /api/reextract_span; this one sends
 * `section_path`, which is the exact anchor, rather than character offsets
 * recovered from a selection.
 */
const _segProbeNote = new Map();   // "doc|section" -> last probe's result line

function _segmentAnnotateBar(sec) {
  const bar = document.createElement("div");
  bar.className = "seg-run";
  const model = document.createElement("select");
  model.className = "seg-run-model";
  const guide = document.createElement("select");
  guide.className = "seg-run-guideline";
  const btn = document.createElement("button");
  btn.className = "tool-btn";
  btn.textContent = "Annotate this segment";
  btn.title = "Run Phase-1 on this clause only, with the chosen annotator and "
            + "guideline. Existing facts from this clause by that annotator are "
            + "replaced.";
  const status = document.createElement("span");
  status.className = "hint";
  status.textContent = _segProbeNote.get(`${state.doc}|${sec}`) || "";

  if (!_segLists) {
    _loadSegLists();
    // usable now: the annotators that produced the columns below
    for (const a of state.allAnnotators) {
      const o = document.createElement("option");
      o.value = a; o.textContent = _shortAnnot(a); o.title = a;
      model.appendChild(o);
    }
    const o = document.createElement("option");
    o.value = ""; o.textContent = "(loading the full model list…)";
    o.disabled = true;
    model.appendChild(o);
    const g = document.createElement("option");
    g.value = window.__variant || "v1"; g.textContent = window.__variant || "v1";
    guide.appendChild(g);
  } else {
    for (const [key, label] of [["local", "Local"], ["cloud", "Cloud"]]) {
      const list = _segLists.models[key] || [];
      if (!list.length) continue;
      const og = document.createElement("optgroup");
      og.label = label;
      for (const m of list) {
        const o = document.createElement("option");
        o.value = m.id; o.textContent = m.tag;
        og.appendChild(o);
      }
      model.appendChild(og);
    }
    // default to an annotator that already worked on this document, so the
    // obvious action — re-run the one that produced the column you are reading
    // — is the default one
    const here = state.allAnnotators.find((a) =>
      [...model.options].some((o) => o.value === a));
    if (here) model.value = here;

    const gl = _segLists.guidelines;
    for (const g of (gl.length ? gl : [{version: "v1"}])) {
      const o = document.createElement("option");
      o.value = g.version; o.textContent = g.version;
      guide.appendChild(o);
    }
    guide.value = window.__variant || "v1";
  }

  // The bar used to be built and never wired: "Annotate this segment" did
  // nothing, while iteration-log entry 6.6 and the System chapter describe it
  // as the probe (BUG-2026-09-23-D). The server writes into the chosen
  // version's own root and refuses a cell of a run of record or an eval
  // document; its reason is shown here rather than swallowed.
  btn.addEventListener("click", async (ev) => {
    ev.stopPropagation();
    if (!model.value) { status.textContent = "pick an annotator"; return; }
    btn.disabled = true;
    status.className = "hint";
    status.textContent = `annotating with ${_shortAnnot(model.value)} under ${guide.value}…`;
    try {
      const r = await fetch("/api/reextract_span", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({doc_id: state.doc, section_path: sec,
                              model: model.value, guideline_version: guide.value}),
      });
      const d = await r.json().catch(() => ({}));
      if (!r.ok) {
        status.className = "hint err";
        status.textContent = `refused (${r.status}): ${d.message || d.error || ""}`;
        return;
      }
      // The pane shows a version's conflicts file when it has one, so a probe
      // appears there only after Phase-2 re-runs — say where it went. The
      // reload rebuilds this bar, so the line is kept per segment.
      _segProbeNote.set(`${state.doc}|${sec}`,
        `${d.n_new_facts} triple(s) written to ${d.out_path}`
        + ((window.__variant || "v1") === guide.value ? ""
           : ` — select ${guide.value} in the top bar to see them`));
      $("#doc-select")?.dispatchEvent(new Event("change"));
    } catch (e) {
      status.className = "hint err";
      status.textContent = `request failed: ${e.message}`;
    } finally {
      btn.disabled = false;
    }
  });

  bar.append(model, guide, btn, status);
  return bar;
}

/** Cite what this pane is showing, in the form the change log stores
 *  (`src/guideline_log.CITE_FIELDS`), so the diagnosed arm's citations stay the
 *  same kind of object as the control's.
 *
 * Until now the only way in was `cite ->` on a reviewed conflict **pair** in the
 * Review tab, which can only say "these two annotators disagreed here". Most of
 * what a guideline revision is read off is not a pair: it is one model's
 * behaviour on one clause — gemma writing eighteen near-duplicates of a
 * proposition on Recital 1 is a defect of FORM with no partner to point at. So
 * the unit the reader is looking at gets its own button: the annotator column
 * (document + segment + model) and the single fact.
 */
function _citeButton(label, payload, title) {
  const b = document.createElement("button");
  b.className = "cite-btn";
  b.textContent = label;
  b.title = title;
  b.addEventListener("click", (ev) => {
    ev.stopPropagation();                 // a fact row is also a select target
    if (!window.GLog) return;
    const n = window.GLog.cite(payload);
    const was = b.textContent;
    b.textContent = `cited (${n})`;
    setTimeout(() => { b.textContent = was; }, 1500);
  });
  return b;
}

function renderFacts() {
  const container = $("#facts-table");
  container.innerHTML = "";
  const visible = visibleFacts();
  // triples, not facts: a fact is one quote with one or more triples, and
  // every row here is one triple (BUG-2026-09-23-I)
  $("#facts-count").textContent = `${visible.length} visible / ${state.factsById.size} total triples`;

  const sortMode = document.getElementById("facts-sort")?.value || "model";

  if (sortMode === "segment") {
    // The baseline reader's layout, inside the interface: one block per
    // segment, the source text on top, one COLUMN per annotator. Stacking the
    // three (the "by paragraph" mode) makes the reader scroll to compare, which
    // is exactly the comparison this pane exists for — so that mode is kept for
    // a narrow pane and this one is the side-by-side reading.
    for (const [sec, byAnnot] of _bySection(visible)) {
      const block = document.createElement("div");
      block.className = "seg-block";
      const head = document.createElement("div");
      head.className = "fact-section-head";
      const shown = state.allAnnotators.filter((a) => state.annotators.has(a));
      // counts in COLUMN order — taken from the Map's own order they would be
      // in whatever sequence the facts happened to arrive, i.e. a caption that
      // labels the wrong annotators
      head.textContent = _sectionLabel(sec) + "  ·  "
        + shown.map((a) => (byAnnot.get(a) || []).length).join(" / ") + " triples";
      head.appendChild(_citeButton("cite →", {
        kind: "segment", doc_id: state.doc, section: sec,
        annotator: shown.map(_shortAnnot).join(" / "),
        excerpt: shown.map((a) => `${_shortAnnot(a)}: ${(byAnnot.get(a) || []).length}`)
                      .join(", ") + " triples",
        version: window.__variant || "v1",
      }, "Cite this segment — all annotators and their counts — in the change log"));
      head.dataset.section = sec;
      head.classList.add("clickable");
      head.classList.toggle("active", state.segment === sec);
      head.title = "Show only this segment in the graph (click again to clear)";
      head.addEventListener("click", () => setSegmentFilter(sec));
      block.appendChild(head);
      const src = _segmentText(sec);
      if (src) {
        const pre = document.createElement("pre");
        pre.className = "seg-src";
        pre.textContent = src;
        block.appendChild(pre);
      }
      block.appendChild(_segmentAnnotateBar(sec));
      const cols = document.createElement("div");
      cols.className = "seg-cols";
      // Every annotator gets a column, including one that produced nothing
      // here: an empty column is the finding (a missed clause), and dropping it
      // would make the three look like they agreed.
      for (const annot of shown) {
        const col = document.createElement("div");
        col.className = "seg-col";
        col.dataset.annotator = annot;
        const ch = document.createElement("div");
        ch.className = "seg-col-head";
        const facts = byAnnot.get(annot) || [];
        ch.textContent = `${_shortAnnot(annot)} (${facts.length})`;
        ch.title = annot;
        ch.appendChild(_citeButton("cite →", {
          kind: "annotator_segment", doc_id: state.doc, section: sec,
          annotator: annot,
          excerpt: `${facts.length} triple${facts.length === 1 ? "" : "s"}`
                   + (facts[0] ? ": " + (facts[0].natural_language || "").slice(0, 160) : ""),
          version: window.__variant || "v1",
        }, "Cite this model's output on this segment in the change log"));
        // the annotator palette is owned by the legend IIFE (window.__vaColor);
        // one palette for the legend, the text marks and these columns, or the
        // same annotator is a different colour in three places
        ch.style.setProperty("--annot-color",
                             window.__vaColor?.(annot) || "#9ca3af");
        col.appendChild(ch);
        if (!facts.length) {
          const none = document.createElement("div");
          none.className = "seg-none";
          none.textContent = "— nothing here —";
          col.appendChild(none);
        }
        for (const f of facts) {
          const item = _makeFactItem(f);
          // The identifying fields the control console stores, computed the
          // same way (`make_baseline_reader`: group_id, else fact_id), so a
          // citation from either arm resolves to the same annotation. Without
          // an id this stored only an excerpt, and two annotations of one
          // model on one clause had the same citation key — citing the second
          // silently un-cited the first (BUG-2026-09-23-H). `version` because
          // the ids do not change between guideline versions.
          item.appendChild(_citeButton("❝", {
            kind: "fact", doc_id: state.doc, section: sec, annotator: annot,
            group_id: f.group_id || f.fact_id,
            label: f._edge_conflict_label || "unlabeled",
            excerpt: (f.natural_language
                      || `${f.subject || ""} ${f.predicate || ""} ${f.object || ""}`)
                     .slice(0, 240),
            version: window.__variant || "v1",
          }, "Cite this single annotation in the change log"));
          col.appendChild(item);
        }
        cols.appendChild(col);
      }
      block.appendChild(cols);
      container.appendChild(block);
    }
  } else if (sortMode === "paragraph") {
    // Group by section_path in document order
    const sectionOrder = [];
    const sectionFacts = new Map(); // section_path → Map<annotator → fact[]>
    for (const f of visible) {
      const sec = (f.source_locator || {}).section_path || "(no section)";
      if (!sectionFacts.has(sec)) { sectionFacts.set(sec, new Map()); sectionOrder.push(sec); }
      const byAnnot = sectionFacts.get(sec);
      if (!byAnnot.has(f._annotator)) byAnnot.set(f._annotator, []);
      byAnnot.get(f._annotator).push(f);
    }
    // Sort in document order: preamble before enacting, then numerically
    sectionOrder.sort((a, b) => {
      const aPre = a.startsWith("preamble"), bPre = b.startsWith("preamble");
      if (aPre !== bPre) return aPre ? -1 : 1;
      const na = parseInt(a.match(/\d+/)?.[0] ?? "0");
      const nb = parseInt(b.match(/\d+/)?.[0] ?? "0");
      return na - nb;
    });
    for (const sec of sectionOrder) {
      const secDiv = document.createElement("div");
      secDiv.className = "fact-group fact-group-section";
      const secHead = document.createElement("div");
      secHead.className = "fact-section-head";
      secHead.textContent = _sectionLabel(sec);
      secDiv.appendChild(secHead);
      const byAnnot = sectionFacts.get(sec);
      for (const [annot, facts] of byAnnot) {
        const annotHead = document.createElement("div");
        annotHead.className = "fact-annotator-head";
        annotHead.textContent = `${annot} (${facts.length})`;
        secDiv.appendChild(annotHead);
        for (const f of facts) secDiv.appendChild(_makeFactItem(f));
      }
      container.appendChild(secDiv);
    }
  } else {
    // Original by-model layout
    const groups = new Map();
    for (const f of visible) {
      if (!groups.has(f._annotator)) groups.set(f._annotator, []);
      groups.get(f._annotator).push(f);
    }
    for (const [annot, facts] of groups) {
      const grp = document.createElement("div");
      grp.className = "fact-group";
      grp.dataset.annotator = annot;
      const head = document.createElement("div");
      head.className = "fact-group-head";
      head.textContent = `${annot} — ${facts.length} triples`;
      grp.appendChild(head);
      for (const f of facts) grp.appendChild(_makeFactItem(f));
      container.appendChild(grp);
    }
  }
  for (const btn of container.querySelectorAll(".verify-btn")) {
    btn.addEventListener("click", (e) => cycleVerify(e.target));
  }
  updateLabelCounts(visible);
}

function refreshFacts() { renderFacts(); }

function updateLabelCounts(visible) {
  const c = { contradiction: 0, granularity: 0, redundancy: 0, unlabeled: 0 };
  for (const f of visible) {
    const k = f._edge_conflict_label || "unlabeled";
    c[k] = (c[k] || 0) + 1;
  }
  $("#label-counts").textContent =
    `▮ ${c.contradiction} contra · ${c.granularity} gran · ${c.redundancy} redu · ${c.unlabeled} unl`;
}

async function cycleVerify(btn) {
  const cur = btn.classList.contains("verified") ? "verified"
            : btn.classList.contains("rejected") ? "rejected" : "";
  const next = { "": "verified", verified: "rejected", rejected: "" }[cur];
  const factId = btn.dataset.factId;
  const r = await fetch(`/api/verify/${state.doc}/${factId}`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status: next === "" ? "unset" : next }),
  }).then(r => r.json());
  btn.classList.remove("verified", "rejected");
  if (r.status === "verified") { btn.classList.add("verified"); btn.textContent = "✓"; }
  else if (r.status === "rejected") { btn.classList.add("rejected"); btn.textContent = "✗"; }
  else { btn.textContent = "?"; }
}

// ============================================================
// Graph pane
// ============================================================

// Cytoscape extension registration. The UMD bundles (cytoscape-fcose,
// cytoscape-dagre) sometimes auto-register on window.cytoscape and sometimes
// don't (depends on the load order and CDN variant). Calling `use()` again
// when it's already registered is a no-op, so this is safe.
function _registerCyExtensions() {
  if (typeof cytoscape === "undefined") return;
  try { if (typeof window.cytoscapeFcose === "function") cytoscape.use(window.cytoscapeFcose); } catch (e) { console.warn("fcose register:", e); }
  try { if (typeof window.cytoscapeDagre === "function") cytoscape.use(window.cytoscapeDagre); } catch (e) { console.warn("dagre register:", e); }
}
_registerCyExtensions();

// Returns true if the named layout is actually available on this cytoscape build.
function _layoutAvailable(name) {
  if (!["fcose", "dagre"].includes(name)) return true;       // built-ins always there
  try {
    const probe = cytoscape({ headless: true, elements: [] });
    const out = !!probe.extension && !!probe.extension("layout", name);
    probe.destroy && probe.destroy();
    return out;
  } catch { return false; }
}

// Shared layout-options builder so the dropdown AND the edge-length slider
// can drive cytoscape with the same parameters. Picks safe defaults per
// algorithm and folds in state.edgeLength as the ideal-edge-length / spacing.
function _buildLayoutOpts(name, nodeCount) {
  // If a third-party layout isn't registered, fall back to cose so the UI
  // still works instead of throwing inside cy.layout().run().
  if (!_layoutAvailable(name)) {
    console.warn(`Layout "${name}" not registered — falling back to cose.`);
    name = "cose";
  }
  // Tiny graphs render best with concentric — cose collapses them into a blob.
  if (name === "cose" && nodeCount <= 30) name = "concentric";
  const len = Math.max(40, state.edgeLength || 160);
  const opts = {
    name, animate: false, padding: 40,
    nodeDimensionsIncludeLabels: true, avoidOverlap: true,
  };
  if (name === "cose") {
    opts.idealEdgeLength = len;
    opts.nodeOverlap = 30;
    opts.nodeRepulsion = 8000;
  } else if (name === "fcose") {
    // Optimized force-directed profile (fcose = FMMM-style multilevel):
    //  - quality "proof" + 5k iterations: slower (~100ms on 60 nodes) but
    //    visibly better minima than the "default" single pass.
    //  - randomize true: a fresh global arrangement instead of polishing
    //    whatever the previous layout left behind.
    //  - degree-scaled edge length / repulsion: hub neighbourhoods get room
    //    to breathe, leaf pairs stay compact.
    //  - label-aware spacing (nodeDimensionsIncludeLabels is set globally).
    opts.quality = "proof";
    opts.randomize = true;
    opts.idealEdgeLength = (e) => {
      const d = Math.max(e.source().degree(false), e.target().degree(false));
      return len * (d >= 4 ? 1.5 : d >= 2 ? 1.1 : 0.8);
    };
    opts.nodeRepulsion = (n) => 6000 + 2500 * Math.min(n.degree(false), 6);
    opts.edgeElasticity = 0.45;
    opts.gravity = 0.2;
    opts.numIter = 5000;
    opts.tile = true;
    opts.nodeSeparation = 30;
  } else if (name === "dagre") {
    // dagre = Sugiyama-style hierarchical (top-to-bottom by default).
    opts.rankDir = "TB";
    opts.nodeSep = Math.max(30, len * 0.35);
    opts.rankSep = Math.max(60, len * 0.6);
    opts.edgeSep = 16;
  } else if (name === "concentric") {
    opts.minNodeSpacing = 50;
    opts.concentric = (n) => n.data("n_annotators") || 1;
    opts.levelWidth = () => 1;
  } else if (name === "breadthfirst") {
    opts.spacingFactor = Math.max(1.0, len / 140);
  }
  return opts;
}

function renderGraph(graph) {
  if (state.cy) state.cy.destroy();

  const nodeCount = graph.nodes.length;
  const layoutOpts = _buildLayoutOpts(state.layout, nodeCount);
  // On crowded graphs, clause-like leaf nodes (degree 1, usually a whole
  // object phrase) keep their label OFF — hubs carry the structure; leaves
  // are read via hover/selection. Small graphs label everything.
  const labelLeaves = nodeCount <= 25;

  state.cy = cytoscape({
    container: document.getElementById("cy"),
    elements: [...graph.nodes, ...graph.edges],
    layout: layoutOpts,
    minZoom: 0.2,
    maxZoom: 2.5,
    wheelSensitivity: 0.3,
    style: [
      { selector: "node", style: {
          "background-color": "#e2e8f0",
          "label": (ele) => (labelLeaves || ele.degree(false) >= 2 ||
                             ele.hasClass("selected") || ele.hasClass("spo-subj") ||
                             ele.hasClass("spo-obj") || ele.hasClass("mm-sel"))
                            ? truncate(ele.data("label") || "", 24) : "",
          "font-size": 13,
          "font-weight": 500,
          "text-wrap": "wrap",
          "text-max-width": 140,
          "color": "#1f2937",
          "text-valign": "center",
          "text-halign": "center",
          "text-outline-color": "#fff",
          "text-outline-width": 2,
          "width":  "mapData(n_annotators, 1, 3, 44, 78)",
          "height": "mapData(n_annotators, 1, 3, 44, 78)",
          "border-width": 2,
          "border-color": "#94a3b8" }},
      // Reified statement nodes: rounded rectangles, distinct from entity
      // ellipses. Predicate is the label; condition/order ride the edges.
      { selector: 'node[type = "statement"]', style: {
          "shape": "round-rectangle",
          "background-color": "#eef2ff",
          "border-color": "#6366f1",
          "border-width": 2,
          "width": "mapData(n_annotators, 1, 3, 34, 66)",
          "height": 24,
          "font-size": 11,
          "font-weight": 600,
          "text-max-width": 130,
          "color": "#312e81" }},
      // Same S-P-O realized with different condition/ordering scope.
      { selector: 'node[type = "statement"][?scope_variant]', style: {
          "border-style": "double", "border-width": 4, "border-color": "#7c3aed" }},
      { selector: 'node[type = "statement"][?negated]', style: {
          "background-color": "#fef2f2" }},
      { selector: "node[conflict_label = 'contradiction']", style: { "border-color": "#d32f2f", "border-width": 4 }},
      { selector: "node[conflict_label = 'granularity']",   style: { "border-color": "#f57c00", "border-width": 4 }},
      { selector: "node[conflict_label = 'redundancy']",    style: { "border-color": "#1976d2", "border-width": 3 }},
      { selector: "node[conflict_label = 'escalate']",      style: { "border-color": "#8e24aa", "border-width": 3, "border-style": "dashed" }},
      { selector: "node.selected", style: { "background-color": "#fef08a", "border-color": "#eab308", "border-width": 5 }},
      { selector: "node.dimmed", style: { "opacity": 0.25 }},
      { selector: "edge", style: {
          "width": "mapData(n_annotators, 1, 3, 2, 5)",
          "line-color": "#94a3b8",
          "target-arrow-color": "#94a3b8",
          "target-arrow-shape": "triangle",
          "arrow-scale": 1.4,
          "curve-style": "bezier",
          "label": "data(label)",
          "font-size": 11,
          "color": "#475569",
          "text-rotation": "autorotate",
          "text-background-color": "#fff",
          "text-background-opacity": 0.9,
          "text-background-padding": 3,
          "text-background-shape": "roundrectangle" }},
      { selector: 'edge[role = "subject"]', style: {
          "line-color": "#cbd5e1", "target-arrow-color": "#cbd5e1",
          "target-arrow-shape": "triangle", "width": 2, "label": "" }},
      { selector: 'edge[role = "object"]', style: {
          "target-arrow-shape": "triangle", "label": "" }},
      { selector: "edge[conflict_label = 'contradiction']", style: { "line-color": "#d32f2f", "target-arrow-color": "#d32f2f", "width": 4 }},
      { selector: "edge[conflict_label = 'granularity']",   style: { "line-color": "#f57c00", "target-arrow-color": "#f57c00", "width": 3.5 }},
      { selector: "edge[conflict_label = 'redundancy']",    style: { "line-color": "#1976d2", "target-arrow-color": "#1976d2", "width": 3 }},
      { selector: "edge[conflict_label = 'escalate']",      style: { "line-color": "#8e24aa", "target-arrow-color": "#8e24aa", "width": 3, "line-style": "dashed" }},
      { selector: "edge.gold-match",    style: { "line-color": "#16a34a", "target-arrow-color": "#16a34a", "width": 4 }},
      { selector: "edge.gold-spurious", style: { "line-color": "#dc2626", "target-arrow-color": "#dc2626", "width": 4, "line-style": "dashed" }},
      { selector: "edge.selected", style: { "line-color": "#eab308", "target-arrow-color": "#eab308", "width": 5 }},
      { selector: "edge.dimmed", style: { "opacity": 0.2 }},
      { selector: "edge[?conditional]", style: { "line-style": "dashed" }},
      { selector: 'edge[role = "condition"]', style: {
          "line-style": "dashed", "line-color": "#7c3aed",
          "target-arrow-color": "#7c3aed", "target-arrow-shape": "triangle",
          "width": 2.5, "label": "data(label)", "color": "#6d28d9",
          "font-size": 10, "text-background-color": "#f5f3ff",
          "text-background-opacity": 1, "text-background-padding": 2 }},
      { selector: 'edge[role = "temporal"]', style: {
          "line-style": "dotted", "line-color": "#0d9488",
          "target-arrow-color": "#0d9488", "target-arrow-shape": "vee",
          "width": 2.5, "label": "data(label)", "color": "#0f766e",
          "font-size": 10, "text-background-color": "#f0fdfa",
          "text-background-opacity": 1, "text-background-padding": 2 }},
      { selector: "node.stmt-sel", style: {
          "background-color": "#fde68a", "border-color": "#d97706", "border-width": 5 }},
      { selector: ".cy-hidden", style: { "display": "none" }},
      // Phase-6 tri-color S/P/O: subject node = blue, object node = red,
      // predicate edge = green. Stronger borders so they read over conflict colors.
      { selector: "node.spo-subj", style: { "background-color": "#dbeafe", "border-color": "#2563eb", "border-width": 6 }},
      { selector: "node.spo-obj",  style: { "background-color": "#fee2e2", "border-color": "#dc2626", "border-width": 6 }},
      { selector: "edge.spo-pred", style: { "line-color": "#059669", "target-arrow-color": "#059669", "width": 6 }},
    ],
  });

  // Double-tap an edge -> open the conflict-detection (pair) modal. Single-tap
  // is handled by highlightByEdge (node-highlight IIFE) and only highlights the
  // matching annotation(s) in the text + facts panes.
  state.cy.on("dbltap", "edge", (evt) => {
    const role = evt.target.data("role");
    if (role === "subject" || role === "object" || role === "condition" || role === "temporal") {
      // Reified: a subject/object/condition/temporal edge belongs to a statement
      // node — open that statement's detail rather than treating it as a pair edge.
      const stmt = role === "subject" ? evt.target.target() : evt.target.source();
      openPairModal(stmt.data("type") === "statement" ? stmt : evt.target);
    } else {
      openPairModal(evt.target);
    }
  });
  state.cy.on("tap", "node", (evt) => {
    if (evt.target.data("type") === "statement") openPairModal(evt.target);
    else filterByEntity(evt.target.id());
  });
  // Hover readout: node labels are truncated to 24 chars, so surface the
  // full entity label + membership in the legend strip under the canvas.
  state.cy.on("mouseover", "node", (evt) => {
    const d = evt.target.data();
    const el = document.getElementById("cy-hover-info");
    if (!el) return;
    if (d.type === "statement") {
      const bits = [`[${d.conflict_label || "unlabeled"}] ${d.subject_label || "?"} —[${d.predicate || ""}]→ ${d.object_label || "?"}`];
      if ((d.conditions || []).length) bits.push(`if: ${d.conditions.join("; ")}`);
      if ((d.temporal || []).length) bits.push(`⏱ ${d.temporal.join("; ")}`);
      if (d.negated) bits.push("¬negated");
      if (d.complex) bits.push("complex-logic");
      if (d.scope_variant) bits.push("scope-variant");
      el.textContent = bits.join("  ·  ") + `  ·  ${(d.annotators || []).join(" + ")}`;
    } else {
      el.textContent = `${d.label} · ${(d.surface_forms || []).length} surface · ${(d.annotators || []).length} annot`
        + (d.ref_only ? " · (referenced by a condition/ordering)" : "");
    }
  });
  state.cy.on("mouseover", "edge", (evt) => {
    const d = evt.target.data();
    const el = document.getElementById("cy-hover-info");
    const quals = (d.qualifiers || []).length
      ? ` · ⟨${d.qualifiers.join("; ")}⟩` : "";
    const cond = d.conditional ? " · ⧟conditional" : "";
    if (el) el.textContent =
      `[${d.conflict_label || "unlabeled"}] ${d.label} · ${(d.annotators || []).join(" + ")}${quals}${cond}`;
  });
  state.cy.on("mouseout", "node, edge", () => {
    const el = document.getElementById("cy-hover-info");
    if (el) el.textContent = "";
  });
  // Focus mode: double-tap a node → only its 1-hop neighborhood stays; the
  // analyst inspects one entity's disagreements without the rest of the hairball.
  // Double-tap empty background → restore the filtered view.
  state.cy.on("dbltap", "node", (evt) => {
    const hood = evt.target.closedNeighborhood();
    state.cy.batch(() => {
      state.cy.elements().addClass("cy-hidden");
      hood.removeClass("cy-hidden");
    });
    state.cy.fit(hood, 60);
  });
  state.cy.on("dbltap", (evt) => {
    if (evt.target === state.cy) restoreFullView();
  });
  // Expose for the splitter resize observer (Phase-5c).
  window.__cyInstance = state.cy;
  // Fit the graph nicely into the viewport on first render.
  setTimeout(() => state.cy && state.cy.fit(undefined, 50), 50);
  refreshGraphHighlights();
}

async function refreshGraph() {
  if (!state.doc) return;
  const graph = await fetch(graphUrl(state.doc)).then(r => r.json());
  state.graph = graph;
  renderGraph(graph);
}

/** Fact ids anchored in one segment. */
function _factsInSegment(sec) {
  const out = new Set();
  for (const [fid, f] of state.factsById)
    if (((f.source_locator || {}).section_path || "") === sec) out.add(fid);
  return out;
}

/** Read the graph one segment at a time.
 *
 * A whole-document KG is a hairball — 174 nodes for a four-page regulation —
 * and it is not where the diagnosis is read anyway: the question is always
 * "what did the three do with THIS clause". So the facts pane's segment blocks
 * and the graph now share one selection: pick a segment on the right and the
 * graph keeps only what is anchored there, plus the edges that leave it (a
 * conflict whose partner sits in another segment is a finding, not noise —
 * hiding the edge would make the pair look unmatched).
 */
function setSegmentFilter(sec) {
  state.segment = (sec === null || state.segment === sec) ? null : sec;
  for (const h of document.querySelectorAll(".seg-block .fact-section-head"))
    h.classList.toggle("active", h.dataset.section === state.segment);
  const chip = $("#cy-segment-chip");
  if (chip) {
    chip.classList.toggle("hidden", !state.segment);
    chip.firstChild.textContent = state.segment
      ? `segment: ${_sectionLabel(state.segment)} ` : "";
  }
  refreshGraphHighlights();
  // The layout still places the nodes as if the whole document were there, so
  // 27 survivors sit scattered across a canvas sized for 174. Re-run the layout
  // on what is left and fit to it — otherwise "filter to one segment" produces
  // a sparser hairball rather than a readable one.
  if (state.cy) {
    const vis = state.cy.elements().not(".cy-hidden");
    if (vis.nonempty()) {
      vis.layout(_buildLayoutOpts(state.layout, vis.nodes().length)).run();
      setTimeout(() => state.cy.fit(state.cy.elements().not(".cy-hidden"), 40), 60);
    }
  }
}

function refreshGraphHighlights() {
  // Annotator chips, the conflict filter and the segment selection drive the
  // KG: edges that don't match are HIDDEN (not dimmed) and nodes with no
  // visible edge follow. Dimming kept the clutter; hiding is what makes a
  // 60-node KG readable.
  if (!state.cy) return;
  const annotSel = state.annotators;
  const allOn = annotSel.size >= state.allAnnotators.length;
  const segFacts = state.segment ? _factsInSegment(state.segment) : null;
  const inSegment = (el) => {
    if (!segFacts) return true;
    const ids = el.data("fact_ids") || [];
    if (ids.some((fid) => segFacts.has(fid))) return true;
    // statement nodes carry their own anchors, which survive a fact id the
    // conflicts file knows and the facts response does not
    return (el.data("section_paths") || []).includes(state.segment);
  };
  state.cy.batch(() => {
    state.cy.elements().removeClass("dimmed cy-hidden");
    state.cy.edges().forEach(e => {
      const eAnn = e.data("annotators") || [];
      const annotHit = allOn || eAnn.some(a => annotSel.has(a));
      const conflictHit = !state.conflict || e.data("conflict_label") === state.conflict;
      // an edge survives if EITHER end is in the segment, so a conflict that
      // crosses a segment boundary stays visible
      const segHit = !segFacts || inSegment(e) || inSegment(e.source())
                     || inSegment(e.target());
      if (!annotHit || !conflictHit || !segHit) e.addClass("cy-hidden");
    });
    state.cy.nodes().forEach(n => {
      if (!n.connectedEdges().some(e => !e.hasClass("cy-hidden"))) n.addClass("cy-hidden");
    });
    // "Hide leaf islands": drop connected components with <=2 visible nodes
    // (single-fact fragments). They dominate clause-object KGs and bury the
    // hub structure the analyst actually reads.
    if ($("#cy-hide-islands")?.checked) {
      const visible = state.cy.elements().not(".cy-hidden");
      // In the reified graph a minimal unit is entity→statement→entity (3 nodes);
      // only drop true singletons so conditional/one-ended statements survive.
      const islandMax = (state.graph && state.graph.reified) ? 1 : 2;
      for (const comp of visible.components()) {
        if (comp.nodes().length <= islandMax) comp.addClass("cy-hidden");
      }
    }
  });
}

function filterByEntity(clusterId) {
  state.cy.elements().removeClass("selected");
  state.cy.getElementById(clusterId).addClass("selected");
  const node = state.cy.getElementById(clusterId);
  const surfaceForms = new Set(node.data("surface_forms") || []);

  const container = $("#facts-table");
  for (const item of container.querySelectorAll(".fact-item")) {
    const f = state.factsById.get(item.dataset.factId);
    const hit = f && (surfaceForms.has(f.subject) || surfaceForms.has(f.object));
    item.style.display = hit ? "" : "none";
  }
  for (const grp of container.querySelectorAll(".fact-group")) {
    const any = [...grp.querySelectorAll(".fact-item")].some(it => it.style.display !== "none");
    grp.style.display = any ? "" : "none";
  }
}

// Exit KG "focus mode": double-tapping the graph background must restore BOTH
// panes, not just the KG. A node double-tap filters two panes at once -- the
// preceding tap runs filterByEntity (hides non-matching rows in the AT/facts
// list), the dbltap does the 1-hop KG focus -- so the restore has to undo both.
// These are COUPLED. This bug keeps regressing because a later KG edit rewrites
// the dbltap-background handler and silently drops the facts half. Keep both
// halves in this one function. See memory: kg-focus-restore-regression.
function restoreFullView() {
  if (state.cy) {
    refreshGraphHighlights();                 // KG: un-hide all edges/nodes
    state.cy.elements().removeClass("selected");
    state.cy.fit(undefined, 50);
  }
  renderFacts();   // AT: rebuild #facts-table from visibleFacts(), dropping the
                   // entity filter's display:none rows left by filterByEntity().
}

// ============================================================
// Pair detail modal — opens on edge click
// ============================================================

function openPairModal(edge) {
  const factIds = new Set(edge.data("fact_ids") || []);
  // Find AlignedPair rows whose fact_a or fact_b is on this edge.
  const relevant = state.alignedPairs.filter(p =>
    factIds.has(p.fact_a_id) || factIds.has(p.fact_b_id));

  const body = $("#modal-body");
  if (edge.data("type") === "statement") {
    const cond = (edge.data("conditions") || []).join("; ");
    const temp = (edge.data("temporal") || []).join("; ");
    $("#modal-title").textContent =
      `Statement: ${edge.data("subject_label") || "?"} —[${edge.data("predicate") || ""}]→ ${edge.data("object_label") || "?"}`
      + (cond ? `  · if ${cond}` : "") + (temp ? `  · ⏱ ${temp}` : "")
      + `  ·  ${relevant.length} pairs`;
  } else {
    $("#modal-title").textContent =
      `Edge: ${edge.data("source")} —[${edge.data("label")}]→ ${edge.data("target")}  ·  ${relevant.length} pairs`;
  }
  body.innerHTML = "";

  if (relevant.length === 0) {
    // No aligned pair (single-annotator edge). Show every fact_id on the edge.
    for (const fid of factIds) {
      const f = state.factsById.get(fid);
      if (!f) continue;
      body.appendChild(renderSingleFactRow(f));
    }
  } else {
    for (const p of relevant) body.appendChild(renderPairRow(p));
  }
  $("#modal-overlay").classList.remove("hidden");
}

function renderPairRow(p) {
  const div = document.createElement("div");
  div.className = "pair-row";
  const lbl = p.conflict_label || "unlabeled";
  div.innerHTML = `
    <h4>
      <span title="${escapeHtml(p.annotator_a)} ↔ ${escapeHtml(p.annotator_b)}">${escapeHtml(_shortAnnot(p.annotator_a))} ↔ ${escapeHtml(_shortAnnot(p.annotator_b))}</span>
      <span class="cos">cos ${(p.cosine ?? 0).toFixed(3)} · ${escapeHtml(p.status || "")}</span>
      <span class="lbl ${lbl}">${lbl}</span>
    </h4>`;
  // Cite this conflict detection. The key is `fact_a|fact_b`, the same one the
  // review store and the Review tab's `cite ->` use, so a pair cited from the
  // graph and the same pair cited from a review are one citation, not two
  // records of one thing.
  {
    const fa = p.fact_a_id && state.factsById.get(p.fact_a_id);
    const fb = p.fact_b_id && state.factsById.get(p.fact_b_id);
    div.querySelector("h4").appendChild(_citeButton("cite →", {
      kind: "pair", doc_id: state.doc,
      pair_key: `${p.fact_a_id || "-"}|${p.fact_b_id || "-"}`,
      label: lbl, annotator: `${p.annotator_a} ↔ ${p.annotator_b}`,
      section: ((fa || fb || {}).source_locator || {}).section_path || "",
      version: window.__variant || "v1",
      // both sides plus the arbitrator's own words: a conflict citation that
      // needs the view reopened to be understood is not much of a citation
      excerpt: `A: ${(fa?.natural_language || "(missing)").slice(0, 120)}`
             + ` | B: ${(fb?.natural_language || "(missing)").slice(0, 120)}`
             + (p.layer2_reason ? ` | L2: ${p.layer2_reason.slice(0, 160)}` : ""),
    }, "Cite this conflict detection in the change log"));
  }
  // A and B as two columns, each with its annotator on its own line. The old
  // layout put the annotator in a 90px label column beside the text: an hf.co
  // tag is 60 characters, so the name wrapped into a tall stack that ran over
  // the sentence it was labelling. A model name is a label, not a margin note —
  // and the two facts are here to be compared, which wants them side by side.
  const grid = document.createElement("div");
  grid.className = "side";
  for (const [side, fid, ann] of [["A", p.fact_a_id, p.annotator_a], ["B", p.fact_b_id, p.annotator_b]]) {
    const f = fid && state.factsById.get(fid);
    grid.innerHTML += `
      <div class="side-col">
        <div class="ann-tag" title="${escapeHtml(ann)}">${escapeHtml(side)} · ${escapeHtml(_shortAnnot(ann))}</div>
        <div class="nl">${escapeHtml(f?.natural_language || "(missing)")}</div>
        <div class="quote">"${escapeHtml((f?.source_locator || {}).quote || "")}"</div>
      </div>`;
  }
  div.appendChild(grid);
  if (p.layer1_reason || p.layer2_reason) {
    const r = document.createElement("div");
    r.className = "reasons";
    r.textContent = `L1: ${p.layer1_reason || "—"}   L2: ${p.layer2_reason || "—"}`;
    div.appendChild(r);
  }
  // Click anywhere in the row → also select the fact in the three panes.
  div.addEventListener("click", () => {
    if (p.fact_a_id) selectFact(p.fact_a_id);
    else if (p.fact_b_id) selectFact(p.fact_b_id);
  });
  return div;
}

function renderSingleFactRow(f) {
  const div = document.createElement("div");
  div.className = "pair-row";
  div.innerHTML = `
    <h4>
      <span title="${escapeHtml(f._annotator)}">${escapeHtml(_shortAnnot(f._annotator))} (single-annotator edge)</span>
      <span class="lbl unlabeled">solo</span>
    </h4>
    <div class="side">
      <div class="side-col">
        <div class="ann-tag" title="${escapeHtml(f._annotator)}">${escapeHtml(_shortAnnot(f._annotator))}</div>
        <div class="nl">${escapeHtml(f.natural_language || "")}</div>
        <div class="quote">"${escapeHtml((f.source_locator || {}).quote || "")}"</div>
      </div>
    </div>`;
  div.addEventListener("click", () => selectFact(f.fact_id));
  return div;
}

function closeAllModals() {
  $("#modal-overlay").classList.add("hidden");
  $("#compare-overlay").classList.add("hidden");
  $("#gold-overlay")?.classList.add("hidden");
}

// ============================================================
// Gold-standard comparison (WebNLG only)
// ============================================================

function _updateGoldCompareBtn(docId) {
  const btn = $("#gold-compare-btn");
  if (!btn) return;
  const d = (state.docs || []).find(x => x.doc_id === docId);
  const src = d ? _docSource(d) : String(docId || "").split("-")[0];
  const hasGold = src === "webnlg" || src === "redocred";
  btn.classList.toggle("hidden", !hasGold);
  state.gold = null;
  if (state.goldCy) { try { state.goldCy.destroy(); } catch (e) {} state.goldCy = null; }
}

async function openGoldCompare() {
  const docId = state.doc;
  if (!docId) return;
  const body = $("#gold-body");
  body.innerHTML = "<em>scoring…</em>";
  $("#gold-overlay").classList.remove("hidden");
  try {
    const r = await fetch("/api/gold_compare/" + encodeURIComponent(docId));
    if (!r.ok) {
      let msg = "HTTP " + r.status;
      try { msg = (await r.json()).message || msg; } catch {}
      body.innerHTML = `<p class="err">${escapeHtml(msg)}</p>`;
      return;
    }
    _renderGoldCompare(await r.json());
  } catch (e) {
    body.innerHTML = `<p class="err">failed: ${escapeHtml(e.message)}</p>`;
  }
}

function _gcRows(list, cls, info) {
  if (!list || !list.length) return `<tr><td colspan="4" class="gc-empty">none</td></tr>`;
  return list.map(it => {
    const t = it.triple || it.gold || [];
    return `<tr class="${cls}"><td>${escapeHtml(t[0]||"")}</td>` +
           `<td>${escapeHtml(t[1]||"")}</td><td>${escapeHtml(t[2]||"")}</td>` +
           `<td class="gc-info">${info ? info(it) : ""}</td></tr>`;
  }).join("");
}

function _renderGoldCompare(data) {
  const soft = data.soft || null;                 // headline; strict kept inline
  const m = (soft && soft.metrics) || data.metrics || {};
  const sm = data.metrics || {};
  const pct = x => (100 * (x || 0)).toFixed(1) + "%";
  const annot = (data.annotators || []).join(", ") || "—";
  const hint = (m.n_extracted ? "" : " Run Phase-1 extraction on this doc to populate facts.");
  const backend = data.backend || "?";
  const bWarn = backend === "sbert" ? "" :
    ` <b class="gc-warn">backend=${escapeHtml(backend)} — fallback, do not report these numbers</b>`;
  const matched = soft ? (soft.matched || []) : (data.matched || []);
  const missing = soft ? (soft.missing || []) : (data.missing || []);
  const spurious = soft ? (soft.spurious || []) : (data.spurious || []);
  const simInfo = it => (it.sims ? `sim ${it.sims.overall}` : "");
  const reasonInfo = it => {
    let x = it.reason ? `<span class="gc-reason">${escapeHtml(it.reason)}</span>` : "";
    const b = it.best || it.best_gold;
    if (b) {
      const t = b.pred || b.triple || [];
      x += `<br><small>closest: ${escapeHtml(t.join(" | "))}</small>`;
    }
    return x;
  };
  const corpus = _goldCorpusLabel();
  $("#gold-body").innerHTML = `
    <div class="gc-metrics">
      <div class="gc-card"><b>${pct(m.precision)}</b><span>${soft ? "soft " : ""}precision</span></div>
      <div class="gc-card"><b>${pct(m.recall)}</b><span>${soft ? "soft " : ""}recall</span></div>
      <div class="gc-card"><b>${pct(m.f1)}</b><span>${soft ? "soft " : ""}F1</span></div>
      <div class="gc-card"><b>${m.n_matched||0}/${m.n_gold||0}</b><span>matched / gold</span></div>
      <div class="gc-card"><b>${m.n_spurious||0}</b><span>spurious</span></div>
    </div>
    <p class="gc-note">strict: P ${pct(sm.precision)} · R ${pct(sm.recall)} · F1 ${pct(sm.f1)}
      &nbsp;·&nbsp; soft = slot-wise semantic match with per-slot gates${bWarn}<br>
      Annotators: ${escapeHtml(annot)}. Extracted triples: ${m.n_extracted||0}.${hint}
      <button id="gold-corpus-btn" class="gc-corpus-btn">Σ corpus F1 — all ${escapeHtml(corpus.label)} docs</button></p>
    <div id="gold-corpus-out"></div>
    <div class="gc-panel">
      <div class="gc-panel-h"><span>Gold-standard KG (orange = ground truth)</span>
        <label class="gc-ovl"><input type="checkbox" id="gold-overlay-chk"> overlay match status on main KG</label></div>
      <div id="gold-cy" class="gold-cy"></div>
    </div>
    <div class="gc-tables">
      <div><h4 class="gc-h gc-match">✓ Matched${soft ? " (soft)" : ""} (${matched.length})</h4>
        <table class="gc-tbl"><thead><tr><th>subject</th><th>predicate</th><th>object</th><th></th></tr></thead>
        <tbody>${_gcRows(matched, "gc-match", simInfo)}</tbody></table></div>
      <div><h4 class="gc-h gc-missing">○ Missing in extraction (${missing.length})</h4>
        <table class="gc-tbl"><thead><tr><th>subject</th><th>predicate</th><th>object</th><th>why</th></tr></thead>
        <tbody>${_gcRows(missing, "gc-missing", reasonInfo)}</tbody></table></div>
      <div><h4 class="gc-h gc-spurious">✗ Spurious · not in gold (${spurious.length})</h4>
        <table class="gc-tbl"><thead><tr><th>subject</th><th>predicate</th><th>object</th><th>why</th></tr></thead>
        <tbody>${_gcRows(spurious, "gc-spurious", reasonInfo)}</tbody></table></div>
    </div>`;
  const idsFrom = soft || data;
  state.gold = { matched: new Set(idsFrom.matched_fact_ids || []),
                 spurious: new Set(idsFrom.spurious_fact_ids || []) };
  _renderGoldCy(data.gold_graph || { nodes: [], edges: [] });
  const _ovl = $("#gold-overlay-chk");
  if (_ovl) _ovl.addEventListener("change", () => _applyGoldOverlay(_ovl.checked));
  $("#gold-corpus-btn")?.addEventListener("click", _fetchGoldCorpus);
}

function _goldCorpusLabel() {
  const id = String(state.doc || "");
  const src = id.split("-")[0] || "";
  const split = id.split("-")[1] || "";
  const real = split && split !== "seed";
  return { src, split: real ? split : "", label: real ? `${src}-${split}` : src };
}

async function _fetchGoldCorpus() {
  const out = $("#gold-corpus-out");
  if (!out) return;
  const { src, split } = _goldCorpusLabel();
  out.innerHTML = "<em>evaluating corpus… (first run loads the encoder)</em>";
  let url = `/api/gold_eval?slim=1&source=${encodeURIComponent(src)}`;
  if (split) url += `&split=${encodeURIComponent(split)}`;   // variant added by shim
  try {
    const r = await fetch(url);
    if (!r.ok) {
      let msg = "HTTP " + r.status;
      try { msg = (await r.json()).message || msg; } catch {}
      out.innerHTML = `<p class="err">${escapeHtml(msg)}</p>`;
      return;
    }
    const d = await r.json();
    const rows = Object.entries(d.summary || {}).map(([ann, x]) =>
      `<tr><td>${escapeHtml(ann)}</td><td>${x.n_docs}</td>` +
      `<td>${x.strict_micro.f1.toFixed(3)}</td><td>${x.soft_micro.f1.toFixed(3)}</td>` +
      `<td>${x.soft_micro.precision.toFixed(3)}</td><td>${x.soft_micro.recall.toFixed(3)}</td></tr>`).join("");
    out.innerHTML = `
      <table class="gc-tbl gc-corpus"><thead><tr><th>annotator</th><th>docs</th>
        <th>strict µF1</th><th>soft µF1</th><th>soft µP</th><th>soft µR</th></tr></thead>
        <tbody>${rows || `<tr><td colspan="6" class="gc-empty">no scored docs (run extraction on this corpus first)</td></tr>`}</tbody></table>
      <p class="gc-note">version=${escapeHtml(d.version || "v1")} · backend=${escapeHtml(d.backend || "?")}
        · skipped ${(d.skipped || []).length} docs</p>`;
  } catch (e) {
    out.innerHTML = `<p class="err">${escapeHtml(e.message)}</p>`;
  }
}

function _renderGoldCy(goldGraph) {
  const cont = document.getElementById("gold-cy");
  if (!cont || typeof cytoscape !== "function") return;
  if (state.goldCy) { try { state.goldCy.destroy(); } catch (e) {} }
  state.goldCy = cytoscape({
    container: cont,
    elements: [...(goldGraph.nodes || []), ...(goldGraph.edges || [])],
    layout: _buildLayoutOpts(state.layout || "fcose", (goldGraph.nodes || []).length),
    minZoom: 0.2, maxZoom: 2.5, wheelSensitivity: 0.3,
    style: [
      { selector: "node", style: {
          "background-color": "#fff7ed", "border-color": "#fb923c", "border-width": 2,
          "label": (e) => truncate(e.data("label") || "", 22), "font-size": 11,
          "text-valign": "center", "text-halign": "center",
          "text-outline-color": "#fff", "text-outline-width": 2,
          "width": 34, "height": 34, "color": "#7c2d12" }},
      { selector: "edge", style: {
          "label": (e) => e.data("label") || "", "curve-style": "bezier",
          "target-arrow-shape": "triangle", "line-color": "#fb923c",
          "target-arrow-color": "#fb923c", "width": 2, "font-size": 9,
          "color": "#9a3412", "text-rotation": "autorotate",
          "text-outline-color": "#fff", "text-outline-width": 1 }},
    ],
  });
  setTimeout(() => { try { state.goldCy.resize(); state.goldCy.fit(undefined, 20); } catch (e) {} }, 60);
}

function _applyGoldOverlay(on) {
  if (!state.cy || !state.gold) return;
  state.cy.edges().removeClass("gold-match gold-spurious");
  if (!on) return;
  const M = state.gold.matched, S = state.gold.spurious;
  state.cy.edges().forEach((e) => {
    const fids = e.data("fact_ids") || [];
    if (fids.some((f) => M.has(f))) e.addClass("gold-match");
    else if (fids.some((f) => S.has(f))) e.addClass("gold-spurious");
  });
}

// ============================================================
// Compare modal — v1 vs v2 distribution shift
// ============================================================

async function openCompareModal() {
  const body = $("#compare-body");
  body.innerHTML = "<em>loading…</em>";
  $("#compare-overlay").classList.remove("hidden");
  try {
    const data = await fetch(`/api/distribution_shift/${state.doc}`).then(r => r.json());
    body.innerHTML = "";
    if (data.note) {
      body.innerHTML = `<p>${escapeHtml(data.note)}</p>`;
      return;
    }
    const max = Math.max(1, ...data.v1, ...data.v2);
    const grid = document.createElement("div");
    grid.className = "compare-bars";
    for (let i = 0; i < data.labels.length; i++) {
      const lbl = data.labels[i], v1 = data.v1[i], v2 = data.v2[i], dpct = data.delta_pct[i];
      grid.innerHTML += `
        <div class="lab"><span class="lbl ${lbl}" style="padding:1px 6px;border-radius:3px;">${lbl}</span></div>
        <div class="bar-track" title="v1=${v1}  v2=${v2}">
          <div class="bar-v1" style="width:${(v1/max*100).toFixed(1)}%"></div>
          <div class="bar-v2" style="width:${(v2/max*100).toFixed(1)}%"></div>
        </div>
        <div class="pct ${dpct == null ? '' : dpct < 0 ? 'dn' : 'up'}">${dpct == null ? '—' : (dpct>0?'+':'')+dpct+'%'}</div>`;
    }
    body.appendChild(grid);

    const summary = document.createElement("div");
    summary.className = "compare-summary";
    summary.innerHTML = `
      <div class="row"><b>Versions:</b> ${escapeHtml(data.v1_label)} → ${escapeHtml(data.v2_label || "—")}</div>
      <div class="row"><b>Total conflicts:</b> ${data.totals.v1} → ${data.totals.v2}
        (${data.totals.delta_pct == null ? '—' : (data.totals.delta_pct>0?'+':'')+data.totals.delta_pct+'%'})</div>
      <div class="row"><b>Layer-1 filter rate:</b>
        ${(data.layer1_rate.v1.layer1_filter_rate*100).toFixed(1)}%
        → ${(data.layer1_rate.v2.layer1_filter_rate*100).toFixed(1)}%</div>`;
    body.appendChild(summary);
  } catch (e) {
    body.innerHTML = `<em>Failed to load compare data: ${escapeHtml(e.message)}</em>`;
  }
}

// ============================================================
// Guideline editor + job poll
// ============================================================

async function saveGuideline() {
  const text = $("#guideline-text").value.trim();
  const models = $("#guideline-models").value.trim().split(",").map(s => s.trim()).filter(Boolean);
  const status = $("#guideline-status");
  status.classList.remove("err");
  if (!text) { status.textContent = "guideline text is empty"; status.classList.add("err"); return; }
  status.textContent = "enqueueing…";
  try {
    const resp = await fetch("/api/guideline", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ guideline_text: text, models }),
    });
    if (!resp.ok) {
      status.textContent = `error ${resp.status}`;
      status.classList.add("err");
      return;
    }
    const job = await resp.json();
    status.textContent = `enqueued: ${job.job_id} (${job.progress.total} cells)`;
    pollJob(job.job_id);
  } catch (e) {
    status.textContent = "request failed";
    status.classList.add("err");
  }
}

async function pollJob(jobId) {
  const node = $("#job-status");
  const line = document.createElement("div");
  line.className = "job-line";
  line.dataset.jobId = jobId;
  node.prepend(line);

  while (true) {
    let job;
    try {
      job = await fetch(`/api/jobs/${jobId}`).then(r => r.json());
    } catch (e) { break; }
    line.className = "job-line " + job.status;
    line.textContent = `[${jobId}] ${job.status} ${job.progress.done}/${job.progress.total}` +
                       (job.error ? ` — ${job.error}` : "");
    if (job.status === "done" || job.status === "failed") break;
    await new Promise(r => setTimeout(r, 2000));
  }
}

// ============================================================
// Cross-view selection
// ============================================================

// Restore the plain text of a mark element (the SPO-tinted children get
// removed) so a fresh selection paints cleanly on top of the same span.
function _restoreMarkText(m) {
  if (!m) return;
  const t = m.dataset.fullText;
  if (t != null) { m.textContent = t; }
}

// Paint subject/predicate/object substrings inside the mark with the SPO
// palette. We split the mark's textNode into up to 5 nodes (fill | S | gap
// | P | gap | O | fill). When a band can't be located we silently skip it
// — that fact's S/P/O didn't appear verbatim in the source quote.
function _paintSpoInMark(m, fact) {
  if (!m || !fact) return;
  if (m.dataset.fullText == null) m.dataset.fullText = m.textContent;
  const full = m.dataset.fullText;
  const bands = spoBands(full, fact.subject, fact.predicate, fact.object);
  if (bands.length === 0) return; // nothing to split — leave plain mark.

  m.textContent = "";
  let cursor = 0;
  for (const b of bands) {
    if (b.start < cursor) continue; // overlap — skip to keep sequence sane.
    if (b.start > cursor) m.appendChild(document.createTextNode(full.slice(cursor, b.start)));
    const span = document.createElement("span");
    // Fuzzy-located bands get a dashed underline: the S/P/O text is NOT a
    // verbatim substring here (decontextualized or mis-copied by the model),
    // so the band marks the closest supporting region, not an exact quote.
    span.className = `spo-${b.role}` + (b.fuzzy ? " spo-fuzzy" : "");
    span.title = b.fuzzy ? `approximate match for ${b.role}` : "";
    span.textContent = full.slice(b.start, b.end);
    m.appendChild(span);
    cursor = b.end;
  }
  if (cursor < full.length) m.appendChild(document.createTextNode(full.slice(cursor)));
}

function _flash(el) {
  if (!el) return;
  el.classList.remove("spo-flash");
  // Force reflow so the animation restarts on repeat selections.
  // eslint-disable-next-line no-unused-expressions
  el.offsetWidth;
  el.classList.add("spo-flash");
  setTimeout(() => el.classList.remove("spo-flash"), 800);
}

function selectFact(factId) {
  const prevId = state.selectedFactId;
  state.selectedFactId = factId;
  const fact = state.factsById.get(factId);

  // ---- Text pane: restore previous SPO mark, paint new one, scroll+flash.
  for (const m of document.querySelectorAll("#doc-body mark.linked, #doc-body mark.spo-active")) {
    m.classList.remove("linked");
    m.classList.remove("spo-active");
    _restoreMarkText(m);
  }
  const m = state.factSpans.get(factId);
  if (m && fact) {
    m.classList.add("linked");
    m.classList.add("spo-active");
    _paintSpoInMark(m, fact);
    m.scrollIntoView({ behavior: "smooth", block: "center" });
    _flash(m);
  }

  // ---- Facts table: scroll to row + tint S/P/O cells.
  const container = $("#facts-table");
  for (const item of container.querySelectorAll(".fact-item")) {
    item.classList.remove("selected", "spo-active");
  }
  const row = container.querySelector(`.fact-item[data-fact-id="${factId}"]`);
  if (row) {
    row.classList.add("selected");
    row.classList.add("spo-active");
    row.scrollIntoView({ behavior: "smooth", block: "center" });
    _flash(row);
  }

  // ---- KG: clear prior SPO, paint subject/object nodes + predicate edge,
  //       and animate-zoom to the connected nodes.
  if (state.cy) {
    state.cy.elements().removeClass("selected spo-subj spo-obj spo-pred stmt-sel");
    if (state.graph && state.graph.reified) {
      // Reified KG: light up the STATEMENT node(s) carrying this fact, plus its
      // subject (blue) and object (red) entities. One fact may merge into one
      // statement, but several statements can share an entity.
      const stmts = state.cy.nodes('[type = "statement"]')
        .filter(n => (n.data("fact_ids") || []).includes(factId));
      if (stmts.length) {
        stmts.addClass("stmt-sel selected");
        const conn = stmts.connectedEdges();
        conn.addClass("selected");
        conn.forEach(e => {
          const role = e.data("role");
          if (role === "subject") e.source().addClass("spo-subj");
          else if (role === "object") e.target().addClass("spo-obj");
        });
        state.cy.animate(
          { fit: { eles: stmts.closedNeighborhood(), padding: 60 } },
          { duration: 400 });
      }
    } else {
      const matchingEdges = state.cy.edges().filter(e => (e.data("fact_ids") || []).includes(factId));
      if (matchingEdges.length) {
        const e0 = matchingEdges[0];
        e0.addClass("spo-pred");
        e0.source().addClass("spo-subj");
        e0.target().addClass("spo-obj");
        matchingEdges.addClass("selected");
        state.cy.animate(
          { fit: { eles: matchingEdges.connectedNodes(), padding: 60 } },
          { duration: 400 });
      }
    }
  }
}

// ============================================================
// Keyboard navigation
// ============================================================

function onKeyDown(e) {
  if (e.key === "Escape") { closeAllModals(); return; }
  // Don't intercept when typing in an input or textarea
  const tag = (e.target.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea") return;

  if (e.key === "j" || e.key === "k") {
    const facts = visibleFacts();
    if (!facts.length) return;
    let idx = facts.findIndex(f => f.fact_id === state.selectedFactId);
    if (idx < 0) idx = e.key === "j" ? -1 : facts.length;
    const next = e.key === "j" ? (idx + 1) % facts.length : (idx - 1 + facts.length) % facts.length;
    selectFact(facts[next].fact_id);
    e.preventDefault();
  }
}

// ============================================================
// Export
// ============================================================

function exportFacts(format) {
  const rows = visibleFacts();
  let blob, ext;
  if (format === "json") {
    blob = new Blob([JSON.stringify(rows, null, 2)], { type: "application/json" });
    ext = "json";
  } else {
    const headers = ["fact_id", "annotator", "subject", "predicate", "object", "natural_language",
                     "section_path", "char_start", "char_end", "conflict_label", "verification"];
    const lines = [headers.join(",")];
    for (const f of rows) {
      const loc = f.source_locator || {};
      const cells = [
        f.fact_id, f._annotator, f.subject, f.predicate, f.object,
        f.natural_language, loc.section_path,
        loc.char_start, loc.char_end,
        f._edge_conflict_label, f._verification?.status || "",
      ];
      lines.push(cells.map(csvCell).join(","));
    }
    blob = new Blob([lines.join("\n")], { type: "text/csv" });
    ext = "csv";
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `facts_${state.doc}_visible.${ext}`;
  a.click();
}

function csvCell(v) {
  if (v == null) return "";
  const s = String(v);
  if (/[,"\n]/.test(s)) return '"' + s.replaceAll('"', '""') + '"';
  return s;
}

// ============================================================
// Helpers
// ============================================================

function truncate(s, n) { return s && s.length > n ? s.slice(0, n) + "…" : s; }
function escapeHtml(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}

// Expose for cross-IIFE access (initPhase5 needs to trigger re-render)
window._refreshFacts = () => refreshFacts();
// Shared surface for the Phase-5e IIFE (BUG-2026-07-02-E): its legend helpers
// must drive the SAME selection state + re-renders as the topbar. Bare `state`
// or `renderChips` references from other IIFEs are ReferenceErrors.
window.__va = {
  get state() { return state; },
  renderChips: () => renderChips(),
  refreshText: () => refreshText(),
  refreshFacts: () => refreshFacts(),
  refreshGraphHighlights: () => refreshGraphHighlights(),
};

boot();
})();

/* ===========================================================
 * Phase-5 additions: append-only block. Reaches into the IIFE
 * above through the global `window` shim we install below so the
 * additions can stay self-contained without re-writing the file.
 * =========================================================== */

(() => {
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);
const escapeHtml = (s) => String(s ?? "")
  .replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")
  .replaceAll('"',"&quot;").replaceAll("'","&#39;");

// State global to Phase-5 features. Independent of the IIFE state above.
const ext = {
  guidelines: [],         // [{version, summary, size}]
  models: (window.__vaRecord?.annotators || []).slice(),
  matrixCols: [],         // [{model, guideline}]
  docs: [],
  matrixStatus: {},       // key="doc|model" -> {fact_count, guideline_version}
  spanSelection: null,    // {start, end, text}
};

document.addEventListener("DOMContentLoaded", initPhase5);
// Also fire now in case DOM is already ready (script is at end of body).
if (document.readyState !== "loading") initPhase5();

/** Switch the bottom drawer to a tab, by key. Exported because two other
 *  modules need to bring a tab forward (a citation landing in the change log,
 *  the workflow panel's hand-offs) and none of them should re-implement the
 *  class toggling. */
window.goToTab = function goToTab(key) {
  document.querySelector(`#drawer-tabs .tab[data-tab="${key}"]`)?.click();
};

let _initialised = false;
function initPhase5() {
  if (_initialised) return;
  _initialised = true;

  // Drawer tab switching
  for (const tab of $$("#drawer-tabs .tab")) {
    tab.addEventListener("click", () => {
      const key = tab.dataset.tab;
      $$("#drawer-tabs .tab").forEach(t => t.classList.toggle("active", t === tab));
      $$(".tab-pane").forEach(p => p.classList.toggle("active", p.dataset.tab === key));
      if (key === "run-matrix") renderRunMatrix();
      if (key === "annotation") window._svLoad?.();
      if (key === "arbitrate-manager") window._amLoad?.();
      // the guideline editor moved in with its change log, so the version list
      // has to load with that tab, not with the schema frame's
      if (key === "change-log") { loadGuidelinesList(); window.GLog?.reload(); }
    });
  }

  // "…is its own Change log tab" / "selected in Annotation": a sentence that
  // names another tab should take you there, or it is just a place name.
  for (const link of $$("[data-goto-tab]")) {
    link.addEventListener("click", () =>
      window.goToTab?.(link.dataset.gotoTab));
  }

  // Upload modal wiring
  $("#upload-btn")?.addEventListener("click", openUploadModal);
  $("#upload-close")?.addEventListener("click", closeUploadModal);
  $("#upload-overlay")?.addEventListener("click", (e) => {
    if (e.target.id === "upload-overlay") closeUploadModal();
  });
  $("#upload-submit")?.addEventListener("click", submitUpload);

  // Span re-extract floater
  document.addEventListener("mouseup", maybeShowSpanMenu);
  $("#span-cancel")?.addEventListener("click", hideSpanMenu);
  $("#span-reextract")?.addEventListener("click", submitSpanReextract);

  // Facts sort dropdown
  document.getElementById("facts-sort")?.addEventListener("change", () => window._refreshFacts?.());

  // Guideline manager (Annotation tab — right panel)
  $("#gm-load")?.addEventListener("click", gmLoad);
  $("#gm-save")?.addEventListener("click", gmSave);
  $("#gm-delete")?.addEventListener("click", gmDelete);
  $("#gm-freeze")?.addEventListener("click", gmFreeze);

  // Run matrix
  $("#rm-add-combo")?.addEventListener("click", addMatrixCombo);
  $("#rm-submit")?.addEventListener("click", submitMatrix);

  // Pre-fill guidelines list (need it for span menu + upload modal too)
  refreshGuidelinesIntoSelects();

  // Model pickers (Local vs Cloud) + fix the model-checks -> hidden sync
  // (the static markup never wired it, so toggling boxes did nothing).
  wireModelChecks();
  refreshModelsIntoSelects();
}

// ----------------------------------------------------------
// Upload modal
// ----------------------------------------------------------
function openUploadModal() {
  // refresh guideline + model selects
  refreshGuidelinesIntoSelects();
  $("#upload-overlay")?.classList.remove("hidden");
  $("#upload-title").value = "";
  $("#upload-textarea").value = "";
  $("#upload-status").textContent = "";
  $("#upload-status").className = "";
}
function closeUploadModal() { $("#upload-overlay")?.classList.add("hidden"); }

async function submitUpload() {
  const title = $("#upload-title").value.trim();
  const text = $("#upload-textarea").value;
  const split_mode = $("#upload-split").value;
  const extract = $("#upload-extract").checked;
  const model = $("#upload-model").value;
  const guideline_version = $("#upload-guideline").value;
  const status = $("#upload-status");
  if (!text.trim()) { status.textContent = "text is empty"; status.className = "err"; return; }
  status.textContent = "uploading…"; status.className = "";
  try {
    const r = await fetch("/api/upload_text", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ title, text, split_mode, extract, model, guideline_version }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    status.textContent = `uploaded: ${data.doc_id} (${data.n_sections} sections)` +
                         (data.job_id ? ` · extraction job ${data.job_id}` : "");
    status.className = "ok";
    // Refresh the doc dropdown — re-fetch the list and switch to the new doc
    state.docs = (await fetch("/api/docs").then(r => r.json())).docs;
    state.sourceFilter = "all";
    const srcSel = $("#source-filter"); if (srcSel) srcSel.value = "all";
    const sel = $("#doc-select");
    populateDocSelect({ select: data.doc_id });
    sel.value = data.doc_id;
    sel.dispatchEvent(new Event("change"));
    setTimeout(closeUploadModal, 600);
  } catch (e) {
    status.textContent = `failed: ${e.message}`; status.className = "err";
  }
}

// ----------------------------------------------------------
// Span re-extract
// ----------------------------------------------------------
function _getDocBodySelection() {
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed) return null;
  const body = $("#doc-body");
  if (!body) return null;
  const range = sel.getRangeAt(0);
  if (!body.contains(range.commonAncestorContainer)) return null;
  // Compute char offsets relative to the whole doc-body innerText.
  const pre = document.createRange();
  pre.selectNodeContents(body);
  pre.setEnd(range.startContainer, range.startOffset);
  const start = pre.toString().length;
  const end = start + range.toString().length;
  const text = range.toString();
  if (text.trim().length < 5) return null;
  return { start, end, text, rect: range.getBoundingClientRect() };
}

function maybeShowSpanMenu(e) {
  // Don't trigger when clicking inside the menu itself.
  if (e.target.closest("#span-menu")) return;
  const sel = _getDocBodySelection();
  if (!sel) { hideSpanMenu(); return; }
  ext.spanSelection = sel;
  // Populate model + guideline selects
  refreshGuidelinesIntoSelects();
  const menu = $("#span-menu");
  menu.classList.remove("hidden");
  const paneRect = $("#pane-text").getBoundingClientRect();
  // Position just below the selection
  menu.style.left = `${Math.max(8, sel.rect.left - paneRect.left)}px`;
  menu.style.top = `${sel.rect.bottom - paneRect.top + 4}px`;
}
function hideSpanMenu() {
  ext.spanSelection = null;
  $("#span-menu")?.classList.add("hidden");
}

async function submitSpanReextract() {
  if (!ext.spanSelection) return;
  const docSelect = $("#doc-select");
  const doc_id = docSelect?.value;
  if (!doc_id) return;
  const model = $("#span-model").value;
  const guideline_version = $("#span-guideline").value;
  // Note: char_start/end here are offsets into the rendered text — for user
  // docs (which have only preamble_text), this is correct. For EUR-Lex docs
  // the section_path form is safer; the backend supports both.
  const body = {
    doc_id, model, guideline_version,
    char_start: ext.spanSelection.start,
    char_end: ext.spanSelection.end,
  };
  try {
    const r = await fetch("/api/reextract_span", {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const t = await r.text();
      alert(`span re-extract failed: ${r.status} ${t.slice(0,200)}`);
      return;
    }
    const data = await r.json();
    hideSpanMenu();
    // Reload the doc to surface the new facts
    docSelect.dispatchEvent(new Event("change"));
    setTimeout(() => alert(`Span re-extracted: ${data.n_new_facts} triple(s) for ${data.model} written to ${data.out_path}.`), 200);
  } catch (e) {
    alert(`span re-extract error: ${e.message}`);
  }
}

// ----------------------------------------------------------
// Guideline manager
// ----------------------------------------------------------
async function loadGuidelinesList() {
  try {
    const data = await fetch("/api/guidelines").then(r => r.json());
    ext.guidelines = data.guidelines || [];
    const sel = $("#gm-select");
    sel.innerHTML = "";
    for (const g of ext.guidelines) {
      const opt = document.createElement("option");
      opt.value = g.version;
      // a frozen arm is listed (it must stay runnable) but marked, so its
      // text is not something you reach for while authoring the next version
      opt.textContent = `${g.version} (${g.size}b)` + (g.locked ? " — frozen" : "");
      sel.appendChild(opt);
    }
    refreshGuidelinesIntoSelects();
  } catch (e) {
    $("#gm-status").textContent = `list failed: ${e.message}`;
    $("#gm-status").className = "err";
  }
}

/** Commit a guideline version: the point after which it is the artefact.
 *
 * The lock carries the sha256 of the text AND the change log's digest, because
 * freezing the guideline while leaving its record editable would freeze the
 * artefact and not the account of it — and RQ2 reports the account. Same lock
 * shape as the control console's, so the two arms are frozen the same way.
 */
async function gmFreeze() {
  const version = ($("#gm-new-version").value || "").trim()
                  || $("#gm-select")?.value || "";
  const status = $("#gm-status");
  if (!version) { status.textContent = "pick a version first"; status.className = "err"; return; }
  const ed = $("#gm-text");
  if (ed && window.__gmSaved !== undefined && ed.value !== window.__gmSaved) {
    status.textContent = "the editor has unsaved edits — save or log them first";
    status.className = "err";
    return;
  }
  if (!confirm(`Freeze extract_${version}.md?

`
             + "This writes the lock (sha256 + the change log's digest) and "
             + "refuses every later edit to the guideline AND to its reasons. "
             + "A committed arm is superseded by a new version, not re-opened.")) return;
  status.textContent = "freezing…";
  const r = await fetch(`/api/guideline_freeze/${encodeURIComponent(version)}`,
                        {method: "POST"});
  const d = await r.json().catch(() => ({}));
  if (!r.ok) {
    status.textContent = "freeze refused: " + (d.message || d.description || r.status);
    status.className = "err";
    return;
  }
  status.textContent = `frozen ${d.lock.committed_at}`;
  status.className = "ok";
  renderGuidelineLock(d.lock);
  loadGuidelinesList();
  window.GLog?.reload();
}

function renderGuidelineLock(lock) {
  const box = $("#gm-lock");
  if (!box) return;
  box.classList.toggle("hidden", !lock);
  if (!lock) return;
  box.innerHTML = `<b>FROZEN</b> ${escapeHtml(lock.committed_at)} · sha256 `
    + `<code>${escapeHtml(lock.sha256.slice(0, 12))}</code> · ${lock.chars} chars / `
    + `${lock.words} words · change log: ${lock.entries} entries, ${lock.cites} `
    + `citations, ${lock.shots} screenshots (digest `
    + `<code>${escapeHtml((lock.change_log || {}).sha256 ? lock.change_log.sha256.slice(0, 8) : "?")}</code>)`;
}

async function gmLoad() {
  const version = $("#gm-select").value;
  if (!version) return;
  try {
    const r = await fetch(`/api/guidelines/${encodeURIComponent(version)}`);
    const data = await r.json();
    if (r.status === 423) {          // committed arm of the experiment
      $("#gm-status").textContent = data.message || "this version is frozen";
      $("#gm-status").className = "err";
      return;
    }
    if (!r.ok) throw new Error(data.message || r.status);
    $("#gm-text").value = data.text || "";
    // what is on disk for this version, so the change log can tell whether the
    // editor holds unsaved edits when a change is logged
    window.__gmSaved = data.text || "";
    $("#gm-new-version").value = version;
    // tell the change log which version it is now writing to — it reads these
    // two controls, and nothing else here announces that they moved
    $("#gm-new-version").dispatchEvent(new Event("change"));
    fetch(`/api/guideline_freeze/${encodeURIComponent(version)}`)
      .then((r) => r.json())
      .then((d) => renderGuidelineLock(d.frozen ? d.lock : null))
      .catch(() => renderGuidelineLock(null));
    $("#gm-status").textContent = `loaded ${version} (${data.size}b)`;
    $("#gm-status").className = "ok";
  } catch (e) {
    $("#gm-status").textContent = `load failed: ${e.message}`;
    $("#gm-status").className = "err";
  }
}

async function gmSave() {
  const version = ($("#gm-new-version").value || "").trim();
  const text = $("#gm-text").value;
  const status = $("#gm-status");
  if (!version) { status.textContent = "new version is required"; status.className = "err"; return; }
  if (!/^[A-Za-z0-9_.\-]+$/.test(version)) {
    status.textContent = "version must be alphanumeric/dot/dash/underscore"; status.className = "err"; return;
  }
  try {
    const r = await fetch(`/api/guidelines/${encodeURIComponent(version)}`, {
      method: "PUT", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ text }),
    });
    if (!r.ok) {
      const t = await r.text();
      status.textContent = `save failed: ${r.status} ${t.slice(0,160)}`;
      status.className = "err";
      return;
    }
    const data = await r.json();
    window.__gmSaved = text;
    status.textContent = `saved ${data.version} (${data.size}b) — now editing ${data.version}`;
    status.className = "ok";
    await loadGuidelinesList();
    // Land on the version just written. Leaving "Version" on v1 with the new
    // name only in "Save as" made the editor claim one version while the change
    // log wrote to another — and clearing the "Save as" box would then have
    // silently pointed the log back at the baseline. One field answers "which
    // version am I on" once the save has happened.
    const sel = $("#gm-select");
    if (sel && [...sel.options].some((o) => o.value === data.version)) {
      sel.value = data.version;
      $("#gm-new-version").value = "";
      sel.dispatchEvent(new Event("change"));
    }
  } catch (e) {
    status.textContent = `save failed: ${e.message}`;
    status.className = "err";
  }
}

async function gmDelete() {
  const version = $("#gm-select")?.value || "";
  if (!version || version === "v1") { alert("v1 is the baseline guideline and cannot be deleted."); return; }
  if (!confirm(`Delete extract_${version}.md? This cannot be undone.`)) return;
  try {
    const r = await fetch(`/api/guidelines/${encodeURIComponent(version)}`,
      { method: "DELETE" });
    if (!r.ok) { _setInlineStatus("#gm-status", "delete failed: " + r.status, "err"); return; }
    await loadGuidelinesList();
    const ta = $("#gm-text"); if (ta) ta.value = "";
    _setInlineStatus("#gm-status", `deleted extract_${version}.md`, "ok");
  } catch(e) { _setInlineStatus("#gm-status", "error: " + e.message, "err"); }
}

// ----------------------------------------------------------
// Run matrix
// ----------------------------------------------------------
function _rmSelectedModels() {
  return [...$$('.rm-model-checks input[type="checkbox"]:checked')].map(el => el.value);
}

async function renderRunMatrix() {
  // Fetch docs, guidelines, matrix status
  const [docs, gl, status] = await Promise.all([
    fetch("/api/docs").then(r => r.json()),
    fetch("/api/guidelines").then(r => r.json()),
    fetch("/api/run_matrix/status").then(r => r.json()),
  ]);
  ext.docs = docs.docs || [];
  ext.guidelines = gl.guidelines || [];
  ext.matrixStatus = status || {};
  if (!ext.matrixCols.length) {
    // Default columns: every checked model × every guideline.
    const models = _rmSelectedModels();
    ext.matrixCols = [];
    for (const m of models) for (const g of ext.guidelines) {
      ext.matrixCols.push({ model: m, guideline: g.version });
    }
  }
  drawMatrix();
}

function drawMatrix() {
  const wrap = $("#rm-matrix-wrap");
  wrap.innerHTML = "";
  if (!ext.matrixCols.length || !ext.docs.length) {
    wrap.innerHTML = "<p class='hint'>No docs or columns yet. Add models above and click 'Add'.</p>";
    return;
  }
  const tbl = document.createElement("table");
  tbl.className = "rm-matrix";
  const thead = document.createElement("thead");
  let h = "<tr><th>doc</th>";
  for (const c of ext.matrixCols) h += `<th title="${escapeHtml(c.model)} · ${escapeHtml(c.guideline)}">${escapeHtml(c.model.slice(0,12))}<br><small>${escapeHtml(c.guideline)}</small></th>`;
  h += "</tr>";
  thead.innerHTML = h;
  tbl.appendChild(thead);
  const tbody = document.createElement("tbody");
  for (const d of ext.docs) {
    const tr = document.createElement("tr");
    // An eval document is listed but cannot be picked: the server refuses it
    // anyway (invariant 1), and a checkbox that always fails is a trap.
    const locked = d.split === "eval";
    tr.innerHTML = `<td class="doc-id">${locked ? "🔒 " : d.is_user_doc ? "📝 " : ""}${escapeHtml(d.doc_id)}</td>`;
    for (const c of ext.matrixCols) {
      const key = `${d.doc_id}|${c.model}`;
      const cur = ext.matrixStatus[key];
      const done = cur && cur.guideline_version === c.guideline;
      const title = locked ? "locked eval document — extracted only by the batch run of record"
        : cur ? `existing: ${cur.fact_count} triples from ${cur.guideline_version}` : "no prior run";
      tr.innerHTML += `<td class="cell${done ? " done" : ""}" title="${escapeHtml(title)}">
        <input type="checkbox" data-doc="${escapeHtml(d.doc_id)}"${locked ? " disabled" : ""}
               data-model="${escapeHtml(c.model)}" data-guideline="${escapeHtml(c.guideline)}"></td>`;
    }
    tbody.appendChild(tr);
  }
  tbl.appendChild(tbody);
  wrap.appendChild(tbl);
}

function addMatrixCombo() {
  const models = _rmSelectedModels();
  ext.matrixCols = [];
  for (const m of models) for (const g of ext.guidelines) {
    ext.matrixCols.push({ model: m, guideline: g.version });
  }
  drawMatrix();
}

async function submitMatrix() {
  const phase = (document.querySelector('input[name="rm-phase"]:checked')?.value) || "phase1";
  const checks = $$('#rm-matrix-wrap input[type="checkbox"]:checked');
  const cells = [...checks].map(el => ({
    doc_id: el.dataset.doc,
    model: el.dataset.model,
    guideline_version: el.dataset.guideline,
  }));
  const status = $("#rm-status");
  if (!cells.length) { status.textContent = "select at least one cell"; status.className = "err"; return; }
  status.textContent = `queueing ${cells.length} cells (${phase})…`; status.className = "";
  try {
    if (phase === "phase2") {
      // Phase-2: group by (doc_id × guideline_version), one job per unique pair
      const pairs = [...new Map(cells.map(c => [`${c.doc_id}|${c.guideline_version}`, c])).values()];
      // The Runs tab's Layer-2 settings, not a hard-coded stub: every matrix
      // Phase-2 used to run with skip_layer2 (BUG-2026-09-23-G).
      const l2 = _phase2Settings();
      let ok = 0, err = 0, why = "";
      for (const c of pairs) {
        const r = await fetch("/api/run_phase2", {
          method: "POST", headers: {"Content-Type":"application/json"},
          body: JSON.stringify({ version: c.guideline_version, doc_ids: [c.doc_id], ...l2 }),
        });
        if (r.ok) ok++; else { err++; why = why || await _errMessage(r); }
      }
      status.textContent = `queued ${ok} Phase-2 jobs (${err} refused${why ? ": " + why : ""})`;
      status.className = ok ? "ok" : "err";
    } else {
      const r = await fetch("/api/run_matrix", {
        method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ cells }),
      });
      if (!r.ok) { status.textContent = `refused (${r.status}): ${await _errMessage(r)}`; status.className = "err"; return; }
      const data = await r.json();
      const ok = data.jobs.filter(j => j.job_id).length;
      const err = data.jobs.length - ok;
      status.textContent = `queued ${ok} Phase-1 jobs (${err} skipped/errored)`;
      status.className = ok ? "ok" : "err";
    }
  } catch (e) {
    status.textContent = `failed: ${e.message}`; status.className = "err";
  }
}

// ----------------------------------------------------------
// Shared: populate model + guideline selects everywhere
// ----------------------------------------------------------
async function refreshGuidelinesIntoSelects() {
  let glList = ext.guidelines;
  if (!glList.length) {
    try {
      const data = await fetch("/api/guidelines").then(r => r.json());
      glList = data.guidelines || [];
      ext.guidelines = glList;
    } catch { glList = []; }
  }
  const guidelineSelects = [$("#span-guideline"), $("#upload-guideline")];
  for (const s of guidelineSelects) {
    if (!s) continue;
    const prev = s.value;
    s.innerHTML = "";
    for (const g of glList) {
      const opt = document.createElement("option");
      opt.value = g.version; opt.textContent = g.version;
      s.appendChild(opt);
    }
    if (prev && [...s.options].some(o => o.value === prev)) s.value = prev;
  }
  const groups = await _fetchModelGroups();
  _populateModelDropdowns(groups);
}

// ----------------------------------------------------------
// Model pickers: Local vs Cloud (Phase-1 / Phase-2).
// Also fixes the .model-checks -> hidden-input sync that the
// static markup never had wired (toggling boxes did nothing).
// ----------------------------------------------------------
const FALLBACK_LOCAL = ["qwen3.5:4b", "gemma3:4b", "nemotron-3-nano:4b"];
// Defaults come from the server (window.__vaRecord, see index.html): the
// frozen annotators and the arbitrator of record. These used to be the retired
// local 4B models, with every Phase-1 picker defaulting to "all local models"
// (BUG-2026-09-23-G).
const RECORD = window.__vaRecord || { annotators: [], layer2: "" };
const PREFERRED_LAYER2 = RECORD.layer2;

/** The server's reason for a refusal (403 eval set, 409 run of record). */
async function _errMessage(r) {
  try { const d = await r.json(); return d.message || d.error || `HTTP ${r.status}`; }
  catch { return `HTTP ${r.status}`; }
}

/** The Runs tab's Layer-2 settings, shared by every Phase-2 it can start. */
function _phase2Settings() {
  const out = {
    skip_layer2: !!$("#p2-skip-layer2")?.checked,
    arbitrate_version: $("#p2-arbitrate-version")?.value || "v1",
  };
  const m = $("#p2-layer2-model")?.value || PREFERRED_LAYER2;
  if (m) out.layer2_model = m;
  return out;
}
window.__vaPhase2Settings = _phase2Settings;   // for "Build KG" in the graph IIFE
let _modelGroupsCache = null;
let _checksBuilt = false;

function _syncModelHidden(box) {
  const hidden = document.getElementById(box.dataset.target);
  if (!hidden) return;
  const single = box.classList.contains("single");
  const vals = [...box.querySelectorAll('input[type="checkbox"]:checked')].map(c => c.value);
  hidden.value = single ? (vals[0] || "") : vals.join(",");
}

function wireModelChecks() {
  for (const box of $$('.model-checks[data-target]')) {
    if (box.dataset.wired) { _syncModelHidden(box); continue; }
    box.dataset.wired = "1";
    const single = box.classList.contains("single");
    // Delegated listener on the box survives innerHTML rebuilds.
    box.addEventListener("change", (e) => {
      if (single && e.target.checked) {
        for (const c of box.querySelectorAll('input[type="checkbox"]')) {
          if (c !== e.target) c.checked = false;
        }
      }
      _syncModelHidden(box);
    });
    _syncModelHidden(box);
  }
}

// The model list and the guideline list live in this IIFE; the facts pane's
// per-segment annotate bar is in the one above it. Exported rather than
// re-fetched there, so there is one answer to "which annotators exist" and one
// cache behind it.
window.__vaModels = () => _fetchModelGroups();
window.__vaGuidelines = () => ext.guidelines || [];

async function _fetchModelGroups() {
  if (_modelGroupsCache) return _modelGroupsCache;
  try {
    const g = await fetch("/api/models").then(r => r.json());
    _modelGroupsCache = { local: g.local || [], cloud: g.cloud || [] };
  } catch {
    _modelGroupsCache = {
      local: FALLBACK_LOCAL.map(m => ({ id: m, tag: m, provider: "local" })),
      cloud: [],
    };
  }
  // The models of record are always offered, even when the box's model list
  // did not name them: a default that silently falls back to another model is
  // worse than a run that fails loudly because the cloud is not configured.
  const have = new Set([..._modelGroupsCache.local, ..._modelGroupsCache.cloud].map(m => m.id));
  for (const id of [...RECORD.annotators, RECORD.layer2]) {
    if (id && !have.has(id)) {
      _modelGroupsCache.cloud.push({ id, tag: id.replace(/^cloud:/, ""), provider: "cloud" });
      have.add(id);
    }
  }
  return _modelGroupsCache;
}

function _checkHtml(entry, checked) {
  const cloud = entry.provider === "cloud";
  return `<label class="${cloud ? "is-cloud" : "is-local"}" title="${escapeHtml(entry.id)}">` +
         `<input type="checkbox" value="${escapeHtml(entry.id)}"${checked ? " checked" : ""}> ` +
         `${escapeHtml(entry.tag)}</label>`;
}

function _ddGroups(groups) {
  return [
    { label: "Local", items: (groups.local || []).map(m => ({ value: m.id, label: m.tag })) },
    { label: "Cloud", items: (groups.cloud || []).map(m => ({ value: m.id, label: m.tag })) },
  ];
}

// Phase-1 = multi-select dropdown (keeps batch) -> hidden #p1-models comma list.
// Phase-2 Layer-2 = native single <select> with Local/Cloud optgroups.
function _buildPhaseModelPickers(groups) {
  const allIds = new Set([...(groups.local || []), ...(groups.cloud || [])].map(m => m.id));
  const p1host = document.getElementById("p1-models-dd");
  const p1hidden = document.getElementById("p1-models");
  if (p1host && p1hidden && window.renderMultiDropdown) {
    const sel = new Set((p1hidden.value || "").split(",").map(s => s.trim()).filter(Boolean));
    for (const v of [...sel]) if (!allIds.has(v)) sel.delete(v);
    if (!sel.size) RECORD.annotators.filter(m => allIds.has(m)).forEach(m => sel.add(m));
    window.renderMultiDropdown(p1host, {
      groups: _ddGroups(groups), selected: sel, single: false, placeholder: "Models",
      onChange: (arr) => { p1hidden.value = arr.join(","); },
    });
    p1hidden.value = [...sel].join(",");
  }
  const p2 = document.getElementById("p2-layer2-model");
  if (p2 && p2.tagName === "SELECT") {
    const prev = p2.value || PREFERRED_LAYER2;
    p2.innerHTML = "";
    for (const [key, label] of [["local", "Local"], ["cloud", "Cloud"]]) {
      const list = groups[key] || [];
      if (!list.length) continue;
      const og = document.createElement("optgroup"); og.label = label;
      for (const m of list) { const o = document.createElement("option"); o.value = m.id; o.textContent = m.tag; og.appendChild(o); }
      p2.appendChild(og);
    }
    if ([...p2.options].some(o => o.value === prev)) p2.value = prev;
  }
}

function _populateModelDropdowns(groups) {
  for (const s of [$("#span-model"), $("#upload-model")]) {
    if (!s) continue;
    const prev = s.value;
    s.innerHTML = "";
    for (const [key, label] of [["local", "Local"], ["cloud", "Cloud"]]) {
      const list = groups[key] || [];
      if (!list.length) continue;
      const og = document.createElement("optgroup");
      og.label = label;
      for (const m of list) {
        const opt = document.createElement("option");
        opt.value = m.id; opt.textContent = m.tag;
        og.appendChild(opt);
      }
      s.appendChild(og);
    }
    const opts = [...s.querySelectorAll("option")].map(o => o.value);
    if (prev && opts.includes(prev)) s.value = prev;
    else if (RECORD.annotators[0] && opts.includes(RECORD.annotators[0])) s.value = RECORD.annotators[0];
  }
}

async function refreshModelsIntoSelects() {
  const groups = await _fetchModelGroups();
  ext.models = [...(groups.local || []), ...(groups.cloud || [])].map(m => m.id);
  if (!_checksBuilt) {            // build pickers once; don't clobber user selections
    _buildPhaseModelPickers(groups);
    _checksBuilt = true;
  }
  _populateModelDropdowns(groups);
}
})();

/* ============================================================
 * Phase-5b additions: Live jobs polling + Phase-1/2 batch buttons.
 * Self-contained IIFE; reuses #live-jobs container in topbar and the
 * Background-runs drawer pane.
 * ============================================================ */
(() => {
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);
const escapeHtml = (s) => String(s ?? "")
  .replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")
  .replaceAll('"',"&quot;").replaceAll("'","&#39;");

let _pollTimer = null;
let _panelOpen = false;
let _knownJobs = new Set();   // track new jobs so we auto-open the panel
let _seenFailed = new Set();  // jobs we've already auto-popped the panel for on failure

document.addEventListener("DOMContentLoaded", initLiveJobs);
if (document.readyState !== "loading") initLiveJobs();
let _initialised = false;
function initLiveJobs() {
  if (_initialised) return;
  _initialised = true;
  $("#jobs-toggle")?.addEventListener("click", toggleJobsPanel);

  // Background-runs buttons
  $("#run-phase1-btn")?.addEventListener("click", runPhase1);
  $("#run-phase1-doc-btn")?.addEventListener("click", runPhase1CurrentDoc);
  $("#run-phase2-btn")?.addEventListener("click", runPhase2);

  // Populate guideline dropdown for Phase-1 form
  populateP1Guidelines();

  // Start polling on a 2-second cadence regardless of panel visibility
  // (so a new job auto-opens the panel). Stop only when no in-flight jobs.
  startPolling();

  // Schema viewer (Annotation tab — left panel) + Arbitrate manager tab listeners
  $("#sv-load")?.addEventListener("click", svLoad);
  $("#sv-save")?.addEventListener("click", svSave);
  $("#sv-delete")?.addEventListener("click", svDelete);
  $("#am-load")?.addEventListener("click", amLoad);
  $("#am-save")?.addEventListener("click", amSave);
  $("#am-delete")?.addEventListener("click", amDelete);
  // Expose for cross-IIFE tab-switch auto-load
  window._svLoad = svLoad;
  window._amLoad = amLoad;
}

// ---------------------------------------------------------------------------
// Schema viewer tab
// ---------------------------------------------------------------------------

async function svLoad() {
  const version = $("#sv-select")?.value || "v1";
  try {
    const r = await fetch(`/api/schema_prompt?version=${encodeURIComponent(version)}`);
    if (!r.ok) { _setInlineStatus("#sv-status", "load failed: " + r.status, "err"); return; }
    const d = await r.json();
    const ta = $("#sv-text"); if (ta) ta.value = d.text || "";
    _setInlineStatus("#sv-status", `loaded ${d.filename}`, "ok");
  } catch(e) { _setInlineStatus("#sv-status", "error: " + e.message, "err"); }
}

async function svSave() {
  const version = ($("#sv-new-version")?.value || "").trim() || $("#sv-select")?.value || "v1";
  const text = $("#sv-text")?.value || "";
  try {
    const r = await fetch(`/api/schema_prompt?version=${encodeURIComponent(version)}`,
      { method: "PUT", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({text}) });
    if (!r.ok) { _setInlineStatus("#sv-status", "save failed: " + r.status, "err"); return; }
    await populateP1Guidelines();
    const sel = $("#sv-select"); if (sel) sel.value = version;
    _setInlineStatus("#sv-status", `saved schema_${version}.md`, "ok");
  } catch(e) { _setInlineStatus("#sv-status", "error: " + e.message, "err"); }
}

async function svDelete() {
  const version = $("#sv-select")?.value || "";
  if (!version || version === "v1") { alert("v1 is protected and cannot be deleted."); return; }
  if (!confirm(`Delete schema_${version}.md? This cannot be undone.`)) return;
  try {
    const r = await fetch(`/api/schema_prompt?version=${encodeURIComponent(version)}`,
      { method: "DELETE" });
    if (!r.ok) { _setInlineStatus("#sv-status", "delete failed: " + r.status, "err"); return; }
    await populateP1Guidelines();
    const ta = $("#sv-text"); if (ta) ta.value = "";
    _setInlineStatus("#sv-status", `deleted schema_${version}.md`, "ok");
  } catch(e) { _setInlineStatus("#sv-status", "error: " + e.message, "err"); }
}

// ---------------------------------------------------------------------------
// Arbitrate manager tab
// ---------------------------------------------------------------------------

async function amLoad() {
  const version = $("#am-select")?.value || "v1";
  try {
    const r = await fetch(`/api/arbitrate_prompt?version=${encodeURIComponent(version)}`);
    if (!r.ok) { _setInlineStatus("#am-status", "load failed: " + r.status, "err"); return; }
    const d = await r.json();
    const ta = $("#am-text"); if (ta) ta.value = d.text || "";
    _setInlineStatus("#am-status", `loaded ${d.filename}`, "ok");
  } catch(e) { _setInlineStatus("#am-status", "error: " + e.message, "err"); }
}

async function amSave() {
  const newVer = ($("#am-new-version")?.value || "").trim();
  const version = newVer || $("#am-select")?.value || "v1";
  const text = $("#am-text")?.value || "";
  try {
    const r = await fetch(`/api/arbitrate_prompt?version=${encodeURIComponent(version)}`,
      { method: "PUT", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({text}) });
    if (!r.ok) { _setInlineStatus("#am-status", "save failed: " + r.status, "err"); return; }
    await populateP1Guidelines();
    const sel = $("#am-select"); if (sel) sel.value = version;
    _setInlineStatus("#am-status", `saved arbitrate_${version}.md`, "ok");
  } catch(e) { _setInlineStatus("#am-status", "error: " + e.message, "err"); }
}

async function amDelete() {
  const version = $("#am-select")?.value || "";
  if (!version || version === "v1") { alert("v1 is protected and cannot be deleted."); return; }
  if (!confirm(`Delete arbitrate_${version}.md? This cannot be undone.`)) return;
  try {
    const r = await fetch(`/api/arbitrate_prompt?version=${encodeURIComponent(version)}`,
      { method: "DELETE" });
    if (!r.ok) { _setInlineStatus("#am-status", "delete failed: " + r.status, "err"); return; }
    await populateP1Guidelines();
    _setInlineStatus("#am-status", `deleted arbitrate_${version}.md`, "ok");
    const ta = $("#am-text"); if (ta) ta.value = "";
  } catch(e) { _setInlineStatus("#am-status", "error: " + e.message, "err"); }
}

async function populateP1Guidelines() {
  try {
    const data = await fetch("/api/guidelines").then(r => r.json());
    for (const id of ["#p1-guideline", "#p2-guideline", "#gm-select"]) {
      const sel = $(id);
      if (!sel) continue;
      sel.innerHTML = "";
      for (const g of data.guidelines || []) {
        const opt = document.createElement("option");
        opt.value = g.version;
        // frozen arms stay selectable for runs, and say what they are
        opt.textContent = g.version + (g.locked ? " — frozen" : "");
        sel.appendChild(opt);
      }
      sel.value = "v1";
    }
  } catch {}
  // Populate arbitrate version selects
  try {
    const data = await fetch("/api/arbitrate_prompts").then(r => r.json());
    for (const id of ["#p2-arbitrate-version", "#am-select"]) {
      const sel = $(id);
      if (!sel) continue;
      sel.innerHTML = "";
      for (const g of data.arbitrate_prompts || []) {
        const opt = document.createElement("option");
        opt.value = g.version; opt.textContent = g.version;
        sel.appendChild(opt);
      }
      sel.value = "v1";
    }
  } catch {}
  // Populate schema version select
  try {
    const data = await fetch("/api/schema_prompts").then(r => r.json());
    const sel = $("#sv-select");
    if (sel) {
      sel.innerHTML = "";
      for (const g of data.schema_prompts || []) {
        const opt = document.createElement("option");
        opt.value = g.version; opt.textContent = g.version;
        sel.appendChild(opt);
      }
      sel.value = "v1";
    }
  } catch {}
}

function toggleJobsPanel() {
  const panel = $("#live-jobs");
  _panelOpen = panel.classList.toggle("hidden") ? false : true;
  $("#jobs-toggle").textContent = _panelOpen ? "Jobs ▴" : "Jobs ▾";
  if (_panelOpen) refreshJobsOnce();
}

function startPolling() {
  if (_pollTimer) return;
  const tick = async () => {
    try {
      const haveLive = await refreshJobsOnce();
      // Keep polling forever — the cost is one tiny request every 2s. The
      // benefit is the panel becomes "live" the moment a job appears even
      // if the user kicked it off via a separate flow (run_matrix etc.).
      // We do throttle if nothing is in-flight: slow to 10s.
      const interval = haveLive ? 2000 : 10000;
      _pollTimer = setTimeout(tick, interval);
    } catch {
      _pollTimer = setTimeout(tick, 5000);
    }
  };
  tick();
}

async function refreshJobsOnce() {
  let jobs;
  try {
    jobs = (await fetch("/api/jobs").then(r => r.json())).jobs || [];
  } catch { return false; }

  // Auto-open the panel when a new job appears or any job has just failed.
  let haveLive = false;
  let newSeen = false;
  let newFailure = false;
  const ids = new Set();
  for (const j of jobs) {
    ids.add(j.job_id);
    if (j.status === "queued" || j.status === "running") haveLive = true;
    if (!_knownJobs.has(j.job_id)) newSeen = true;
    if (j.status === "failed" && !_seenFailed.has(j.job_id)) {
      newFailure = true;
      _seenFailed.add(j.job_id);
    }
  }
  _knownJobs = ids;
  if ((newSeen || newFailure) && !_panelOpen) {
    _panelOpen = true;
    $("#live-jobs")?.classList.remove("hidden");
    if ($("#jobs-toggle")) $("#jobs-toggle").textContent = "Jobs ▴";
  }

  const panel = $("#live-jobs");
  if (!panel) return haveLive;
  panel.innerHTML = "";
  for (const j of jobs.slice(0, 20)) {
    const pct = j.progress.total ? Math.round(100 * j.progress.done / j.progress.total) : 0;
    const row = document.createElement("div");
    row.className = "job-row";
    row.dataset.jobId = j.job_id;
    const errInline = j.status === "failed" && j.error
      ? `<div class="job-error" style="grid-column: 2 / -1; color: #ffb4b4; font-size: 11px; margin-top: 2px;">⚠ ${escapeHtml(j.error)}</div>`
      : "";
    row.innerHTML = `
      <span class="kind ${j.kind}">${j.kind}</span>
      <span class="lbl" title="${escapeHtml(j.label || j.job_id)}">${escapeHtml(j.label || j.job_id)}</span>
      <div class="progress" title="${j.progress.done}/${j.progress.total}">
        <div class="bar" style="width:${pct}%"></div>
        <span class="label">${j.progress.done}/${j.progress.total} (${pct}%)</span>
      </div>
      <div style="display:flex;gap:4px;justify-content:flex-end;align-items:center;">
        <span class="status ${j.status}" title="${escapeHtml(j.error || '')}">${j.status}</span>
        ${j.status === "queued" ? `<button class="cancel-btn" data-job="${j.job_id}">×</button>` : ""}
      </div>
      ${errInline}`;
    panel.appendChild(row);
  }
  for (const b of panel.querySelectorAll(".cancel-btn")) {
    b.addEventListener("click", () => cancelJob(b.dataset.job));
  }
  return haveLive;
}

async function cancelJob(jobId) {
  try {
    const r = await fetch(`/api/jobs/${jobId}/cancel`, { method: "POST" });
    if (!r.ok) alert(`cancel failed: HTTP ${r.status}`);
  } catch (e) { alert(`cancel failed: ${e.message}`); }
  refreshJobsOnce();
}

// ------------------------------------------------------------
// Phase-1 / Phase-2 batch buttons
// ------------------------------------------------------------
function _openJobsPanel() {
  const panel = $("#live-jobs");
  if (!panel) return;
  panel.classList.remove("hidden");
  _panelOpen = true;
  if ($("#jobs-toggle")) $("#jobs-toggle").textContent = "Jobs ▴";
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function _setInlineStatus(selector, text, cls) {
  const el = $(selector);
  if (!el) return;
  el.textContent = text;
  el.className = cls || "";
}

async function runPhase1CurrentDoc() {
  const docId = $("#doc-select")?.value;
  if (!docId) { _setInlineStatus("#p1-status", "no document open", "err"); return; }
  const models = ($("#p1-models").value || "").split(",").map(s => s.trim()).filter(Boolean);
  const guideline_version = $("#p1-guideline").value || "v1";
  if (!models.length) { _setInlineStatus("#p1-status", "enter at least one model", "err"); return; }
  _setInlineStatus("#p1-status", "enqueueing " + docId + "\u2026", "");
  try {
    const r = await fetch("/api/run_phase1", {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ models, guideline_version, doc_ids: [docId] }),
    });
    if (!r.ok) { const t = await r.text(); _setInlineStatus("#p1-status", "failed: " + r.status + " " + t.slice(0,140), "err"); return; }
    const job = await r.json();
    _setInlineStatus("#p1-status", "enqueued " + docId + " \u00b7 " + guideline_version + " \u00b7 " + models.join(",") + " \u2014 watch Live jobs", "ok");
    _openJobsPanel(); refreshJobsOnce();
  } catch (e) {
    _setInlineStatus("#p1-status", "request failed: " + e.message, "err");
  }
}

async function runPhase1() {
  const models = ($("#p1-models").value || "").split(",").map(s => s.trim()).filter(Boolean);
  const guideline_version = $("#p1-guideline").value || "v1";
  const subset = ($("#p1-doc-subset").value || "").split(",").map(s => s.trim()).filter(Boolean);
  const body = { models, guideline_version };
  if (subset.length) body.doc_ids = subset;

  if (!models.length) {
    _setInlineStatus("#p1-status", "enter at least one model", "err");
    return;
  }

  const expectedDocs = subset.length || 50;
  const ok = confirm(
    "Phase-1 batch:\n" +
    "  models:    " + models.join(", ") + "\n" +
    "  guideline: " + guideline_version + "\n" +
    "  approx " + models.length + " x " + expectedDocs + " = " + (models.length * expectedDocs) + " cells\n\n" +
    "Needs Ollama running with the models pulled. On 6 GB GPU each cell is ~3-5 min.\n\n" +
    "Proceed?"
  );
  if (!ok) return;

  _setInlineStatus("#p1-status", "enqueueing…", "");
  try {
    const r = await fetch("/api/run_phase1", {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const t = await r.text();
      _setInlineStatus("#p1-status", "failed: " + r.status + " " + t.slice(0,140), "err");
      return;
    }
    const job = await r.json();
    _setInlineStatus("#p1-status",
      "enqueued · " + job.job_id + " (" + job.progress.total + " cells) — watch the Live jobs panel",
      "ok");
    _openJobsPanel();
    refreshJobsOnce();
  } catch (e) {
    _setInlineStatus("#p1-status", "request failed: " + e.message, "err");
  }
}

async function runPhase2() {
  const skip_layer2 = $("#p2-skip-layer2").checked;
  const layer2_model = $("#p2-layer2-model").value || window.__vaRecord?.layer2 || "";
  const align_threshold = parseFloat($("#p2-align-threshold").value) || 0.78;
  const version = $("#p2-guideline")?.value || "v1";
  const arbitrate_version = $("#p2-arbitrate-version")?.value || "v1";
  const body = { skip_layer2, layer2_model, align_threshold, version, arbitrate_version };
  const ok = confirm("Phase-2 batch:\n" +
                     "  skip_layer2:     " + skip_layer2 + "\n" +
                     "  layer2_model:    " + layer2_model + "\n" +
                     "  align_threshold: " + align_threshold + "\n" +
                     "  guideline:       " + version + "\n" +
                     "  conflict detect: " + arbitrate_version + "\n\n" +
                     (skip_layer2 ? "Uses the deterministic STUB — its labels are not arbitrated; do not diagnose on them.\n\n" : "") +
                     "Proceed?");
  if (!ok) return;

  _setInlineStatus("#p2-status", "enqueueing…", "");
  try {
    const r = await fetch("/api/run_phase2", {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const t = await r.text();
      _setInlineStatus("#p2-status", "failed: " + r.status + " " + t.slice(0,140), "err");
      return;
    }
    const job = await r.json();
    _setInlineStatus("#p2-status",
      "enqueued · " + job.job_id + " (" + job.progress.total + " docs) — watch the Live jobs panel",
      "ok");
    _openJobsPanel();
    refreshJobsOnce();
  } catch (e) {
    _setInlineStatus("#p2-status", "request failed: " + e.message, "err");
  }
}
})();

/* ============================================================
 * Phase-5c additions: resizable three-pane splitters.
 * Mounts drag handlers on every .splitter element. Each splitter sits
 * between two .pane siblings and adjusts their flex-basis on drag.
 * ============================================================ */
(() => {
  const init = () => {
    const splitters = document.querySelectorAll(".splitter");
    if (!splitters.length) return;

    splitters.forEach((sp) => sp.addEventListener("mousedown", startDrag));

    // When the graph pane changes size, tell Cytoscape to re-measure its
    // canvas and re-fit the graph into the new viewport.
    const cyEl = document.getElementById("cy");
    if (cyEl && window.ResizeObserver) {
      let fitTimer = null;
      let lastW = 0, lastH = 0;
      const ro = new ResizeObserver((entries) => {
        const cy = window.__cyInstance;
        if (!cy) return;
        const r = entries[entries.length - 1].contentRect;
        const dW = Math.abs(r.width - lastW), dH = Math.abs(r.height - lastH);
        lastW = r.width; lastH = r.height;
        cy.resize();
        // Only re-fit on substantial changes (splitter drags / pane toggles).
        // Small reflows (a sibling label wrapping) used to yank the camera
        // back to the global view on every hover — see iteration log 2.18.
        if (dW < 30 && dH < 30) return;
        clearTimeout(fitTimer);
        fitTimer = setTimeout(() => cy && cy.fit(undefined, 50), 120);
      });
      ro.observe(cyEl);
    }
  };

  function startDrag(e) {
    e.preventDefault();
    const splitter = e.currentTarget;
    const prev = splitter.previousElementSibling;
    const next = splitter.nextElementSibling;
    if (!prev || !next) return;

    const startX = e.clientX;
    const startPrevW = prev.getBoundingClientRect().width;
    const startNextW = next.getBoundingClientRect().width;
    const MIN = 160; // matches .pane { min-width }

    splitter.classList.add("dragging");
    document.body.classList.add("col-resizing");

    const onMove = (ev) => {
      let dx = ev.clientX - startX;
      // Clamp so neither pane goes below MIN.
      if (startPrevW + dx < MIN) dx = MIN - startPrevW;
      if (startNextW - dx < MIN) dx = startNextW - MIN;
      const newPrev = startPrevW + dx;
      const newNext = startNextW - dx;
      prev.style.flex = `0 0 ${newPrev}px`;
      next.style.flex = `0 0 ${newNext}px`;
    };
    const onUp = () => {
      splitter.classList.remove("dragging");
      document.body.classList.remove("col-resizing");
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }

  if (document.readyState !== "loading") init();
  else document.addEventListener("DOMContentLoaded", init);
})();

/* ============================================================
 * Phase-5d additions:
 *   - Delete-doc button
 *   - Text font-size controls
 *   - KG settings popover (node/label sizes, edge labels)
 *   - Horizontal splitter (drawer height)
 *   - Live facts refresh while a reextract job is in-flight on current doc
 * ============================================================ */
(() => {
  const $ = (s) => document.querySelector(s);

  const STYLE = {
    nodeSize: 60, labelSize: 13, edgeLabelSize: 11, edgeWidth: 3,
    showEdgeLabels: true, showNodeLabels: true,
  };

  function applyCyStyle() {
    const cy = window.__cyInstance;
    if (!cy) return;
    cy.style()
      .selector("node").style({
        "width": STYLE.nodeSize,
        "height": STYLE.nodeSize,
        "font-size": STYLE.labelSize,
        "label": STYLE.showNodeLabels ? "data(label)" : "",
      })
      .selector("edge").style({
        "width": STYLE.edgeWidth,
        "font-size": STYLE.edgeLabelSize,
        "label": STYLE.showEdgeLabels ? "data(label)" : "",
      })
      .update();
  }

  function initFontControls() {
    const body = document.body;
    const setFs = (px) => {
      px = Math.max(10, Math.min(28, px));
      document.documentElement.style.setProperty("--doc-font-size", px + "px");
      body.dataset.docFontPx = String(px);
    };
    setFs(14);
    $("#text-font-inc")?.addEventListener("click", () =>
      setFs(parseInt(body.dataset.docFontPx || "14", 10) + 1));
    $("#text-font-dec")?.addEventListener("click", () =>
      setFs(parseInt(body.dataset.docFontPx || "14", 10) - 1));
  }

  function initCySettings() {
    const panel = $("#graph-settings");
    const toggleBtn = $("#cy-settings-btn");
    const fitBtn = $("#cy-fit-btn");
    const resetBtn = $("#cy-settings-reset");

    toggleBtn?.addEventListener("click", () => {
      panel?.classList.toggle("hidden");
    });
    document.addEventListener("click", (e) => {
      if (!panel || panel.classList.contains("hidden")) return;
      if (panel.contains(e.target) || toggleBtn?.contains(e.target)) return;
      panel.classList.add("hidden");
    });

    fitBtn?.addEventListener("click", () => {
      const cy = window.__cyInstance;
      if (cy) cy.fit(undefined, 40);
    });

    const wireSlider = (id, key, valId) => {
      const el = $("#" + id), v = $("#" + valId);
      if (!el) return;
      el.addEventListener("input", () => {
        STYLE[key] = parseFloat(el.value);
        if (v) v.textContent = el.value;
        applyCyStyle();
      });
    };
    wireSlider("cy-node-size", "nodeSize", "cy-node-size-v");
    wireSlider("cy-label-size", "labelSize", "cy-label-size-v");
    wireSlider("cy-edge-label-size", "edgeLabelSize", "cy-edge-label-size-v");
    wireSlider("cy-edge-width", "edgeWidth", "cy-edge-width-v");

    $("#cy-show-edge-labels")?.addEventListener("change", (e) => {
      STYLE.showEdgeLabels = e.target.checked; applyCyStyle();
    });
    $("#cy-show-node-labels")?.addEventListener("change", (e) => {
      STYLE.showNodeLabels = e.target.checked; applyCyStyle();
    });

    resetBtn?.addEventListener("click", () => {
      Object.assign(STYLE, {
        nodeSize: 60, labelSize: 13, edgeLabelSize: 11, edgeWidth: 3,
        showEdgeLabels: true, showNodeLabels: true,
      });
      $("#cy-node-size").value = 60;       $("#cy-node-size-v").textContent = "60";
      $("#cy-label-size").value = 13;      $("#cy-label-size-v").textContent = "13";
      $("#cy-edge-label-size").value = 11; $("#cy-edge-label-size-v").textContent = "11";
      $("#cy-edge-width").value = 3;       $("#cy-edge-width-v").textContent = "3";
      $("#cy-show-edge-labels").checked = true;
      $("#cy-show-node-labels").checked = true;
      applyCyStyle();
    });

    // Re-apply on every fresh cytoscape render
    const reapply = () => setTimeout(applyCyStyle, 80);
    const obs = new MutationObserver(reapply);
    const cyEl = document.getElementById("cy");
    if (cyEl) obs.observe(cyEl, { childList: true, subtree: false });
  }

  function initDeleteDoc() {
    const btn = $("#delete-doc-btn");
    if (!btn) return;
    btn.addEventListener("click", async () => {
      const sel = $("#doc-select");
      const docId = sel?.value;
      if (!docId) { alert("No doc selected."); return; }
      const ok = confirm(
        "Permanently DELETE all artefacts for doc: " + docId + "?\n\n" +
        "This removes:\n" +
        "  data/parsed/" + docId + ".json\n" +
        "  data/facts/*/" + docId + ".json (all annotators)\n" +
        "  data/conflicts/" + docId + "*.json\n" +
        "  data/graphs/" + docId + ".json\n" +
        "  data/verifications/" + docId + ".json\n\n" +
        "Cannot be undone."
      );
      if (!ok) return;
      try {
        const r = await fetch("/api/doc/" + encodeURIComponent(docId), { method: "DELETE" });
        if (!r.ok) {
          let msg = "HTTP " + r.status;
          try { const j = await r.json(); msg = j.message || msg; } catch {}
          alert("Delete failed: " + msg);
          return;
        }
        const data = await r.json();
        // Refresh the doc list and switch to the first remaining doc
        state.docs = (await fetch("/api/docs").then(r => r.json())).docs;
        const shown = populateDocSelect();
        if (shown.length) {
          sel.dispatchEvent(new Event("change"));
        } else {
          document.querySelector("#doc-title").textContent = "(no docs left — upload one or run scripts/run_dry_run)";
          document.querySelector("#doc-body").textContent = "";
        }
        // brief toast via the live-jobs status area isn't appropriate; just alert
        console.info("deleted", data.n_removed, "files for", docId);
      } catch (e) {
        alert("Delete request failed: " + e.message);
      }
    });
  }

  // Horizontal splitter: drag to resize bottombar height
  function initHSplitter() {
    const split = document.querySelector(".hsplitter");
    const bar = document.getElementById("bottombar");
    if (!split || !bar) return;
    split.addEventListener("mousedown", (e) => {
      e.preventDefault();
      const startY = e.clientY;
      const startH = bar.getBoundingClientRect().height;
      split.classList.add("dragging");
      document.body.classList.add("row-resizing");
      const onMove = (ev) => {
        const dy = ev.clientY - startY;
        // drag UP = bigger bottombar
        let newH = startH - dy;
        const winH = window.innerHeight;
        newH = Math.max(60, Math.min(winH - 220, newH));
        bar.style.height = newH + "px";
        // tell cytoscape its container changed
        const cy = window.__cyInstance;
        if (cy) cy.resize();
      };
      const onUp = () => {
        split.classList.remove("dragging");
        document.body.classList.remove("row-resizing");
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
        const cy = window.__cyInstance;
        if (cy) cy.fit(undefined, 40);
      };
      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
    });
  }

  // Live refresh: while any reextract job for the current doc is in flight,
  // poll /api/facts and re-render the table so the user sees new facts appear.
  function initLiveFactsRefresh() {
    const docSelect = $("#doc-select");
    let timer = null;

    async function tick() {
      try {
        const jobs = (await fetch("/api/jobs").then(r => r.json())).jobs || [];
        const docId = docSelect?.value;
        if (!docId) return;
        const docKey = docId + ".json";
        const relevant = jobs.find(j =>
          (j.status === "running" || j.status === "queued") &&
          j.kind === "reextract" &&
          (j.label || "").includes(docId) ||
          // also match jobs with doc_paths containing the doc id (server-side
          // label-only check is too brittle):
          ((j.label || "").includes(docKey)));
        if (!relevant) return;
        // Re-fetch facts + graph
        const facts = await fetch("/api/facts/" + encodeURIComponent(docId)).then(r => r.json());
        // Push the new facts back into the page's #facts-table.
        // We can't reach the IIFE's `state`, so we dispatch a custom event that
        // anyone wanting to know about it can listen for.
        document.dispatchEvent(new CustomEvent("facts-changed", { detail: { docId, facts: facts.facts || [] } }));
        // Also flash the table to show it updated.
        const tbl = document.getElementById("facts-table");
        if (tbl && facts.facts) {
          tbl.classList.remove("live-updated");
          // force reflow
          void tbl.offsetWidth;
          tbl.classList.add("live-updated");
          // Update the visible facts count immediately
          const cnt = document.getElementById("facts-count");
          if (cnt) cnt.textContent = facts.count + " facts (live)";
        }
      } catch {}
    }

    const start = () => {
      if (timer) return;
      timer = setInterval(tick, 3000);
    };
    start();

    // Also when user switches doc, fire a refresh immediately
    docSelect?.addEventListener("change", () => setTimeout(tick, 500));
  }

  // Hook into the facts-changed event from inside the original IIFE: we need
  // to actually trigger a re-fetch of the doc to update the in-memory state.
  // Since the bootstrap IIFE doesn't expose state, the simplest is to
  // re-trigger the doc-select 'change' event when a reextract job finishes.
  function initJobCompletionReload() {
    let lastJobIds = new Set();
    let lastStatuses = new Map();
    setInterval(async () => {
      try {
        const jobs = (await fetch("/api/jobs").then(r => r.json())).jobs || [];
        const docSelect = $("#doc-select");
        const curDoc = docSelect?.value;
        for (const j of jobs) {
          const prev = lastStatuses.get(j.job_id);
          // When a reextract job for current doc transitions to done, reload
          if (prev && prev !== j.status && j.status === "done" && j.kind === "reextract") {
            if (curDoc && (j.label || "").includes(curDoc)) {
              docSelect.dispatchEvent(new Event("change"));
            }
          }
          lastStatuses.set(j.job_id, j.status);
        }
      } catch {}
    }, 3500);
  }

  function init() {
    initFontControls();
    initCySettings();
    initDeleteDoc();
    initHSplitter();
    initLiveFactsRefresh();
    initJobCompletionReload();
  }

  if (document.readyState !== "loading") init();
  else document.addEventListener("DOMContentLoaded", init);
})();

/* ============================================================
 * Phase-5e additions:
 *   - Build KG (single-doc Phase-2) button
 *   - Per-annotator color stripes on text marks
 *   - KG node click → highlight matching marks + facts rows
 *   - Atomic-facts sort dropdown (alignment / model / section)
 *   - Empty-facts CTA + "Extract here" single-doc Phase-1 button
 *
 * Strategy: this IIFE doesn't reach into the main IIFE's closure. Instead
 * it caches facts independently via /api/facts, then uses MutationObservers
 * to re-apply effects every time the main IIFE re-renders the text body or
 * facts tbody.
 * ============================================================ */
(() => {
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => document.querySelectorAll(s);

  const PALETTE = ["#3b82f6", "#10b981", "#ef4444", "#f59e0b", "#8b5cf6",
                   "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#6366f1"];

  const cache = {
    docId: null,
    factById: new Map(),        // fact_id → fact object (with _annotator)
    annotators: [],             // sorted list, deterministic order
    annotatorColor: new Map(),  // annotator → hex
    pairsByFactId: new Map(),   // fact_id → cluster_id (smallest fact_id in pair)
    sort: "default",
    graph: null,                // cached graph for surface-form lookups
  };

  function colorFor(annot) {
    return cache.annotatorColor.get(annot) || "#9ca3af";
  }
  // One palette across the legend, the in-text marks and the facts pane's
  // per-annotator columns. Exported rather than duplicated: a second palette is
  // a second answer to "which annotator is green".
  window.__vaColor = colorFor;

  async function refreshCacheFor(docId) {
    cache.docId = docId;
    cache.factById = new Map();
    cache.annotators = [];
    cache.annotatorColor = new Map();
    cache.pairsByFactId = new Map();

    try {
      const facts = await fetch(`/api/facts/${encodeURIComponent(docId)}`)
        .then(r => r.json());
      const annotSet = new Set();
      for (const f of facts.facts || []) {
        cache.factById.set(f.fact_id, f);
        annotSet.add(f._annotator);
      }
      cache.annotators = [...annotSet].sort();
      cache.annotators.forEach((a, i) =>
        cache.annotatorColor.set(a, PALETTE[i % PALETTE.length]));
    } catch (e) {
      console.warn("Phase-5e: facts cache failed", e);
    }

    try {
      const pairs = await fetch(`/api/pairs/${encodeURIComponent(docId)}`)
        .then(r => r.json());
      // Each pair gives us an alignment cluster: union all fact_ids that
      // appear together. Two facts in the same pair → same cluster id.
      // Use simple union-find keyed by fact_id.
      const parent = new Map();
      const find = (x) => { while (parent.get(x) !== x) { parent.set(x, parent.get(parent.get(x))); x = parent.get(x); } return x; };
      const union = (a, b) => {
        if (!parent.has(a)) parent.set(a, a);
        if (!parent.has(b)) parent.set(b, b);
        parent.set(find(a), find(b));
      };
      for (const p of pairs || []) {
        if (p.fact_a_id && p.fact_b_id) union(p.fact_a_id, p.fact_b_id);
      }
      for (const fid of cache.factById.keys()) {
        if (!parent.has(fid)) parent.set(fid, fid);
        cache.pairsByFactId.set(fid, find(fid));
      }
    } catch {}

    try {
      cache.graph = await fetch(`/api/graph/${encodeURIComponent(docId)}`)
        .then(r => r.json());
    } catch { cache.graph = null; }
  }

  // ---------- Annotator legend (above doc-body) -------------------
  function renderAnnotatorLegend() {
    let leg = $("#annotator-legend");
    if (!leg) {
      leg = document.createElement("div");
      leg.id = "annotator-legend";
      const docTitle = $("#doc-title");
      docTitle?.parentNode?.insertBefore(leg, docTitle.nextSibling);
    }
    leg.innerHTML = "";
    leg.title = "click: show/hide this annotator everywhere · double-click: solo";
    for (const a of cache.annotators) {
      const c = colorFor(a);
      const span = document.createElement("span");
      span.className = "leg";
      span.dataset.annot = a;
      span.innerHTML = `<span class="sw" style="background:${c}"></span>${a}`;
      // Proxy to the top-bar chips (owned by the main IIFE) via DOM clicks so
      // text pane, facts table AND the KG all follow one filter state.
      span.addEventListener("click", () => { _toggleTopbarChip(a); setTimeout(_syncLegendStates, 0); });
      span.addEventListener("dblclick", (e) => {
        e.preventDefault(); _soloAnnotator(a); setTimeout(_syncLegendStates, 0);
      });
      leg.appendChild(span);
    }
    if (!window.__legSyncWired) {
      window.__legSyncWired = true;
      $("#annotator-chips")?.addEventListener("click", () => setTimeout(_syncLegendStates, 0));
    }
    _syncLegendStates();
  }

  function _topbarChips() { return [...$$("#annotator-chips .chip")]; }
  function _toggleTopbarChip(annot) {
    // Topbar annotator selectors are multi-dropdowns; drive the MAIN IIFE's
    // selection state through window.__va (this IIFE has no closure access —
    // bare `state` here is a ReferenceError, see BUG-2026-07-02-E).
    const va = window.__va;
    if (!va) return;
    if (va.state.annotators.has(annot)) va.state.annotators.delete(annot);
    else va.state.annotators.add(annot);
    va.renderChips(); va.refreshText(); va.refreshFacts(); va.refreshGraphHighlights();
  }
  function _soloAnnotator(annot) {
    const va = window.__va;
    if (!va) return;
    va.state.annotators = new Set([annot]);
    va.renderChips(); va.refreshText(); va.refreshFacts(); va.refreshGraphHighlights();
  }
  function _syncLegendStates() {
    const on = window.__va && window.__va.state.annotators;
    if (!on) return;
    for (const sp of $$("#annotator-legend .leg")) {
      sp.classList.toggle("off", !on.has(sp.dataset.annot));
    }
  }

  // ---------- Apply colors to text marks --------------------------
  function applyMarkColors() {
    const marks = $$("#doc-body mark[data-fact-id]");
    for (const m of marks) {
      const f = cache.factById.get(m.dataset.factId);
      if (!f) continue;
      const c = colorFor(f._annotator);
      m.style.setProperty("--annot-color", c);
      m.setAttribute("data-annotator-color", "1");
      m.dataset.annotator = f._annotator;
    }
  }

  // ---------- Empty-facts CTA -------------------------------------
  function updateEmptyCTA() {
    const cta = $("#facts-empty-cta");
    const table = document.querySelector("#facts-table");
    if (!cta) return;
    // this IIFE has no closure access to the app state; window.__va is the
    // hand-off the rest of this file already uses
    const st = (window.__va && window.__va.state) || {};
    // Whether the document has facts is the APP's question, not this module's.
    // `cache.factById` is this IIFE's own copy, filled by its own fetch, and it
    // can legitimately be empty while the table below is full — in which case
    // this used to hide a rendered table behind "No facts on disk for this doc"
    // (BUG-2026-09-20-C).
    const hasNone = st.factsById ? st.factsById.size === 0
                                 : cache.factById.size === 0;
    cta.classList.toggle("hidden", !hasNone);
    if (table) table.classList.toggle("hidden", hasNone);
    if (hasNone) {
      // Two different empty documents: one waiting to be extracted, one that
      // must not be. Offering "Extract here" on the second is how invariant 1
      // gets broken by a single click.
      const d = (st.docs || []).find((x) => x.doc_id === st.doc);
      const locked = d && d.split === "eval";
      cta.classList.toggle("locked", !!locked);
      cta.querySelector("strong").textContent = locked
        ? "Locked eval document — empty on purpose."
        : "No facts on disk for this doc.";
      cta.querySelector(".hint").innerHTML = locked
        ? "This document is in <code>data/doc_lists/eurlex_eval.json</code>, "
        + "frozen 2026-07-31. The eval set is touched <b>once per guideline "
        + "version</b>, only to record a result — not browsed, not tuned "
        + "against. Diagnose on a <b>dev</b> document instead."
        : "Click <b>Extract here</b> above to run Phase-1 (needs Ollama). "
        + "The table will fill live as facts get extracted.";
    }
  }

  // ---------- Sort the facts tbody --------------------------------
  function sortTbody() {
    const container = document.querySelector("#facts-table");
    // Only handle Phase-5e sort modes; "paragraph" and "default" are left to renderFacts.
    if (!container || !["alignment", "model", "section"].includes(cache.sort)) return;
    const keyOf = (item) => {
      const f = cache.factById.get(item.dataset.factId);
      if (!f) return "";
      switch (cache.sort) {
        case "alignment":
          return (cache.pairsByFactId.get(f.fact_id) || f.fact_id) + "|" + f._annotator;
        case "model":
          return (f._annotator || "") + "|" + (f.source_locator?.section_path || "");
        case "section":
          return (f.source_locator?.section_path || "") + "|" + (f._annotator || "");
        default:
          return "";
      }
    };
    for (const grp of container.querySelectorAll(".fact-group")) {
      const items = [...grp.querySelectorAll(".fact-item[data-fact-id]")];
      const sorted = items.slice().sort((a, b) => keyOf(a).localeCompare(keyOf(b)));
      // Only touch the DOM if the order actually changed (prevents observer loop).
      if (sorted.some((el, i) => el !== items[i])) {
        sorted.forEach(it => grp.appendChild(it));
      }
    }
  }

  // ---------- KG node click → text + table highlight --------------
  function clearNodeHighlight() {
    for (const m of $$("#doc-body mark.node-highlight")) {
      m.classList.remove("node-highlight");
      m.style.removeProperty("--node-color");
    }
    for (const it of $$("#facts-table .fact-item.node-highlight")) {
      it.classList.remove("node-highlight");
      it.style.removeProperty("--node-color");
    }
  }

  function highlightByNode(nodeId, nodeData) {
    clearNodeHighlight();
    if (!nodeData) return;
    const surfaceForms = new Set(nodeData.surface_forms || []);
    const factIds = new Set(nodeData.fact_ids || []);
    const conflictColor = {
      contradiction: "#d32f2f",
      granularity: "#f57c00",
      redundancy: "#1976d2",
      escalate: "#8e24aa",
    }[nodeData.conflict_label] || "#2563eb";

    for (const m of $$("#doc-body mark[data-fact-id]")) {
      const f = cache.factById.get(m.dataset.factId);
      if (!f) continue;
      const hit = factIds.has(f.fact_id)
        || surfaceForms.has(f.subject) || surfaceForms.has(f.object);
      if (hit) {
        m.classList.add("node-highlight");
        m.style.setProperty("--node-color", conflictColor);
      }
    }
    for (const it of $$("#facts-table .fact-item[data-fact-id]")) {
      const f = cache.factById.get(it.dataset.factId);
      if (!f) continue;
      const hit = factIds.has(f.fact_id)
        || surfaceForms.has(f.subject) || surfaceForms.has(f.object);
      if (hit) {
        it.classList.add("node-highlight");
        it.style.setProperty("--node-color", conflictColor);
      }
    }
  }

  // Single-tap an edge -> highlight the annotation(s) it represents in the text
  // + facts panes (mirrors highlightByNode). In the reified KG the role edges
  // (subject/object/condition/temporal) carry no fact_ids of their own, so hop
  // to the statement node that does. clearNodeHighlight() (wired to background
  // tap) clears this, same as node highlights.
  function highlightByEdge(edge) {
    clearNodeHighlight();
    if (!edge) return;
    let src = edge;
    const role = edge.data("role");
    if (!(edge.data("fact_ids") || []).length &&
        (role === "subject" || role === "object" || role === "condition" || role === "temporal")) {
      const stmt = role === "subject" ? edge.target() : edge.source();
      if (stmt && stmt.data("type") === "statement") src = stmt;
    }
    const factIds = new Set(src.data("fact_ids") || []);
    if (!factIds.size) return;
    const conflictColor = {
      contradiction: "#d32f2f",
      granularity: "#f57c00",
      redundancy: "#1976d2",
    }[src.data("conflict_label")] || "#2563eb";
    let firstMark = null, firstItem = null;
    for (const m of $$("#doc-body mark[data-fact-id]")) {
      if (!factIds.has(m.dataset.factId)) continue;
      m.classList.add("node-highlight");
      m.style.setProperty("--node-color", conflictColor);
      if (!firstMark) firstMark = m;
    }
    for (const it of $$("#facts-table .fact-item[data-fact-id]")) {
      if (!factIds.has(it.dataset.factId)) continue;
      it.classList.add("node-highlight");
      it.style.setProperty("--node-color", conflictColor);
      if (!firstItem) firstItem = it;
    }
    // Bring the matched annotation into view: the AT/facts pane (right) first,
    // then the text mark if one exists.
    if (firstItem) firstItem.scrollIntoView({ behavior: "smooth", block: "center" });
    if (firstMark) firstMark.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  // Attach a node-tap handler to the *current* cytoscape instance. The main
  // IIFE re-creates `window.__cyInstance` on every render, so we re-bind.
  let _lastBoundCy = null;
  function attachCyHandler() {
    const cy = window.__cyInstance;
    if (!cy || cy === _lastBoundCy) return;
    _lastBoundCy = cy;
    cy.on("tap", "node", (evt) => {
      const n = evt.target;
      highlightByNode(n.id(), n.data());
    });
    // Single-tap an edge -> highlight its annotation(s); double-tap opens the
    // conflict modal (handled in the render block, not here).
    cy.on("tap", "edge", (evt) => { highlightByEdge(evt.target); });
    // Tapping the background clears the highlight.
    cy.on("tap", (evt) => { if (evt.target === cy) clearNodeHighlight(); });
  }

  // the server's reason for a refusal (403 eval set, 409 run of record)
  async function _why(r) {
    try { const d = await r.json(); return d.message || d.error || ""; }
    catch { return ""; }
  }

  // ---------- "Build KG" — re-run Phase-2 alignment for current doc ------
  async function buildKGForCurrentDoc() {
    const docId = $("#doc-select")?.value;
    if (!docId) return;
    const btn = $("#cy-rebuild-btn");
    const orig = btn?.textContent;
    if (btn) { btn.disabled = true; btn.textContent = "Building…"; }
    try {
      // The version on screen and the Runs tab's Layer-2 settings. This used
      // to run a STUB Phase-2 for v1 whatever was selected, i.e. overwrite the
      // baseline conflicts of record with unarbitrated labels in one click
      // (BUG-2026-09-23-E); the server now refuses that, and says why.
      const version = window.__variant || "v1";
      const l2 = window.__vaPhase2Settings?.() || {};
      const r = await fetch("/api/run_phase2", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc_ids: [docId], version, ...l2 }),
      });
      if (!r.ok) { alert(`Build KG refused (HTTP ${r.status}): ${await _why(r)}`); return; }
    } catch (e) {
      alert("Build KG request failed: " + e.message);
    } finally {
      if (btn) { btn.disabled = false; btn.textContent = orig || "Build KG"; }
    }
  }

  // ---------- "Extract here" — single-doc Phase-1 -----------------------
  async function extractForCurrentDoc() {
    const docId = $("#doc-select")?.value;
    if (!docId) return;
    const models = ($("#p1-models")?.value || (window.__vaRecord?.annotators || []).join(","))
      .split(",").map(s => s.trim()).filter(Boolean);
    const guideline_version = $("#p1-guideline")?.value || "v1";
    const btn = $("#facts-extract-btn");
    const orig = btn?.textContent;
    if (btn) { btn.disabled = true; btn.textContent = "Extracting…"; }
    try {
      const r = await fetch("/api/run_phase1", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ models, guideline_version, doc_ids: [docId] }),
      });
      if (!r.ok) { alert(`Extract refused (HTTP ${r.status}): ${await _why(r)}`); return; }
    } catch (e) {
      alert("Extract request failed: " + e.message);
    } finally {
      if (btn) { btn.disabled = false; btn.textContent = orig || "Extract here"; }
    }
  }

  // ---------- Wire-up ---------------------------------------------
  function watchDocChanges() {
    const sel = $("#doc-select");
    if (!sel) return;
    const onChange = async () => {
      const docId = sel.value;
      if (!docId) return;
      await refreshCacheFor(docId);
      renderAnnotatorLegend();
      setTimeout(() => {
        applyMarkColors();
        sortTbody();
        updateEmptyCTA();
        attachCyHandler();
      }, 250);
    };
    sel.addEventListener("change", onChange);
    if (sel.value) onChange();
  }

  function setupObservers() {
    const docBody = document.getElementById("doc-body");
    if (docBody) {
      const ob = new MutationObserver(() => applyMarkColors());
      ob.observe(docBody, { childList: true, subtree: true });
    }
    const factsEl = document.querySelector("#facts-table");
    if (factsEl) {
      const ob = new MutationObserver(() => {
        sortTbody();
        updateEmptyCTA();
      });
      ob.observe(factsEl, { childList: true, subtree: true });
    }
    const cyEl = document.getElementById("cy");
    if (cyEl) {
      const ob = new MutationObserver(() => setTimeout(attachCyHandler, 100));
      ob.observe(cyEl, { childList: true });
    }
  }

  function initSortDropdown() {
    const sel = $("#facts-sort");
    if (!sel) return;
    sel.addEventListener("change", () => {
      cache.sort = sel.value;
      sortTbody();
    });
  }

  function init() {
    initSortDropdown();
    $("#cy-rebuild-btn")?.addEventListener("click", buildKGForCurrentDoc);
    $("#facts-extract-btn")?.addEventListener("click", extractForCurrentDoc);
    setupObservers();
    setTimeout(watchDocChanges, 400);
  }

  if (document.readyState !== "loading") init();
  else document.addEventListener("DOMContentLoaded", init);
})();

/* ============================================================
 * Reusable multi/single-select dropdown (Local/Cloud groups).
 *   window.renderMultiDropdown(host, {
 *     groups:[{label, items:[{value,label}]}], selected:Set,
 *     single:bool, placeholder:str, onChange:(arr)=>void })
 * Used by the top annotator selectors and the Phase-1 model picker.
 * ============================================================ */
(() => {
  let openPop = null;
  document.addEventListener("click", (e) => {
    if (openPop && !openPop.contains(e.target) &&
        !(openPop.__btn && openPop.__btn.contains(e.target))) {
      openPop.classList.add("hidden"); openPop = null;
    }
  });
  window.addEventListener("scroll", () => {
    if (openPop) { openPop.classList.add("hidden"); openPop = null; }
  }, true);
  function summarize(host, opts) {
    const items = opts.groups.flatMap(g => g.items);
    const sel = items.filter(it => opts.selected.has(it.value));
    const txtEl = host.querySelector(".msd-txt");
    if (!txtEl) return;
    let txt;
    if (opts.single) txt = sel.length ? sel[0].label : (opts.placeholder || "select");
    else if (!items.length) txt = (opts.placeholder || "") + " (none)";
    else if (sel.length === 0) txt = (opts.placeholder || "none") + ": 0";
    else if (sel.length === items.length) txt = (opts.placeholder || "all") + ": all";
    else txt = (opts.placeholder ? opts.placeholder + ": " : "") + sel.length + "/" + items.length;
    txtEl.textContent = txt;
  }
  window.renderMultiDropdown = function(host, opts) {
    host.classList.add("msd");
    host.innerHTML = "";
    const btn = document.createElement("button");
    btn.type = "button"; btn.className = "msd-btn";
    btn.innerHTML = '<span class="msd-txt"></span><span class="msd-caret">▾</span>';
    const pop = document.createElement("div");
    pop.className = "msd-pop hidden";
    pop.__btn = btn;
    const items = opts.groups.flatMap(g => g.items);

    if (!opts.single && items.length) {
      const tools = document.createElement("div"); tools.className = "msd-tools";
      const mk = (label, fn) => {
        const b = document.createElement("button");
        b.type = "button"; b.className = "msd-mini"; b.textContent = label;
        b.addEventListener("click", (e) => { e.stopPropagation(); fn(); });
        return b;
      };
      tools.appendChild(mk("all",  () => { items.forEach(it => opts.selected.add(it.value)); sync(); }));
      tools.appendChild(mk("none", () => { items.forEach(it => opts.selected.delete(it.value)); sync(); }));
      pop.appendChild(tools);
    }

    for (const g of opts.groups) {
      if (g.label) { const h = document.createElement("div"); h.className = "msd-group"; h.textContent = g.label; pop.appendChild(h); }
      if (!g.items.length) { const e = document.createElement("div"); e.className = "msd-empty"; e.textContent = "(none)"; pop.appendChild(e); continue; }
      for (const it of g.items) {
        const row = document.createElement("label"); row.className = "msd-row";
        const inp = document.createElement("input");
        inp.type = opts.single ? "radio" : "checkbox";
        if (opts.single) inp.name = "msd-" + (host.id || "r");
        inp.value = it.value; inp.checked = opts.selected.has(it.value);
        inp.addEventListener("change", () => {
          if (opts.single) { opts.selected.clear(); if (inp.checked) opts.selected.add(it.value); }
          else if (inp.checked) opts.selected.add(it.value); else opts.selected.delete(it.value);
          summarize(host, opts);
          opts.onChange && opts.onChange([...opts.selected]);
          if (opts.single) { pop.classList.add("hidden"); openPop = null; }
        });
        row.appendChild(inp);
        const sp = document.createElement("span"); sp.textContent = it.label; row.appendChild(sp);
        pop.appendChild(row);
      }
    }
    function sync() {
      pop.querySelectorAll("input").forEach(i => { i.checked = opts.selected.has(i.value); });
      summarize(host, opts);
      opts.onChange && opts.onChange([...opts.selected]);
    }
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const willOpen = pop.classList.contains("hidden");
      if (openPop && openPop !== pop) openPop.classList.add("hidden");
      if (willOpen) {                          // position the fixed popover under the button
        const r = btn.getBoundingClientRect();
        pop.style.top = (r.bottom + 4) + "px";
        pop.style.left = r.left + "px";
      }
      pop.classList.toggle("hidden", !willOpen);
      openPop = willOpen ? pop : null;
    });
    host.appendChild(btn); host.appendChild(pop);
    summarize(host, opts);
    return host;
  };
})();

/* ============================================================
 * Collapsible top control bar. A draggable horizontal handle
 * between #topbar and #layout: drag to resize the header, drag
 * up (or double-click) to collapse it and give the three views
 * more vertical room. State persists in localStorage.
 * ============================================================ */
(() => {
  const init = () => {
    const tb = document.getElementById("topbar");
    const bar = document.getElementById("topbar-splitter");
    if (!tb || !bar) return;
    const KEY = "va-topbar";
    try {
      if (localStorage.getItem(KEY) === "collapsed") tb.classList.add("tb-collapsed");
    } catch {}
    const save = () => { try {
      localStorage.setItem(KEY, tb.classList.contains("tb-collapsed") ? "collapsed" : "expanded");
    } catch {} };
    bar.addEventListener("mousedown", (e) => {
      e.preventDefault();
      const startY = e.clientY;
      bar.classList.add("dragging");
      document.body.classList.add("row-resizing");
      const onMove = (ev) => {
        const dy = ev.clientY - startY;
        if (dy < -10) tb.classList.add("tb-collapsed");        // drag up = collapse
        else if (dy > 10) tb.classList.remove("tb-collapsed"); // drag down = expand
        if (window.__cyInstance) window.__cyInstance.resize();
      };
      const onUp = () => {
        bar.classList.remove("dragging");
        document.body.classList.remove("row-resizing");
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
        save();
      };
      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
    });
    bar.addEventListener("dblclick", () => {
      tb.classList.toggle("tb-collapsed");
      save();
    });
  };
  if (document.readyState !== "loading") init();
  else document.addEventListener("DOMContentLoaded", init);
})();
