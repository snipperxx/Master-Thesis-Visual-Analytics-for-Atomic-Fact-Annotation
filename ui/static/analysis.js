/* ============================================================
 * analysis.js — poster-aligned analysis views (append-only module).
 *
 * Adds three sibling views to the centre pane, switched by the
 * #center-tabs buttons injected in index.html:
 *
 *   Graph     — the existing Cytoscape KG (owned by app.js; untouched)
 *   Multiples — small-multiple per-annotator KGs with a SHARED layout
 *               (Schmidt et al. poster, Fig. 1 / "Graph-View")
 *   Heatmap   — fact×fact semantic-similarity matrix for an annotator
 *               pair, Hungarian matches outlined (poster Fig. 2 left)
 *   Stats     — fact-count histogram per annotator + per-section
 *               counts + IAA table (poster Fig. 2 right)
 *
 * Also owns the top-bar guideline-version dropdown (#variant-select):
 * picking a version sets window.__variant (see variant_shim.js) and
 * re-triggers the existing switchDoc plumbing. The version is a SESSION
 * MODE, not a property of the open document: it is populated corpus-wide
 * from /api/guideline_versions, it survives switching documents, and it is
 * remembered in localStorage. It used to be rebuilt per document and reset
 * to baseline on every document change, so "look at v2 across the corpus"
 * was impossible — one click on another document silently put you back on
 * v1 while the dropdown still read as it had.
 * Talks to app.js ONLY through the DOM + window.__cyInstance.
 * ============================================================ */
(() => {
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);
const esc = (s) => String(s ?? "")
  .replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")
  .replaceAll('"',"&quot;").replaceAll("'","&#39;");
const trunc = (s, n) => (s && s.length > n ? s.slice(0, n - 1) + "…" : (s || ""));

const PALETTE = ["#2563eb", "#f57c00", "#16a34a", "#9333ea", "#dc2626", "#0891b2"];
const CONFLICT_COLOR = {
  contradiction: "#d32f2f", granularity: "#f57c00",
  redundancy: "#1976d2", no_conflict: "#16a34a", unlabeled: "#94a3b8",
};

const VERSION_KEY = "va-guideline-version";
const A = {
  view: "graph",
  doc: null,
  variant: (() => { try { return localStorage.getItem(VERSION_KEY) || ""; }
                    catch { return ""; } })(),
  sim: null,            // last /api/similarity_matrix payload
  simKey: null,
  selAnnotA: null,
  selAnnotB: null,
  miniCys: [],
  docsMeta: new Map(),  // doc_id -> {variants: [...]}
};

// ------------------------------------------------------------
// Bootstrapping + doc/variant change detection
// ------------------------------------------------------------

let _initialised = false;
function init() {
  if (_initialised) return;
  _initialised = true;

  for (const tab of $$("#center-tabs .ctab")) {
    tab.addEventListener("click", () => setView(tab.dataset.view));
  }

  const vsel = $("#variant-select");
  vsel?.addEventListener("change", async () => {
    A.variant = vsel.value;
    window.__variant = vsel.value;
    try { localStorage.setItem(VERSION_KEY, vsel.value); } catch {}
    vsel.classList.toggle("variant-active", !!vsel.value);
    invalidate();
    // The document list is version-specific too: which documents have a
    // Phase-2 result depends on which version you are looking at.
    await refreshDocList();
    // Re-run the existing switchDoc plumbing (fetches go through the shim).
    $("#doc-select")?.dispatchEvent(new Event("change"));
    reloadActiveView();
  });

  window.__variant = A.variant;
  populateVersionSelect().then(refreshDocList);

  initPaneManager();

  // app.js boot() may switch the first doc without firing a change event,
  // so poll the select's value instead of relying on events.
  setInterval(() => {
    const cur = $("#doc-select")?.value || null;
    if (cur && cur !== A.doc) onDocChanged(cur);
  }, 800);
}
document.addEventListener("DOMContentLoaded", init);
if (document.readyState !== "loading") init();

function invalidate() {
  A.sim = null; A.simKey = null;
}

// Tell the reader where the labels on screen came from. `layer2_model` is
// "stub" for a --skip-layer2 run, whose granularity/redundancy labels are
// produced by a keyword heuristic, not by an arbitrator; those must never be
// mistaken for diagnosis material.
function renderRunBadge(docId) {
  const el = $("#run-badge");
  if (!el) return;
  const meta = A.docsMeta.get(docId) || {};
  const m = meta.layer2_model;
  if (!meta.has_conflicts || !m) {
    el.className = "run-badge hidden";
    el.textContent = "";
    return;
  }
  const stub = String(m) === "stub";
  el.className = "run-badge " + (stub ? "stub" : "real");
  el.textContent = stub ? "Layer 2: STUB — labels are not arbitrated"
                        : `Layer 2: ${m}`;
  el.title = stub
    ? "This Phase-2 run used --skip-layer2. The granularity / redundancy "
      + "labels come from a deterministic keyword stub, not from an "
      + "arbitrator model. Do not diagnose on them."
    : `Conflict labels arbitrated by ${m}.`;
}

// Which split this document is in. The eval set is locked (invariant 1: touched
// once per guideline version, only to record a result), and the interface lists
// it beside the dev set, so the only thing standing between a browse and a
// broken invariant is this badge. It is shown whenever the document is an eval
// one — NOT only when the panes are empty: several eval documents still carry
// facts from the retired local-4B pilot, so "it has annotations" is exactly the
// case where the reader needs telling.
function renderSplitBadge(docId) {
  const el = $("#split-badge");
  if (!el) return;
  const split = (A.docsMeta.get(docId) || {}).split;
  if (split !== "eval") {
    el.className = "run-badge hidden";
    el.textContent = "";
    return;
  }
  el.className = "run-badge stub";
  el.textContent = "🔒 EVAL — locked set, do not diagnose here";
  el.title = "data/doc_lists/eurlex_eval.json, frozen 2026-07-31. The eval set "
    + "is touched once per guideline version, only to record a result: never "
    + "inspected, never tuned against, never re-extracted from this screen. "
    + "Any facts shown are leftovers from the retired local-4B pilot.";
}

async function onDocChanged(docId) {
  A.doc = docId;
  renderRunBadge(docId);
  renderSplitBadge(docId);
  // The guideline version is deliberately NOT reset here: it is a session
  // mode that spans documents (see the header comment).
  A.selAnnotA = A.selAnnotB = null;
  invalidate();
  reloadActiveView();
}

async function populateVersionSelect() {
  const vsel = $("#variant-select");
  if (!vsel) return;
  let versions = [{ value: "", label: "v1 (baseline)", n_docs: 0 }];
  try {
    const r = await fetch("/api/guideline_versions").then(r => r.json());
    if (r.versions && r.versions.length) versions = r.versions;
  } catch {}
  const want = A.variant;
  vsel.innerHTML = versions.map(v =>
    `<option value="${esc(v.value)}">${esc(v.label)}` +
    (v.n_docs != null ? ` — ${v.n_docs} docs` : "") + `</option>`).join("");
  // A remembered version whose files have since been deleted must not leave
  // the page silently reading the baseline under a stale label.
  const ok = versions.some(v => v.value === want);
  vsel.value = ok ? want : "";
  if (!ok && want) {
    A.variant = ""; window.__variant = "";
    try { localStorage.removeItem(VERSION_KEY); } catch {}
  }
  vsel.classList.toggle("variant-active", !!vsel.value);
  vsel.style.display = "";          // always visible: it is a session mode
}

// Reload /api/docs under the current version (the shim appends `variant`)
// and rebuild #doc-select through app.js's own renderer.
async function refreshDocList() {
  try {
    const docs = (await fetch("/api/docs").then(r => r.json())).docs || [];
    for (const d of docs) A.docsMeta.set(d.doc_id, d);
    if (typeof state === "object" && state) state.docs = docs;
    if (typeof populateDocSelect === "function") populateDocSelect();
    renderRunBadge($("#doc-select")?.value);
  } catch {}
}

// ------------------------------------------------------------
// Pane maximize / collapse (text | graph | facts)
// ------------------------------------------------------------

const PANES = { text: "#pane-text", graph: "#pane-graph", facts: "#pane-facts" };
const PANE_LABELS = { text: "Document text", graph: "Graph / analysis", facts: "Atomic facts" };
let _maximized = null;

function initPaneManager() {
  for (const btn of $$(".pane-max")) {
    btn.addEventListener("click", () => {
      const key = btn.dataset.pane;
      _maximized = _maximized === key ? null : key;
      const el = $(PANES[key]);
      if (el) el.dataset.collapsed = "";          // maximizing un-collapses
      applyPaneLayout();
    });
  }
  for (const btn of $$(".pane-hide")) {
    btn.addEventListener("click", () => {
      const key = btn.dataset.pane;
      const el = $(PANES[key]);
      if (!el) return;
      el.dataset.collapsed = "1";
      if (_maximized === key) _maximized = null;
      applyPaneLayout();
    });
  }
  // Clicking a collapsed strip restores the pane.
  for (const [key, sel] of Object.entries(PANES)) {
    const el = $(sel);
    if (!el) continue;
    el.dataset.paneLabel = PANE_LABELS[key];
    el.addEventListener("click", () => {
      if (el.classList.contains("pane-collapsed")) {
        el.dataset.collapsed = "";
        applyPaneLayout();
      }
    });
  }
}

function applyPaneLayout() {
  for (const [key, sel] of Object.entries(PANES)) {
    const el = $(sel);
    if (!el) continue;
    const collapsed = el.dataset.collapsed === "1";
    if (_maximized) {
      el.style.display = key === _maximized ? "" : "none";
      el.classList.toggle("pane-maxed", key === _maximized);
      el.classList.remove("pane-collapsed");
    } else {
      el.style.display = "";
      el.classList.remove("pane-maxed");
      el.classList.toggle("pane-collapsed", collapsed);
    }
  }
  // Splitters make no sense while maximized / next to a collapsed pane edge.
  for (const sp of $$("#layout .splitter")) {
    sp.style.display = _maximized ? "none" : "";
  }
  // Re-measure every live cytoscape canvas (main + multiples) and re-fit —
  // the canvas geometry change here is intentional, unlike hover reflows.
  setTimeout(() => {
    try { window.__cyInstance?.resize(); window.__cyInstance?.fit(undefined, 50); } catch {}
    for (const c of A.miniCys) { try { c.resize(); c.fit(undefined, 12); } catch {} }
  }, 80);
}

// ------------------------------------------------------------
// View switching
// ------------------------------------------------------------

function setView(view) {
  A.view = view;
  for (const tab of $$("#center-tabs .ctab")) {
    tab.classList.toggle("active", tab.dataset.view === view);
  }
  $("#cy").classList.toggle("hidden", view !== "graph");
  $("#cy-multiples").classList.toggle("hidden", view !== "multiples");
  $("#heatmap-wrap").classList.toggle("hidden", view !== "heatmap");
  $("#stats-wrap").classList.toggle("hidden", view !== "stats");
  // Graph-specific chrome only makes sense on the graph view.
  for (const sel of ["#cy-rebuild-btn", "#cy-fit-btn", "#cy-settings-btn"]) {
    const el = $(sel);
    if (el) el.style.display = view === "graph" ? "" : "none";
  }
  const legend = $("#cy-legend");
  if (legend) legend.style.display = (view === "graph" || view === "multiples") ? "" : "none";
  const settings = $("#graph-settings");
  if (settings && view !== "graph") settings.classList.add("hidden");

  if (view === "graph" && window.__cyInstance) {
    setTimeout(() => { try { window.__cyInstance.resize(); window.__cyInstance.fit(undefined, 50); } catch {} }, 30);
  }
  reloadActiveView();
}

function reloadActiveView() {
  if (!A.doc) return;
  if (A.view === "heatmap") renderHeatmap();
  else if (A.view === "stats") renderStats();
  else if (A.view === "multiples") renderMultiples();
}

// ------------------------------------------------------------
// Heatmap view (poster Fig. 2 left)
// ------------------------------------------------------------

async function renderHeatmap() {
  const wrap = $("#heatmap-wrap");
  const key = `${A.doc}|${A.selAnnotA || ""}|${A.selAnnotB || ""}|${A.variant}`;
  if (A.sim && A.simKey === key) return;  // already rendered
  wrap.innerHTML = `<p class="hint" style="padding:8px">loading similarity matrix…</p>`;

  let q = "";
  if (A.selAnnotA && A.selAnnotB) {
    q = `?a=${encodeURIComponent(A.selAnnotA)}&b=${encodeURIComponent(A.selAnnotB)}`;
  }
  let d;
  try {
    d = await fetch(`/api/similarity_matrix/${A.doc}${q}`).then(r => r.json());
  } catch (e) {
    wrap.innerHTML = `<p class="hint" style="padding:8px">failed: ${esc(e.message)}</p>`;
    return;
  }
  if (d.note) {
    wrap.innerHTML = `<p class="hint" style="padding:8px">${esc(d.note)}</p>`;
    return;
  }
  A.sim = d; A.simKey = key;
  A.selAnnotA = d.annotator_a; A.selAnnotB = d.annotator_b;

  const matchByCell = new Map();
  for (const m of d.matched) matchByCell.set(`${m.ia},${m.ib}`, m);

  const annOpts = (sel) => (d.annotators || [])
    .map(a => `<option value="${esc(a)}" ${a === sel ? "selected" : ""}>${esc(a)}</option>`).join("");

  let html = `
    <div class="hm-controls">
      <label>rows <select id="hm-a">${annOpts(d.annotator_a)}</select></label>
      <span class="hm-x">×</span>
      <label>cols <select id="hm-b">${annOpts(d.annotator_b)}</select></label>
      <button id="hm-swap" class="tool-btn" title="Swap rows and columns">⇄</button>
      <span class="hm-legend">
        cosine <span class="hm-scale"></span> 1.0
        · outline = Hungarian match:
        <span class="dot contradiction"></span><span class="dot granularity"></span><span class="dot redundancy"></span>
      </span>
      ${d.backend === "tfidf" ? `<span class="hm-backend-warn" title="sentence-transformers not installed in this Python env — char-trigram TF-IDF fallback in use. Numbers are NOT the evaluation metric.">backend: TF-IDF fallback ⚠</span>` : ""}
    </div>
    <div class="hm-scroll"><table class="hm-table"><thead><tr>
      <th class="hm-corner">${esc(trunc(d.annotator_a, 14))} \\ ${esc(trunc(d.annotator_b, 14))}</th>`;
  d.facts_b.forEach((f, j) => {
    html += `<th class="hm-col" title="${esc(`b${j + 1}: ${f.text}`)}">b${j + 1}</th>`;
  });
  html += `</tr></thead><tbody>`;
  d.matrix.forEach((row, i) => {
    const fa = d.facts_a[i];
    html += `<tr><th class="hm-row" title="${esc(fa.text)}">a${i + 1}</th>`;
    row.forEach((v, j) => {
      const m = matchByCell.get(`${i},${j}`);
      const alpha = Math.max(0, Math.pow(Math.max(0, v), 2.2));
      const outline = m ? `box-shadow: inset 0 0 0 2px ${CONFLICT_COLOR[m.conflict_label] || "#475569"};` : "";
      html += `<td class="hm-cell ${m ? "matched" : ""}" data-i="${i}" data-j="${j}"
        title="${esc(`a${i + 1} × b${j + 1} — cos ${v.toFixed(3)}${m ? " · " + m.conflict_label : ""}`)}"
        style="background: rgba(37,99,235,${alpha.toFixed(3)}); ${outline}"></td>`;
    });
    html += `</tr>`;
  });
  html += `</tbody></table></div><div id="hm-detail" class="hint" style="padding:6px 8px">click a cell to inspect the pair — outlined cells are pipeline matches</div>`;
  wrap.innerHTML = html;

  $("#hm-a").addEventListener("change", (e) => { A.selAnnotA = e.target.value; A.simKey = null; renderHeatmap(); });
  $("#hm-b").addEventListener("change", (e) => { A.selAnnotB = e.target.value; A.simKey = null; renderHeatmap(); });
  $("#hm-swap").addEventListener("click", () => {
    [A.selAnnotA, A.selAnnotB] = [A.selAnnotB, A.selAnnotA];
    A.simKey = null; renderHeatmap();
  });
  wrap.querySelector("tbody").addEventListener("click", (e) => {
    const td = e.target.closest("td.hm-cell");
    if (!td) return;
    showHeatmapDetail(parseInt(td.dataset.i, 10), parseInt(td.dataset.j, 10), matchByCell);
    wrap.querySelectorAll("td.hm-cell.sel").forEach(c => c.classList.remove("sel"));
    td.classList.add("sel");
  });
}

function showHeatmapDetail(i, j, matchByCell) {
  const d = A.sim;
  const fa = d.facts_a[i], fb = d.facts_b[j];
  const v = d.matrix[i][j];
  const m = matchByCell.get(`${i},${j}`);
  const spo = (f) => `<span class="spo">⟨${esc(f.subject)} · ${esc(f.predicate)} · ${esc(f.object)}⟩</span>`;
  let html = `
    <div class="hm-pair">
      <div class="hm-pair-head">
        cos <b>${v.toFixed(3)}</b>
        ${m ? `<span class="lbl ${esc(m.conflict_label)}">${esc(m.conflict_label)}</span>
               <button class="tool-btn" id="hm-to-review" title="Open this pair in the Review tab">review →</button>`
            : `<span class="lbl unlabeled">not matched by pipeline</span>`}
      </div>
      <div class="hm-pair-grid">
        <div><b>a${i + 1}</b> · ${esc(d.annotator_a)}<div class="nl">${esc(fa.text)}</div>${spo(fa)}
          <div class="quote">"${esc(trunc(fa.quote, 220))}"</div></div>
        <div><b>b${j + 1}</b> · ${esc(d.annotator_b)}<div class="nl">${esc(fb.text)}</div>${spo(fb)}
          <div class="quote">"${esc(trunc(fb.quote, 220))}"</div></div>
      </div>
      ${m && (m.layer1_reason || m.layer2_reason)
        ? `<div class="reasons">L1: ${esc(m.layer1_reason || "—")}   L2: ${esc(m.layer2_reason || "—")}</div>` : ""}
    </div>`;
  $("#hm-detail").innerHTML = html;
  $("#hm-detail").classList.remove("hint");
  if (m) {
    $("#hm-to-review")?.addEventListener("click", () => {
      window.dispatchEvent(new CustomEvent("ava:open-review", {
        detail: { pairKey: `${m.fact_a_id || "-"}|${m.fact_b_id || "-"}` },
      }));
    });
  }
}

// ------------------------------------------------------------
// Stats view: triple-count histogram + per-section counts + IAA.
// Every count here is of TRIPLES (/api/facts returns one row per triple); a
// fact is one quote with one or more triples (BUG-2026-09-23-I).
// ------------------------------------------------------------

async function renderStats() {
  const wrap = $("#stats-wrap");
  wrap.innerHTML = `<p class="hint" style="padding:8px">loading…</p>`;
  let facts, iaa;
  try {
    [facts, iaa] = await Promise.all([
      fetch(`/api/facts/${A.doc}`).then(r => r.json()),
      fetch(`/api/iaa/${A.doc}`).then(r => r.json()),
    ]);
  } catch (e) {
    wrap.innerHTML = `<p class="hint" style="padding:8px">failed: ${esc(e.message)}</p>`;
    return;
  }

  const byAnn = new Map();
  const bySection = new Map();   // section -> Map(ann -> count)
  for (const f of facts.facts || []) {
    byAnn.set(f._annotator, (byAnn.get(f._annotator) || 0) + 1);
    const sec = (f.source_locator || {}).section_path || "?";
    if (!bySection.has(sec)) bySection.set(sec, new Map());
    const m = bySection.get(sec);
    m.set(f._annotator, (m.get(f._annotator) || 0) + 1);
  }
  const anns = [...byAnn.keys()].sort();
  const annColor = (a) => PALETTE[anns.indexOf(a) % PALETTE.length];
  const maxCount = Math.max(1, ...byAnn.values());

  // --- histogram (poster Fig. 2 right) ---
  let html = `<div class="stats-block"><h3>Triples per annotator
      <span class="hint">granularity disagreement = different bar lengths on the same doc</span></h3>`;
  for (const a of anns) {
    const n = byAnn.get(a);
    html += `<div class="st-bar-row">
      <span class="st-name" title="${esc(a)}">${esc(trunc(a, 22))}</span>
      <span class="st-track"><span class="st-fill" style="width:${(n / maxCount * 100).toFixed(1)}%; background:${annColor(a)}"></span></span>
      <span class="st-n">${n}</span></div>`;
  }
  html += `</div>`;

  // --- per-section table ---
  const secs = [...bySection.keys()].sort((x, y) =>
    x.localeCompare(y, undefined, { numeric: true }));
  html += `<div class="stats-block"><h3>Triples per section
      <span class="hint">hot rows = sections where annotators decompose differently</span></h3>
    <div class="st-scroll"><table class="st-table"><thead><tr><th>section</th>`;
  for (const a of anns) html += `<th title="${esc(a)}" style="color:${annColor(a)}">${esc(trunc(a, 12))}</th>`;
  html += `<th>spread</th></tr></thead><tbody>`;
  for (const sec of secs) {
    const m = bySection.get(sec);
    const vals = anns.map(a => m.get(a) || 0);
    const spread = Math.max(...vals) - Math.min(...vals);
    html += `<tr class="${spread >= 2 ? "st-hot" : ""}"><td class="st-sec">${esc(sec)}</td>`;
    for (const v of vals) html += `<td class="st-cnt">${v || "·"}</td>`;
    html += `<td class="st-spread">${spread > 0 ? "±" + spread : ""}</td></tr>`;
  }
  html += `</tbody></table></div></div>`;

  // --- IAA table ---
  html += `<div class="stats-block"><h3>Inter-annotator agreement
      <span class="hint">Jaccard = |matched| / |union| (poster metric)</span></h3>`;
  if (iaa.note || !(iaa.pairs || []).length) {
    html += `<p class="hint">${esc(iaa.note || "no aligned pairs")}</p>`;
  } else {
    html += `<table class="st-table iaa"><thead><tr>
      <th>pair</th><th>n_a</th><th>n_b</th><th>matched</th><th>Jaccard</th><th>mean cos</th><th>labels</th>
      </tr></thead><tbody>`;
    for (const p of iaa.pairs) {
      const jac = p.jaccard == null ? "—" : p.jaccard.toFixed(3);
      const chips = Object.entries(p.labels || {})
        .map(([l, n]) => `<span class="lbl ${esc(l)}" title="${esc(l)}">${n}</span>`).join(" ");
      html += `<tr>
        <td title="${esc(p.a)} ↔ ${esc(p.b)}">${esc(trunc(p.a, 14))} ↔ ${esc(trunc(p.b, 14))}</td>
        <td>${p.n_a}</td><td>${p.n_b}</td><td>${p.n_matched}</td>
        <td><span class="st-track sm"><span class="st-fill" style="width:${((p.jaccard || 0) * 100).toFixed(0)}%"></span></span> ${jac}</td>
        <td>${p.mean_cosine == null ? "—" : p.mean_cosine.toFixed(3)}</td>
        <td>${chips}</td></tr>`;
    }
    html += `</tbody></table>`;
  }
  html += `</div>`;
  wrap.innerHTML = html;
}

// ------------------------------------------------------------
// Small multiples view (poster Fig. 1) — shared layout per annotator
// ------------------------------------------------------------

async function renderMultiples() {
  const wrap = $("#cy-multiples");
  wrap.innerHTML = `<p class="hint" style="padding:8px">building small multiples…</p>`;
  for (const c of A.miniCys) { try { c.destroy(); } catch {} }
  A.miniCys = [];

  const merge = $("#merge-threshold")?.value || 0.78;
  let graph;
  try {
    const core = (document.querySelector("#cy-core-entities")?.checked ?? true) ? 1 : 0;
    graph = await fetch(`/api/graph/${A.doc}?merge_threshold=${merge}&core=${core}`).then(r => r.json());
  } catch (e) {
    wrap.innerHTML = `<p class="hint" style="padding:8px">failed: ${esc(e.message)}</p>`;
    return;
  }
  if (!graph.nodes.length) {
    wrap.innerHTML = `<p class="hint" style="padding:8px">no KG for this doc yet — run Phase-2 first (Build KG button on the Graph tab)</p>`;
    return;
  }

  const anns = [...new Set(graph.edges.flatMap(e => e.data.annotators || []))].sort();
  if (!anns.length) {
    wrap.innerHTML = `<p class="hint" style="padding:8px">edges carry no annotator info</p>`;
    return;
  }

  // SHARED layout: positions computed once on the union graph (headless),
  // then re-used as a preset layout in every multiple so the same entity
  // sits at the same spot in every panel — that's what makes them comparable.
  const headless = cytoscape({
    headless: true,
    elements: [...graph.nodes, ...graph.edges],
  });
  headless.layout({
    name: "concentric", boundingBox: { x1: 0, y1: 0, w: 900, h: 700 },
    concentric: (n) => n.degree(), levelWidth: () => 2, minNodeSpacing: 24,
  }).run();
  const pos = {};
  headless.nodes().forEach(n => { pos[n.id()] = { ...n.position() }; });
  headless.destroy();

  wrap.innerHTML = "";
  const grid = document.createElement("div");
  grid.className = "mm-grid";
  grid.style.gridTemplateColumns = `repeat(${anns.length <= 2 ? anns.length : Math.ceil(anns.length / (anns.length > 4 ? 2 : 1)) <= 2 ? 2 : 3}, 1fr)`;
  if (anns.length === 3) grid.style.gridTemplateColumns = "repeat(3, 1fr)";
  wrap.appendChild(grid);

  for (const ann of anns) {
    const ownEdges = graph.edges.filter(e => (e.data.annotators || []).includes(ann));
    const ownNodeIds = new Set(ownEdges.flatMap(e => [e.data.source, e.data.target]));
    const cell = document.createElement("div");
    cell.className = "mm-cell";
    const headStats = `${ownEdges.length} rel · ${ownNodeIds.size}/${graph.nodes.length} ent`;
    cell.innerHTML = `<div class="mm-head"><b title="${esc(ann)}">${esc(trunc(ann, 24))}</b>
      <span class="hint mm-hover">${headStats}</span></div>
      <div class="mm-cy"></div>`;
    grid.appendChild(cell);

    const nodes = graph.nodes.map(n => ({
      data: { ...n.data }, position: pos[n.data.id],
      classes: ownNodeIds.has(n.data.id) ? "" : "ghost",
    }));
    const edges = ownEdges.map(e => ({ data: { ...e.data } }));

    const cy = cytoscape({
      container: cell.querySelector(".mm-cy"),
      elements: [...nodes, ...edges],
      layout: { name: "preset", fit: false },
      minZoom: 0.05, maxZoom: 2.5, wheelSensitivity: 0.3,
      style: [
        { selector: "node", style: {
            "background-color": "#e2e8f0", "border-width": 1.5, "border-color": "#94a3b8",
            "width": 20, "height": 20, "label": "" }},
        { selector: "node[conflict_label = 'contradiction']", style: { "border-color": "#d32f2f", "border-width": 3 }},
        { selector: "node[conflict_label = 'granularity']",   style: { "border-color": "#f57c00", "border-width": 3 }},
        { selector: "node[conflict_label = 'redundancy']",    style: { "border-color": "#1976d2", "border-width": 2 }},
        { selector: "node.ghost", style: { "opacity": 0.13 }},
        { selector: "node.mm-sel", style: {
            "background-color": "#fef08a", "border-color": "#eab308", "border-width": 4,
            "label": "data(label)", "font-size": 10, "text-wrap": "wrap", "text-max-width": 110,
            "text-outline-color": "#fff", "text-outline-width": 2, "z-index": 99 }},
        { selector: "edge", style: {
            "width": 1.6, "line-color": "#cbd5e1", "curve-style": "straight",
            "target-arrow-shape": "triangle", "arrow-scale": 0.7, "target-arrow-color": "#cbd5e1" }},
        { selector: "edge[conflict_label = 'contradiction']", style: { "line-color": "#d32f2f", "target-arrow-color": "#d32f2f", "width": 2.4 }},
        { selector: "edge[conflict_label = 'granularity']",   style: { "line-color": "#f57c00", "target-arrow-color": "#f57c00", "width": 2.2 }},
        { selector: "edge[conflict_label = 'redundancy']",    style: { "line-color": "#1976d2", "target-arrow-color": "#1976d2", "width": 2 }},
      ],
    });
    cy.fit(undefined, 12);
    // Hover affordance: canvas nodes have no DOM tooltip, so surface the
    // hovered entity's label (and which annotators use it) in the panel head.
    const hoverEl = cell.querySelector(".mm-hover");
    cy.on("mouseover", "node", (evt) => {
      const d = evt.target.data();
      hoverEl.textContent = `${d.label} · ${(d.annotators || []).length} annot`;
      hoverEl.classList.add("mm-hover-active");
    });
    cy.on("mouseout", "node", () => {
      hoverEl.textContent = headStats;
      hoverEl.classList.remove("mm-hover-active");
    });
    cy.on("tap", "node", (evt) => {
      const id = evt.target.id();
      for (const c of A.miniCys) {
        c.nodes().removeClass("mm-sel");
        c.getElementById(id).addClass("mm-sel");
      }
      // Mirror the selection onto the main graph too.
      try {
        if (window.__cyInstance) {
          window.__cyInstance.elements().removeClass("selected");
          window.__cyInstance.getElementById(id).addClass("selected");
        }
      } catch {}
    });
    A.miniCys.push(cy);
  }
}
})();
