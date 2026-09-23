/* ============================================================
 * workflow.js — closes the Phase-4 loop (append-only module).
 *
 * Experiment tab:
 *   one click = Phase-1 (versioned facts root) → Phase-2 (versioned
 *   conflicts) → auto-opened v1↔vX comparison (corpus + current doc).
 *   Replaces the manual 3-tab relay (Guidelines → Background runs →
 *   Compare). Also renders the guideline-evidence panel built from
 *   conflict reviews.
 *
 * Review tab:
 *   a queue of aligned pairs for the current doc. Each review attributes
 *   the disagreement to specific guideline rule(s) + a note; the tally
 *   (/api/review_summary) is the empirical basis for authoring v2 —
 *   "rule §6 implicated in 12 granularity conflicts" beats memory.
 *
 * Talks to app.js only through the DOM (#compare-overlay reuse,
 * drawer-tab buttons) — no shared JS state.
 * ============================================================ */
(() => {
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);
const esc = (s) => String(s ?? "")
  .replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")
  .replaceAll('"',"&quot;").replaceAll("'","&#39;");
const trunc = (s, n) => (s && s.length > n ? s.slice(0, n - 1) + "…" : (s || ""));

const W = {
  doc: null,
  pairs: [],            // aligned pairs for current doc
  factsById: new Map(),
  reviews: {},          // pair_key -> record
  rules: [],            // [{id, title}]
  rulesVersion: "v1",
  selKey: null,
  expJobId: null,
  expVersion: null,
  expPollTimer: null,
};

const pairKey = (p) => `${p.fact_a_id || "-"}|${p.fact_b_id || "-"}`;

// ------------------------------------------------------------
// Init
// ------------------------------------------------------------

let _initialised = false;
function init() {
  if (_initialised) return;
  _initialised = true;

  for (const tab of $$("#drawer-tabs .tab")) {
    tab.addEventListener("click", () => {
      if (tab.dataset.tab === "review") { ensureDrawerHeight(420); loadReview(); }
      if (tab.dataset.tab === "experiment") { ensureDrawerHeight(360); loadExperiment(); }
    });
  }

  $("#exp-run")?.addEventListener("click", runExperiment);
  $("#exp-compare")?.addEventListener("click", () => {
    // Survive page reloads: fall back to the currently selected guideline
    // (results live on disk, not in this tab's memory).
    const v = W.expVersion || $("#exp-guideline")?.value;
    if (v && v !== "v1") openExperimentCompare(v);
  });
  $("#exp-edit-guideline")?.addEventListener("click", () => {
    const v = $("#exp-guideline")?.value;
    // "guideline-manager" has not been a tab key since the drawer was
    // reorganised — the guideline editor lives under "annotation" — so this
    // silently switched to nothing and then set a select the user could not see.
    window.goToTab?.("change-log");
    setTimeout(() => {
      const gm = $("#gm-select");
      if (gm && v) { gm.value = v; $("#gm-load")?.click(); }
    }, 250);
  });
  $("#evidence-refresh")?.addEventListener("click", loadEvidence);

  $("#review-label-filter")?.addEventListener("change", renderReviewList);
  $("#review-unreviewed-only")?.addEventListener("change", renderReviewList);

  // Heatmap "review →" cross-link.
  window.addEventListener("ava:open-review", async (e) => {
    window.goToTab?.("review");
    await loadReview(true);
    $("#review-unreviewed-only").checked = false;
    renderReviewList();
    selectPair(e.detail.pairKey);
  });

  // Doc switches invalidate the review cache.
  setInterval(() => {
    const cur = $("#doc-select")?.value || null;
    if (cur && cur !== W.doc) {
      W.doc = cur; W.pairs = []; W.selKey = null;
      if (document.querySelector('.tab-pane[data-tab="review"]')?.classList.contains("active")) {
        loadReview(true);
      }
    }
  }, 800);
}
document.addEventListener("DOMContentLoaded", init);
if (document.readyState !== "loading") init();

// ------------------------------------------------------------
// Experiment tab
// ------------------------------------------------------------

async function loadExperiment() {
  try {
    const data = await fetch("/api/guidelines").then(r => r.json());
    const sel = $("#exp-guideline");
    const cur = sel.value;
    sel.innerHTML = "";
    for (const g of data.guidelines || []) {
      const opt = document.createElement("option");
      opt.value = g.version; opt.textContent = g.version;
      sel.appendChild(opt);
    }
    // Default to the newest non-v1 version (that's the one being tested).
    // Prefer clean human-named versions (v2, v3) over machine-suffixed
    // job artifacts (v2_<hash>) which the worker writes for ad-hoc runs.
    const vers = (data.guidelines || []).map(g => g.version);
    const clean = vers.filter(v => v !== "v1" && !/^v\d+_[0-9a-f]{6,}$/.test(v));
    sel.value = cur && vers.includes(cur) ? cur
      : (clean.pop() || vers.filter(v => v !== "v1").pop() || "v1");
    // Compare is meaningful whenever a non-v1 version is selected — the
    // endpoint explains itself when no results exist yet.
    const cmp = $("#exp-compare");
    if (cmp) cmp.disabled = !sel.value || sel.value === "v1";
    sel.addEventListener("change", () => {
      if (cmp) cmp.disabled = !sel.value || sel.value === "v1";
    });
  } catch {}
  loadEvidence();
}

async function runExperiment() {
  const version = $("#exp-guideline").value;
  const models = ($("#exp-models").value || "").split(",").map(s => s.trim()).filter(Boolean);
  const docsRaw = ($("#exp-docs").value || "").split(",").map(s => s.trim()).filter(Boolean);
  const skip_extract = $("#exp-skip-extract").checked;
  const skip_layer2 = $("#exp-skip-layer2").checked;
  // the arbitrator of record; the tab has no picker, and the server's default
  // for an unnamed model used to be the retired local 4B qwen
  const layer2_model = window.__vaRecord?.layer2 || "";
  const align_threshold = parseFloat($("#exp-align").value) || 0.78;

  if (!version) { setStatus("#exp-status", "pick a guideline version", "err"); return; }
  if (version === "v1" && !confirm(
      "You picked v1 — the BASELINE. The server refuses every cell of the v1 run of record\n" +
      "(data/facts/manifest.json) and the baseline conflicts; only documents outside it\n" +
      "(e.g. uploaded text) can run. For the v1→vX experiment pick a draft version.\n" +
      "Proceed with v1 anyway?")) return;
  if (!models.length && !skip_extract) { setStatus("#exp-status", "enter at least one model", "err"); return; }

  const ok = confirm(
    "Closed-loop experiment:\n" +
    "  guideline:  " + version + "\n" +
    "  models:     " + (skip_extract ? "(skipped — reuse facts on disk)" : models.join(", ")) + "\n" +
    "  docs:       " + (docsRaw.length ? docsRaw.join(", ") : "all docs with a baseline Phase-2 result") + "\n" +
    "  layer-2:    " + (skip_layer2 ? "STUB (labels not arbitrated)" : layer2_model) + "\n\n" +
    (skip_extract ? "" : "Extraction needs Ollama; each (model × doc) cell ≈ 3–5 min on the 6 GB GPU.\n") +
    "Phase-2 runs automatically afterwards, then the v1↔" + version + " comparison opens.\n\nProceed?");
  if (!ok) return;

  setStatus("#exp-status", "enqueueing…", "");
  const body = { guideline_version: version, models, skip_extract, skip_layer2,
                 align_threshold, layer2_model };
  if (docsRaw.length) body.doc_ids = docsRaw;
  try {
    const r = await fetch("/api/run_pipeline", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const t = await r.text();
      setStatus("#exp-status", `failed: ${r.status} ${t.slice(0, 140)}`, "err");
      return;
    }
    const job = await r.json();
    W.expJobId = job.job_id;
    W.expVersion = version;
    $("#exp-compare").disabled = true;
    setStatus("#exp-status", `running · job ${job.job_id}`, "");
    pollExperiment();
  } catch (e) {
    setStatus("#exp-status", "request failed: " + e.message, "err");
  }
}

function pollExperiment() {
  clearTimeout(W.expPollTimer);
  const tick = async () => {
    if (!W.expJobId) return;
    let j;
    try {
      j = await fetch(`/api/jobs/${W.expJobId}`).then(r => r.json());
    } catch { W.expPollTimer = setTimeout(tick, 4000); return; }
    const ex = j.results.filter(r => r.step === "extract");
    const p2 = j.results.filter(r => r.step === "phase2");
    const detail = j.progress.detail ? ` · ${j.progress.detail}` : "";
    const stepTxt = `extract ${ex.length} cells · phase-2 ${p2.length} docs · ${j.progress.done}/${j.progress.total}${detail}`;
    if (j.status === "done") {
      setStatus("#exp-status", `✔ done — ${stepTxt}`, "ok");
      $("#exp-compare").disabled = false;
      refreshVariantOptions();
      openExperimentCompare(W.expVersion);
      return;
    }
    if (j.status === "failed") {
      setStatus("#exp-status", `✘ ${j.error || "failed"} — ${stepTxt}`, "err");
      const okSteps = p2.some(r => r.status === "ok");
      if (okSteps) { $("#exp-compare").disabled = false; refreshVariantOptions(); }
      return;
    }
    setStatus("#exp-status", `${j.status} · ${stepTxt}`, "");
    W.expPollTimer = setTimeout(tick, 2500);
  };
  tick();
}

async function refreshVariantOptions() {
  // New conflicts variants exist now — refresh the top-bar dropdown in place.
  const vsel = $("#variant-select");
  const docId = $("#doc-select")?.value;
  if (!vsel || !docId) return;
  try {
    const docs = (await fetch("/api/docs").then(r => r.json())).docs || [];
    const meta = docs.find(d => d.doc_id === docId);
    const variants = (meta || {}).variants || [];
    const cur = vsel.value;
    vsel.innerHTML = `<option value="">v1 (baseline)</option>` +
      variants.map(v => `<option value="${esc(v)}">${esc(v)}</option>`).join("");
    vsel.value = variants.includes(cur) ? cur : "";
    vsel.style.display = variants.length ? "" : "none";
  } catch {}
}

async function openExperimentCompare(version) {
  const body = $("#compare-body");
  if (!body) return;
  body.innerHTML = "<em>loading…</em>";
  $("#compare-overlay").classList.remove("hidden");
  const docId = $("#doc-select")?.value;
  let agg = null, doc = null;
  try { agg = await fetch(`/api/distribution_shift_agg?v2=${encodeURIComponent(version)}`).then(r => r.json()); } catch {}
  try { doc = await fetch(`/api/distribution_shift/${docId}?v2=${encodeURIComponent(version)}`).then(r => r.json()); } catch {}

  body.innerHTML = "";
  const section = (title, d) => {
    const div = document.createElement("div");
    div.className = "compare-section";
    if (!d || d.note) {
      div.innerHTML = `<h4>${esc(title)}</h4><p class="hint">${esc(d?.note || "no data")}</p>`;
      return div;
    }
    let warn = "";
    if (d.annotators_match === false) {
      warn = `<div class="compare-warn">⚠ Annotator sets differ — v1: [${esc((d.v1_annotators || []).join(", "))}]
        vs ${esc(version)}: [${esc((d.v2_annotators || []).join(", "))}].
        Pair counts are <b>not comparable</b>: fewer annotator pairs mechanically yields fewer
        aligned pairs. Re-run the baseline with the same model set before reading these deltas.</div>`;
    }
    const max = Math.max(1, ...d.v1, ...d.v2);
    let bars = `<h4>${esc(title)}</h4>${warn}<div class="compare-bars">`;
    for (let i = 0; i < d.labels.length; i++) {
      const lbl = d.labels[i], v1 = d.v1[i], v2 = d.v2[i], dpct = d.delta_pct[i];
      bars += `
        <div class="lab"><span class="lbl ${esc(lbl)}" style="padding:1px 6px;border-radius:3px;">${esc(lbl)}</span></div>
        <div class="bar-track" title="v1=${v1}  ${esc(version)}=${v2}">
          <div class="bar-v1" style="width:${(v1 / max * 100).toFixed(1)}%"></div>
          <div class="bar-v2" style="width:${(v2 / max * 100).toFixed(1)}%"></div>
        </div>
        <div class="pct ${dpct == null ? "" : dpct < 0 ? "dn" : "up"}">${dpct == null ? "—" : (dpct > 0 ? "+" : "") + dpct + "%"}</div>`;
    }
    bars += `</div>
      <div class="compare-summary"><div class="row">
        <b>Total:</b> ${d.totals.v1} → ${d.totals.v2}
        (${d.totals.delta_pct == null ? "—" : (d.totals.delta_pct > 0 ? "+" : "") + d.totals.delta_pct + "%"})
        ${d.layer1_rate && d.layer1_rate.v1 != null && typeof d.layer1_rate.v1 !== "object"
          ? ` · <b>L1 filter rate:</b> ${d.layer1_rate.v1} → ${d.layer1_rate.v2}` : ""}
        ${d.n_docs != null ? ` · <b>${d.n_docs} doc(s)</b>` : ""}
      </div></div>`;
    div.innerHTML = bars;
    return div;
  };
  const head = document.createElement("p");
  head.className = "hint";
  head.innerHTML = `Distribution shift <b>v1 → ${esc(version)}</b> · grey bar = v1, blue bar = ${esc(version)}.
    Negative deltas on contradiction/granularity = convergence (the thesis metric).`;
  body.appendChild(head);
  body.appendChild(section(`Corpus aggregate`, agg));
  body.appendChild(section(`Current doc · ${docId}`, doc));
}

// ------------------------------------------------------------
// Evidence panel (review_summary)
// ------------------------------------------------------------

async function loadEvidence() {
  const el = $("#evidence-body");
  if (!el) return;
  let d;
  try { d = await fetch("/api/review_summary?version=v1").then(r => r.json()); }
  catch { return; }
  if (!d.n_reviews) {
    el.innerHTML = `<p class="hint">No reviews yet. Use the <b>Review</b> tab to attribute
      conflicts to guideline rules; the tally appears here to guide the v2 revision.</p>`;
    return;
  }
  const maxN = Math.max(1, ...d.rules.map(r => r.n_conflicts));
  let html = `<div class="ev-meta">${d.n_reviews} reviews · ${d.n_docs} doc(s) ·
    ${d.n_unattributed} unattributed</div>`;
  for (const r of d.rules) {
    const chips = Object.entries(r.by_label)
      .map(([l, n]) => `<span class="lbl ${esc(l)}" title="${esc(l)}">${n}</span>`).join(" ");
    const notes = (r.sample_notes || [])
      .map(n => `<li title="${esc(n.doc_id)}">${esc(n.note)}</li>`).join("");
    html += `
      <details class="ev-rule">
        <summary>
          <span class="ev-id">§${esc(r.id)}</span>
          <span class="ev-title" title="${esc(r.title)}">${esc(trunc(r.title, 46))}</span>
          <span class="st-track sm"><span class="st-fill" style="width:${(r.n_conflicts / maxN * 100).toFixed(0)}%"></span></span>
          <b>${r.n_conflicts}×</b> ${chips}
        </summary>
        ${notes ? `<ul class="ev-notes">${notes}</ul>` : ""}
      </details>`;
  }
  el.innerHTML = html;
}

// ------------------------------------------------------------
// Review tab
// ------------------------------------------------------------

async function loadReview(force = false) {
  const docId = $("#doc-select")?.value;
  if (!docId) return;
  if (!force && W.doc === docId && W.pairs.length) { renderReviewList(); return; }
  W.doc = docId;
  $("#review-list").innerHTML = `<p class="hint">loading…</p>`;
  let pairs, facts, reviews;
  try {
    [pairs, facts, reviews] = await Promise.all([
      fetch(`/api/pairs/${docId}`).then(r => r.json()),
      fetch(`/api/facts/${docId}`).then(r => r.json()),
      fetch(`/api/review/${docId}`).then(r => r.json()),
    ]);
  } catch (e) {
    $("#review-list").innerHTML = `<p class="hint">failed: ${esc(e.message)}</p>`;
    return;
  }
  W.pairs = pairs || [];
  W.factsById = new Map((facts.facts || []).map(f => [f.fact_id, f]));
  W.reviews = reviews.reviews || {};

  const anyFact = (facts.facts || [])[0];
  W.rulesVersion = (anyFact && anyFact.guideline_version) || "v1";
  try {
    const rd = await fetch(`/api/guideline_rules?version=${encodeURIComponent(W.rulesVersion)}`).then(r => r.json());
    W.rules = rd.rules || [];
    if (!W.rules.length && W.rulesVersion !== "v1") {
      const rd1 = await fetch(`/api/guideline_rules?version=v1`).then(r => r.json());
      W.rules = rd1.rules || []; W.rulesVersion = "v1";
    }
  } catch { W.rules = []; }

  renderReviewList();
}

const SEVERITY = { contradiction: 0, granularity: 1, redundancy: 2, unlabeled: 3, no_conflict: 4 };

function filteredPairs() {
  const labelFilter = $("#review-label-filter")?.value || "";
  const unreviewedOnly = $("#review-unreviewed-only")?.checked;
  let rows = W.pairs.filter(p => p.status === "matched" || labelFilter === "orphan" || labelFilter === "");
  rows = rows.filter(p => {
    if (labelFilter === "orphan") return p.status !== "matched";
    if (labelFilter) return p.status === "matched" && p.conflict_label === labelFilter;
    return true;
  });
  if (unreviewedOnly) rows = rows.filter(p => !W.reviews[pairKey(p)]);
  rows.sort((x, y) =>
    (SEVERITY[x.conflict_label] ?? 9) - (SEVERITY[y.conflict_label] ?? 9)
    || (y.cosine || 0) - (x.cosine || 0));
  return rows;
}

function renderReviewList() {
  const list = $("#review-list");
  if (!list) return;
  const rows = filteredPairs();
  const nReviewed = Object.keys(W.reviews).length;
  $("#review-progress").textContent = `reviewed ${nReviewed}/${W.pairs.length} · showing ${rows.length}`;

  if (!rows.length) {
    list.innerHTML = `<p class="hint">nothing matches the filter${nReviewed ? " — all reviewed 🎉" : ""}</p>`;
    return;
  }
  list.innerHTML = "";
  for (const p of rows) {
    const key = pairKey(p);
    const fa = W.factsById.get(p.fact_a_id);
    const fb = W.factsById.get(p.fact_b_id);
    const reviewed = W.reviews[key];
    const lbl = p.status === "matched" ? (p.conflict_label || "unlabeled") : "orphan";
    const row = document.createElement("div");
    row.className = "rv-item" + (key === W.selKey ? " sel" : "") + (reviewed ? " reviewed" : "");
    row.dataset.key = key;
    row.innerHTML = `
      <span class="lbl ${esc(lbl)}">${esc(lbl[0].toUpperCase())}</span>
      <span class="rv-cos">${(p.cosine ?? 0).toFixed(2)}</span>
      <span class="rv-txt" title="${esc((fa?.natural_language || "") + " ⟷ " + (fb?.natural_language || ""))}">
        ${esc(trunc(fa?.natural_language || fb?.natural_language || "(missing)", 76))}</span>
      ${reviewed ? `<span class="rv-check" title="reviewed: rules ${esc((reviewed.rules || []).map(r => "§" + r).join(", ") || "—")}">✓</span>` : ""}`;
    row.addEventListener("click", () => selectPair(key));
    list.appendChild(row);
  }
}

function selectPair(key) {
  W.selKey = key;
  for (const el of $$("#review-list .rv-item")) {
    el.classList.toggle("sel", el.dataset.key === key);
  }
  const p = W.pairs.find(x => pairKey(x) === key);
  if (!p) return;
  renderReviewDetail(p);
}

function renderReviewDetail(p) {
  const det = $("#review-detail");
  const key = pairKey(p);
  const fa = W.factsById.get(p.fact_a_id);
  const fb = W.factsById.get(p.fact_b_id);
  const existing = W.reviews[key] || {};
  const lbl = p.status === "matched" ? (p.conflict_label || "unlabeled") : "orphan";

  const factCard = (side, ann, f) => f ? `
    <div class="rv-fact">
      <div class="ann-tag">${esc(side)} · ${esc(ann)} · ${esc((f.source_locator || {}).section_path || "")}</div>
      <div class="nl">${esc(f.natural_language || "")}</div>
      <div class="spo">⟨${esc(f.subject)} · ${esc(f.predicate)} · ${esc(f.object)}⟩</div>
      <div class="quote">"${esc((f.source_locator || {}).quote || "")}"</div>
    </div>` : `<div class="rv-fact missing"><div class="ann-tag">${esc(side)} · ${esc(ann)}</div>
      <div class="hint">no counterpart — orphan side</div></div>`;

  const searchQ = encodeURIComponent([fa?.subject, fa?.predicate, fa?.object]
    .filter(Boolean).join(" ") || fb?.natural_language || "");
  const ruleBoxes = W.rules.map(r => `
    <label class="rv-rule" title="${esc(r.title)}">
      <input type="checkbox" value="${esc(r.id)}"
        ${(existing.rules || []).includes(r.id) ? "checked" : ""}>
      §${esc(r.id)} ${esc(trunc(r.title, 34))}
    </label>`).join("");

  const res = existing.resolution || "agree";
  det.innerHTML = `
    <div class="rv-head">
      <span class="lbl ${esc(lbl)}">${esc(lbl)}</span>
      cos <b>${(p.cosine ?? 0).toFixed(3)}</b>
      <span class="hint">${esc(p.annotator_a)} ↔ ${esc(p.annotator_b)}</span>
      <a class="rv-verify" target="_blank" rel="noopener"
         href="https://www.google.com/search?q=${searchQ}" title="In-tool verification: web search pre-populated with S/P/O">🔍 verify</a>
      <button id="rv-cite" class="tool-btn"
        title="Cite this pair in the guideline change log (Annotation manager tab)">cite →</button>
    </div>
    <div class="rv-pair">${factCard("A", p.annotator_a, fa)}${factCard("B", p.annotator_b, fb)}</div>
    ${p.layer1_reason || p.layer2_reason
      ? `<div class="reasons">L1: ${esc(p.layer1_reason || "—")} · L2: ${esc(p.layer2_reason || "—")}</div>` : ""}
    <div class="rv-form">
      <div class="rv-rules-head">Which guideline rule(s) under-specify this case? <span class="hint">(${esc(W.rulesVersion)})</span></div>
      <div class="rv-rules">${ruleBoxes || '<span class="hint">no parsable rules in guideline</span>'}</div>
      <div class="rv-resolution">
        <label><input type="radio" name="rv-res" value="agree" ${res === "agree" ? "checked" : ""}> label correct</label>
        <label><input type="radio" name="rv-res" value="relabel" ${res.startsWith("relabel") ? "checked" : ""}> relabel as
          <select id="rv-relabel">
            ${["contradiction", "granularity", "redundancy", "no_conflict"].map(l =>
              `<option value="${l}" ${res === "relabel:" + l ? "selected" : ""}>${l}</option>`).join("")}
          </select></label>
        <label><input type="radio" name="rv-res" value="dismiss" ${res === "dismiss" ? "checked" : ""}> dismiss (noise)</label>
      </div>
      <textarea id="rv-note" rows="2" placeholder="note — e.g. 'demonstrative this Decision not resolved; rule §2 lacks an example for legal acts'">${esc(existing.note || "")}</textarea>
      <div class="row">
        <button id="rv-save">Save &amp; next</button>
        ${W.reviews[key] ? '<button id="rv-delete" class="ghost" title="Remove this review">remove</button>' : ""}
        <span id="rv-status"></span>
      </div>
    </div>`;

  $("#rv-save").addEventListener("click", () => saveReview(p));
  $("#rv-delete")?.addEventListener("click", () => deleteReview(p));
  // The diagnosed arm's citation: what the interface showed, in the form the
  // change log stores it (src/guideline_log.CITE_FIELDS).
  $("#rv-cite")?.addEventListener("click", (ev) => {
    if (!window.GLog) return;
    const n = window.GLog.cite({
      kind: "pair", doc_id: W.doc, pair_key: key, label: lbl,
      annotator: `${p.annotator_a} ↔ ${p.annotator_b}`,
      section: (fa || fb || {}).source_locator?.section_path || "",
      excerpt: (fa?.natural_language || fb?.natural_language || "").slice(0, 240),
      version: window.__variant || "v1",
    });
    ev.target.textContent = `cited (${n})`;
    setTimeout(() => { ev.target.textContent = "cite →"; }, 1500);
  });
}

async function saveReview(p) {
  const key = pairKey(p);
  const rules = [...$$("#review-detail .rv-rules input:checked")].map(c => c.value);
  let resolution = document.querySelector('input[name="rv-res"]:checked')?.value || "agree";
  if (resolution === "relabel") resolution = "relabel:" + ($("#rv-relabel")?.value || "no_conflict");
  const payload = {
    pair_key: key, rules,
    note: $("#rv-note")?.value || "",
    resolution,
    label_at_review: p.status === "matched" ? (p.conflict_label || "unlabeled") : "orphan",
    annotators: [p.annotator_a, p.annotator_b],
  };
  try {
    const r = await fetch(`/api/review/${W.doc}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const d = await r.json();
    W.reviews[key] = d.record;
    setStatus("#rv-status", "saved ✓", "ok");
    renderReviewList();
    loadEvidence();
    // Auto-advance to the next unreviewed pair in the filtered order.
    const next = filteredPairs().find(x => !W.reviews[pairKey(x)]);
    if (next) selectPair(pairKey(next));
  } catch (e) {
    setStatus("#rv-status", "save failed: " + e.message, "err");
  }
}

async function deleteReview(p) {
  const key = pairKey(p);
  try {
    await fetch(`/api/review/${W.doc}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pair_key: key, delete: true }),
    });
    delete W.reviews[key];
    renderReviewList();
    selectPair(key);
    loadEvidence();
  } catch {}
}

// ------------------------------------------------------------

function ensureDrawerHeight(px) {
  // Review/Experiment need vertical room; grow the drawer once (never shrink
  // a user-chosen height). Mirrors what dragging the hsplitter does.
  const bar = document.getElementById("bottombar");
  if (!bar) return;
  if (bar.getBoundingClientRect().height < px) {
    bar.style.height = px + "px";
    bar.style.maxHeight = "60vh";
  }
}

// ------------------------------------------------------------
function setStatus(sel, text, cls) {
  const el = $(sel);
  if (!el) return;
  el.textContent = text;
  el.className = cls || "";
}
})();
