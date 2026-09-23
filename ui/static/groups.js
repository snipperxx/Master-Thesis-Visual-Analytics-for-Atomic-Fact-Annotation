/* ============================================================
 * groups.js — the GROUP layer view (append-only module).
 *
 * Groups, not pairs, are the unit of analysis (method.tex), and a
 * one-to-many group is the only place a granularity split is
 * representable at all: a pair is 1:1 by construction, so the
 * existing Graph / Heatmap / facts views cannot show "one annotator
 * wrote one fact where the other wrote three". This adds a fifth
 * centre-pane tab that does.
 *
 * Reads GET /api/groups/<doc> (variant-aware via variant_shim.js).
 * Talks to app.js ONLY by clicking its own .fact-item rows, which is
 * app.js's real selection entry point — so clicking a group here
 * highlights the source span and the KG node exactly as a click in
 * the facts table would.
 *
 * Owns nothing else: its styles are injected below rather than added
 * to app.css, and analysis.js's setView() keeps owning the other four
 * containers.
 * ============================================================ */
(() => {
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);
const esc = (s) => String(s ?? "")
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;").replaceAll("'", "&#39;");

const LABEL_COLOR = {
  contradiction: "#d32f2f", granularity: "#f57c00",
  redundancy: "#1976d2", no_conflict: "#16a34a",
  escalate: "#7c3aed", unlabeled: "#94a3b8",
};
const SHAPE_LABEL = {
  one_to_many: "1 : n", one_to_one: "1 : 1", orphan: "orphan",
};

const G = {
  doc: null,
  key: null,          // doc|variant of the rendered payload
  data: null,
  shape: "",          // filter
  label: "",          // filter
  disputedOnly: false,
  hideSchemaFanout: false,
};

// ------------------------------------------------------------
// Styles (kept here so app.css stays untouched)
// ------------------------------------------------------------

const STYLE = `
#groups-wrap { overflow:auto; height:100%; padding:8px 10px; font-size:12px; }
#groups-wrap .gsum { display:flex; flex-wrap:wrap; gap:14px; align-items:flex-start;
  border-bottom:1px solid #e2e8f0; padding-bottom:8px; margin-bottom:8px; }
#groups-wrap .gsum h4 { margin:0 0 4px; font-size:11px; text-transform:uppercase;
  letter-spacing:.04em; color:#64748b; font-weight:600; }
#groups-wrap .gparams { color:#475569; line-height:1.6; }
#groups-wrap .gparams code { background:#f1f5f9; padding:1px 4px; border-radius:3px; }
#groups-wrap .gbars { min-width:200px; }
#groups-wrap .gbar { display:flex; align-items:center; gap:6px; margin:2px 0; }
#groups-wrap .gbar .n { width:34px; text-align:right; color:#475569; font-variant-numeric:tabular-nums; }
#groups-wrap .gbar .fill { height:9px; border-radius:2px; }
#groups-wrap .gbar .k { color:#334155; }
#groups-wrap .gfilters { display:flex; gap:8px; align-items:center; flex-wrap:wrap;
  margin-bottom:8px; }
#groups-wrap .gfilters label { color:#475569; }
#groups-wrap .gcard { border:1px solid #e2e8f0; border-left-width:3px; border-radius:4px;
  padding:6px 8px; margin-bottom:6px; cursor:pointer; background:#fff; }
#groups-wrap .gcard:hover { background:#f8fafc; }
#groups-wrap .gcard.sel { background:#eff6ff; border-color:#93c5fd; }
#groups-wrap .ghead { display:flex; gap:6px; align-items:center; flex-wrap:wrap;
  margin-bottom:4px; }
#groups-wrap .badge { font-size:10px; padding:1px 6px; border-radius:9px; color:#fff;
  font-weight:600; letter-spacing:.02em; }
#groups-wrap .shape { background:#334155; font-variant-numeric:tabular-nums; }
#groups-wrap .src { background:#e2e8f0; color:#475569; }
#groups-wrap .schema { background:#fde68a; color:#78350f; }
#groups-wrap .cos { color:#64748b; font-variant-numeric:tabular-nums; }
#groups-wrap .pair { color:#94a3b8; }
#groups-wrap .was { color:#94a3b8; text-decoration:line-through; }
#groups-wrap .side { display:flex; gap:6px; margin:2px 0; }
#groups-wrap .side .who { flex:0 0 130px; color:#64748b; text-align:right;
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
#groups-wrap .side .what { flex:1; color:#0f172a; }
#groups-wrap .side .what ol { margin:0; padding-left:18px; }
#groups-wrap .side .what li { margin:1px 0; }
#groups-wrap .slots { color:#94a3b8; font-family:ui-monospace,monospace; font-size:11px; }
#groups-wrap .greason { color:#64748b; margin-top:3px; font-style:italic; }
#groups-wrap .gsec { color:#94a3b8; }
#groups-wrap .hint { color:#64748b; }
`;

function injectStyle() {
  if ($("#groups-style")) return;
  const el = document.createElement("style");
  el.id = "groups-style";
  el.textContent = STYLE;
  document.head.appendChild(el);
}

// ------------------------------------------------------------
// Tab + container wiring
//
// analysis.js owns setView() and hides the four containers it knows
// about, so a click on OUR tab already blanks the others. We only have
// to toggle #groups-wrap, on the bubble phase after its handler.
// ------------------------------------------------------------

function activeView() {
  return $("#center-tabs .ctab.active")?.dataset.view || "graph";
}

function syncVisibility() {
  const wrap = $("#groups-wrap");
  if (!wrap) return;
  const on = activeView() === "groups";
  wrap.classList.toggle("hidden", !on);
  if (on) render();
}

let _init = false;
function init() {
  if (_init) return;
  _init = true;
  injectStyle();

  document.addEventListener("click", (e) => {
    if (e.target.closest?.("#center-tabs .ctab")) setTimeout(syncVisibility, 0);
  });

  // app.js can switch docs without firing a change event on #doc-select
  // (same reason analysis.js polls), so poll for a doc/variant change and
  // re-render only when the view is actually showing.
  setInterval(() => {
    const doc = $("#doc-select")?.value || null;
    const key = `${doc}|${window.__variant || ""}`;
    if (key !== G.key && activeView() === "groups") render();
    G.doc = doc;
  }, 800);
}
document.addEventListener("DOMContentLoaded", init);
if (document.readyState !== "loading") init();

// ------------------------------------------------------------
// Render
// ------------------------------------------------------------

async function render() {
  const wrap = $("#groups-wrap");
  const doc = $("#doc-select")?.value;
  if (!wrap || !doc) return;

  const key = `${doc}|${window.__variant || ""}`;
  if (key !== G.key) {
    wrap.innerHTML = `<p class="hint">loading groups…</p>`;
    try {
      G.data = await fetch(`/api/groups/${encodeURIComponent(doc)}`).then(r => r.json());
    } catch (err) {
      wrap.innerHTML = `<p class="hint">could not load groups: ${esc(err.message)}</p>`;
      return;
    }
    G.key = key;
  }
  paint();
}

function paint() {
  const wrap = $("#groups-wrap");
  const d = G.data;
  if (!d) return;

  if (!d.available) {
    wrap.innerHTML = `<p class="hint">No group layer for this document.<br>
      Groups are written by Phase-2 — run it from the <b>Runs</b> tab
      (or <code>python -m scripts.run_phase2 --doc ${esc(d.doc_id || "")}</code>).
      Conflict files written before the group layer existed have pairs only.</p>`;
    return;
  }

  const groups = (d.groups || []).filter(g =>
    (!G.shape || g.shape === G.shape) &&
    (!G.label || g.conflict_label === G.label) &&
    (!G.disputedOnly || overruled(g)) &&
    (!G.hideSchemaFanout || !g.partners_one_annotation));

  const labels = Object.keys(d.label_counts || {});
  wrap.innerHTML = `
    ${summaryHtml(d)}
    <div class="gfilters">
      <label>Shape
        <select id="gf-shape">
          <option value="">all</option>
          ${["one_to_many", "one_to_one", "orphan"].map(s =>
            `<option value="${s}" ${G.shape === s ? "selected" : ""}>${SHAPE_LABEL[s]}</option>`).join("")}
        </select>
      </label>
      <label>Label
        <select id="gf-label">
          <option value="">all</option>
          ${labels.map(l => `<option value="${esc(l)}" ${G.label === l ? "selected" : ""}>${esc(l)}</option>`).join("")}
        </select>
      </label>
      <label title="Groups where Layer-2 overruled the shape rule — only possible when Phase-2 ran with --arbitrate-shapes all">
        <input type="checkbox" id="gf-disputed" ${G.disputedOnly ? "checked" : ""}>
        shape ≠ model
      </label>
      <label title="Hide one-to-many groups whose whole fan-out sits inside ONE annotation of one annotator. That fan-out is the reification schema — one annotation, several triples, one shared natural_language — not one annotator splitting what another merged.">
        <input type="checkbox" id="gf-noschema" ${G.hideSchemaFanout ? "checked" : ""}>
        hide schema fan-out
      </label>
      <span class="hint">${groups.length} of ${d.n_total} groups</span>
    </div>
    ${groups.length ? groups.map(cardHtml).join("")
                    : `<p class="hint">No group matches this filter.</p>`}`;

  $("#gf-shape").addEventListener("change", e => { G.shape = e.target.value; paint(); });
  $("#gf-label").addEventListener("change", e => { G.label = e.target.value; paint(); });
  $("#gf-disputed").addEventListener("change", e => { G.disputedOnly = e.target.checked; paint(); });
  $("#gf-noschema").addEventListener("change", e => { G.hideSchemaFanout = e.target.checked; paint(); });

  for (const card of $$("#groups-wrap .gcard")) {
    card.addEventListener("click", () => selectGroup(card));
  }
}

// How much of the one-to-many rate is the reification schema. Stated on the
// summary because the 1:n percentage above it is the number a reader would
// otherwise take for a granularity rate.
function schemaNote(d) {
  const oneToMany = (d.groups || []).filter(g => g.shape === "one_to_many");
  if (!oneToMany.length) return "";
  const n = oneToMany.filter(g => g.partners_one_annotation).length;
  if (!n) return `<div class="hint" style="margin-top:4px">no schema fan-out</div>`;
  return `<div class="hint" style="margin-top:4px" title="One annotation = one natural_language + N triples; the triples embed identically, so they all link to the same partner.">
    ⚠ ${n} of ${oneToMany.length} (${Math.round(n / oneToMany.length * 100)}%) are
    <b>schema fan-out</b> — one annotation of one annotator, not a split</div>`;
}

function summaryHtml(d) {
  const p = d.params || {};
  const shapes = d.shapes || {};
  const total = Object.values(shapes).reduce((a, b) => a + b, 0) || 1;
  const l1 = d.label_counts_layer1 || {};

  const bars = ["one_to_many", "one_to_one", "orphan"].map(s => {
    const n = shapes[s] || 0;
    const pct = Math.round((n / total) * 100);
    return `<div class="gbar"><span class="n">${n}</span>
      <span class="fill" style="width:${Math.max(2, pct * 1.4)}px;background:${
        s === "one_to_many" ? "#f57c00" : s === "one_to_one" ? "#1976d2" : "#cbd5e1"}"></span>
      <span class="k">${SHAPE_LABEL[s]} · ${pct}%</span></div>`;
  }).join("");

  // Show Layer-1 -> final per label so the arbitrator's contribution is legible.
  const keys = [...new Set([...Object.keys(l1), ...Object.keys(d.label_counts || {})])];
  const labelRows = keys.map(k => {
    const before = l1[k] || 0, after = (d.label_counts || {})[k] || 0;
    const delta = after - before;
    return `<div class="gbar"><span class="n">${after}</span>
      <span class="fill" style="width:${Math.max(2, after * 2)}px;background:${LABEL_COLOR[k] || "#94a3b8"}"></span>
      <span class="k">${esc(k)}${delta ? ` <span class="hint">(${delta > 0 ? "+" : ""}${delta})</span>` : ""}</span></div>`;
  }).join("");

  const shapesArb = (p.arbitrate_shapes || []).join(", ") || "—";
  return `<div class="gsum">
    <div class="gparams">
      <h4>Alignment</h4>
      <code>${esc(p.alignment || "?")}</code> at τ=<code>${esc(p.align_threshold ?? "?")}</code>
      ${p.slot_gates ? " + slot gates" : ""}<br>
      <h4 style="margin-top:6px">Layer-2</h4>
      <code>${esc(p.layer2_model || "?")}</code>,
      ${d.layer2_calls} group call${d.layer2_calls === 1 ? "" : "s"}<br>
      judged shapes: <code>${esc(shapesArb)}</code>
    </div>
    <div class="gbars"><h4>Group shape</h4>${bars}${schemaNote(d)}</div>
    <div class="gbars"><h4>Label (Δ = Layer-2)</h4>${labelRows}</div>
  </div>`;
}

// A genuine disagreement between the shape / Layer-1 verdict and the model —
// NOT the ordinary case of Layer-2 resolving a routing state. `escalate` and
// `unlabeled` mean "nobody has decided yet", so replacing one of those is
// gap-filling, which is what Layer-2 is supposed to do on every call.
const ROUTING_STATES = new Set(["escalate", "unlabeled", ""]);

function overruled(g) {
  return g.label_source === "layer2" && !ROUTING_STATES.has(g.prior_label || "") &&
         g.prior_label !== g.conflict_label;
}

function factLine(f) {
  return esc(f.natural_language ||
             [f.subject, f.predicate, f.object].filter(Boolean).join(" "));
}

// Same trap as conflict_layer2._side_text: the annotators reuse one
// `natural_language` across facts whose slots differ, so a list of NL lines can
// show n identical rows for a real n-way split. Show the triple when the NL
// alone does not tell the partners apart — otherwise the view lies the same way
// the prompt did.
function partnerLines(partners) {
  const nls = partners.map(factLine);
  const ambiguous = new Set(nls).size < nls.length;
  return partners.map((p, i) => ambiguous
    ? `${nls[i]} <span class="slots">[${esc(p.subject || "")} | ${
        esc(p.predicate || "")} | ${esc(p.object || "")}]</span>`
    : nls[i]);
}

function cardHtml(g) {
  const color = LABEL_COLOR[g.conflict_label] || "#94a3b8";
  const anchor = g.anchor || {};
  const partners = g.partners || [];
  const many = partners.length > 1;

  const partnerHtml = partners.length === 0
    ? `<span class="hint">no partner above τ</span>`
    : many
      ? `<ol>${partnerLines(partners).map(l => `<li>${l}</li>`).join("")}</ol>`
      : factLine(partners[0]);

  // Show what the label was before Layer-2 whenever it changed — a struck-through
  // `escalate` says "the arbitrator resolved this", a struck-through
  // `granularity` says "the arbitrator disagreed with the shape rule".
  const priorHtml = (g.prior_label && g.prior_label !== g.conflict_label)
    ? `<span class="was" title="${overruled(g)
        ? "Layer-2 overruled the shape / Layer-1 verdict"
        : "label before Layer-2 (a routing state)"}">${esc(g.prior_label)}</span>` : "";

  return `<div class="gcard" style="border-left-color:${color}"
       data-anchor="${esc(anchor.fact_id || "")}">
    <div class="ghead">
      <span class="badge shape">${SHAPE_LABEL[g.shape] || esc(g.shape)}${
        g.fan_out > 1 ? ` ·${g.fan_out}` : ""}</span>
      <span class="badge" style="background:${color}">${esc(g.conflict_label)}</span>
      ${priorHtml}
      <span class="badge src">${esc(g.label_source || "—")}</span>
      ${g.partners_one_annotation ? `<span class="badge schema"
        title="Every partner is a triple of ONE annotation by one annotator (shared natural_language, so identical embeddings). This fan-out is the reification schema, not a granularity disagreement.">schema fan-out</span>` : ""}
      <span class="cos">cos ${Number(g.max_cosine ?? 0).toFixed(3)}</span>
      <span class="pair">${esc(g.annotator_a)} ↔ ${esc(g.annotator_b)}</span>
      <span class="gsec">${esc(anchor.section_path || "")}</span>
    </div>
    <div class="side"><span class="who">${esc(anchor.annotator || "anchor")}</span>
      <span class="what">${factLine(anchor)}</span></div>
    <div class="side"><span class="who">${esc((partners[0] || {}).annotator ||
      (g.anchor_side === "a" ? g.annotator_b : g.annotator_a))}</span>
      <span class="what">${partnerHtml}</span></div>
    ${g.reason ? `<div class="greason">${esc(g.reason)}</div>` : ""}
  </div>`;
}

// ------------------------------------------------------------
// Cross-view selection — reuse app.js's own entry point
// ------------------------------------------------------------

function selectGroup(card) {
  for (const c of $$("#groups-wrap .gcard")) c.classList.remove("sel");
  card.classList.add("sel");
  const id = card.dataset.anchor;
  if (!id) return;
  // Clicking app.js's row runs its selectFact(): source span + KG node + table.
  const row = document.querySelector(`#facts-table .fact-item[data-fact-id="${id}"]`);
  if (row) {
    row.click();
    row.scrollIntoView({ block: "center", behavior: "smooth" });
  } else if (window.__cyInstance) {
    // Fact filtered out of the table (annotator/label chips) — at least move
    // the graph selection so the click is not silently inert.
    try {
      window.__cyInstance.elements().removeClass("selected");
      window.__cyInstance.getElementById(id).addClass("selected");
    } catch {}
  }
}
})();
