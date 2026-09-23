/* ============================================================
 * neo4j.js — "Neo4j" drawer tab: custom conflict detection over the
 * reified KG (the proposal's Interactive Graph Constraints / patching).
 * Append-only, self-contained IIFE. Talks to app.js only through the DOM
 * (#doc-select) — no shared JS state.
 * ============================================================ */
(() => {
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
  const fmt = (v) => Array.isArray(v) ? v.join(", ") : (v == null ? "" : v);
  let constraints = {};
  let loaded = false;

  async function loadStatus() {
    const el = $("#neo4j-status");
    if (!el) return;
    el.className = "hint";
    try {
      const s = await fetch("/api/neo4j/status").then(r => r.json());
      if (!s.driver_installed) {
        el.textContent = "⚠ neo4j Python driver not installed — pip install neo4j";
        el.className = "hint warn"; return;
      }
      if (s.connected) {
        el.textContent = `✓ connected ${s.uri} · ${s.n_facts ?? "?"} :Fact · ${s.n_conflicts_with ?? 0} :CONFLICTS_WITH`;
        el.className = "hint ok";
      } else {
        el.textContent = `✗ not connected (${s.uri}) — ${s.message || ""}`;
        el.className = "hint warn";
      }
    } catch (e) { el.textContent = "status failed: " + e.message; }
  }

  async function loadConstraints() {
    const sel = $("#neo4j-constraint-select");
    if (!sel) return;
    try {
      const data = await fetch("/api/neo4j/constraints").then(r => r.json());
      constraints = data.constraints || {};
    } catch (e) { return; }
    sel.innerHTML = "";
    for (const [k, c] of Object.entries(constraints)) {
      const o = document.createElement("option");
      o.value = k; o.textContent = c.label || k;
      sel.appendChild(o);
    }
    applyConstraint();
  }

  function applyConstraint() {
    const c = constraints[$("#neo4j-constraint-select").value];
    if (!c) return;
    $("#neo4j-cypher").value = c.cypher || "";
    $("#neo4j-params").value = JSON.stringify(c.params || {});
    $("#neo4j-mode").value = c.mode || "report";
    const d = $("#neo4j-desc"); if (d) d.textContent = c.description || "";
  }

  async function exportDoc() {
    const doc = $("#doc-select")?.value;
    const st = $("#neo4j-export-status");
    if (!doc) { st.textContent = "open a doc first"; return; }
    st.textContent = "exporting…";
    try {
      const r = await fetch(`/api/neo4j/export/${doc}`, { method: "POST" }).then(x => x.json());
      st.textContent = r.ok
        ? "✓ " + Object.entries(r.summary || {}).map(([k, v]) => `${k}:${v}`).join(" ")
        : "✗ " + (r.message || "failed");
    } catch (e) { st.textContent = "✗ " + e.message; }
    loadStatus();
  }

  async function run() {
    const st = $("#neo4j-run-status"); const out = $("#neo4j-results");
    let params = {};
    try { params = JSON.parse($("#neo4j-params").value || "{}"); }
    catch (e) { st.textContent = "params is not valid JSON"; return; }
    st.textContent = "running…"; out.innerHTML = "";
    const body = { cypher: $("#neo4j-cypher").value, params, mode: $("#neo4j-mode").value };
    let r;
    try {
      r = await fetch("/api/neo4j/constraint", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then(x => x.json());
    } catch (e) { st.textContent = "✗ " + e.message; return; }
    if (r.error) { st.textContent = "✗ " + (r.message || r.error); return; }
    st.textContent = `${r.n} row(s) · mode=${r.mode}`;
    out.innerHTML = renderTable(r.columns || [], r.rows || []);
  }

  function renderTable(cols, rows) {
    if (!rows.length) return '<p class="hint">no rows — constraint satisfied (no violations).</p>';
    const head = cols.map(c => `<th>${esc(c)}</th>`).join("");
    const body = rows.map(row =>
      `<tr>${cols.map(c => `<td>${esc(fmt(row[c]))}</td>`).join("")}</tr>`).join("");
    return `<table class="neo4j-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  }

  function init() {
    if (loaded) return; loaded = true;
    loadStatus(); loadConstraints();
    $("#neo4j-refresh")?.addEventListener("click", loadStatus);
    $("#neo4j-export")?.addEventListener("click", exportDoc);
    $("#neo4j-constraint-select")?.addEventListener("change", applyConstraint);
    $("#neo4j-run")?.addEventListener("click", run);
  }

  for (const tab of document.querySelectorAll('#drawer-tabs .tab')) {
    tab.addEventListener("click", () => { if (tab.dataset.tab === "neo4j") init(); });
  }
})();
