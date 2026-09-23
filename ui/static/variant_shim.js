/* ============================================================
 * variant_shim.js — MUST load before app.js.
 *
 * Makes the per-doc read endpoints "guideline-variant aware" without
 * touching app.js: when window.__variant is non-empty, GET requests to
 * the endpoints below get `variant=<v>` appended, so the existing
 * switchDoc() plumbing transparently loads e.g.
 * data/conflicts/<doc>__v2.json instead of the baseline file.
 *
 * The top-bar version dropdown (managed by analysis.js) sets
 * window.__variant and re-dispatches a change event on #doc-select.
 * ============================================================ */
(() => {
  // Restored HERE, not in analysis.js, because this file runs before app.js
  // and app.js's boot() fires the first /api/docs before DOMContentLoaded
  // handlers run. Setting it later meant the first document list was always
  // the baseline's while the dropdown already read as the remembered version.
  // Key must match VERSION_KEY in analysis.js.
  try { window.__variant = localStorage.getItem("va-guideline-version") || ""; }
  catch { window.__variant = ""; }
  // Any per-doc read endpoint added later must be listed here too, or it
  // silently keeps serving the baseline file while the rest of the page shows
  // the selected variant (that is how /api/groups first shipped).
  // `docs` is in here because WHICH documents have a Phase-2 result is itself
  // version-specific — without it the document list advertises v1's coverage
  // while every panel below shows the selected version. `gold_eval` was
  // removed with src/gold_eval.py (2026-09-10, WebNLG/Re-DocRED out of scope)
  // and `gold_compare` is unreachable for EUR-Lex docs, which carry no
  // `gold_triples`.
  const RX = /^\/api\/(?:(?:facts|graph|pairs|groups|coverage|similarity_matrix|iaa)\/|docs\b)/;
  const orig = window.fetch.bind(window);
  window.fetch = (input, init) => {
    try {
      const url = typeof input === "string" ? input : input.url;
      if (window.__variant && RX.test(url)) {
        const patched = url + (url.includes("?") ? "&" : "?") +
          "variant=" + encodeURIComponent(window.__variant);
        input = typeof input === "string" ? patched : new Request(patched, input);
      }
    } catch (e) { /* never break fetch */ }
    return orig(input, init);
  };
})();
