/* The diagnosed arm's revision record: why each guideline edit was made, what
 * it was read off, and what every save changed.
 *
 * Mirrors the control console (scripts/serve_control_console.py) on purpose —
 * RQ2 sets g_v2 against g_v2', and two revisions are only comparable if they
 * left the same kind of record. Both write through src/guideline_log.py. The
 * one difference is what an entry may cite: there, a raw annotation; here, what
 * the interface showed — a reviewed pair, its conflict label, its annotators.
 *
 * One deliberate divergence: Restore fills the editor rather than writing the
 * file. The console autosaves, so a restore there is one more autosave; here
 * nothing is written until Save is pressed, and a restore that wrote straight
 * to extract_<v>.md would destroy unsaved text the user can still see.
 */
(function () {
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s === undefined || s === null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

  let cites = [];           // pending citations for the entry being written
  let shots = [];           // screenshots already uploaded for that entry
  let shown = null;         // checkpoint open in the history detail
  let vsCurrent = false;

  const version = () => ($("#gm-new-version")?.value || "").trim()
                     || $("#gm-select")?.value || "v2";
  const noteKey = () => "gl-note-" + version();
  const shortSec = (s) => String(s || "").replace(/^enacting\./, "")
                                         .replace(/^preamble\./, "");
  // the chips are 200px wide; a 60-character hf.co tag is a chip of one word
  const shortAnnotator = (a) => String(a || "").replace(/^cloud:/, "")
    .replace(/^hf\.co\/[^/]+\//, "").replace(/-(it-)?GGUF.*$/, "")
    .replace(/:.*$/, "").replace(/^NVIDIA-/, "").slice(0, 22);

  /** Colour one unified-diff line: green added, red removed, grey structure.
   *  The `---`/`+++` file headers start with those characters but are not a
   *  deletion and an addition — painting them red and green makes every diff
   *  open with one false removal and one false addition. */
  function diffClass(l) {
    if (l.startsWith("---") || l.startsWith("+++") || l[0] === "@") return "hunk";
    if (l[0] === "+") return "add";
    if (l[0] === "-") return "rem";
    return "";
  }

  /* ---------------------------------------------------------- citations */
  // `version` is part of the identity: the ids do not change between guideline
  // versions, so the same key under v1 and v2 names two different objects
  // (see CITE_FIELDS in src/guideline_log.py). A citation without one is v1.
  function citeKey(c) {
    return [c.kind, c.doc_id, c.pair_key, c.section, c.annotator, c.group_id,
            c.version || "v1"].join("|");
  }
  function cite(obj) {
    const c = Object.assign({kind: "pair"}, obj);
    const i = cites.findIndex((x) => citeKey(x) === citeKey(c));
    if (i >= 0) cites.splice(i, 1); else cites.push(c);
    renderCites();
    remember();
    // The log has its own drawer tab, so a citation can land in a pane nobody
    // is looking at. Bring it forward — a citation you do not see arrive reads
    // as a click that did nothing.
    window.goToTab?.("change-log");
    // the conflict modal covers the drawer it just cited into
    document.querySelector("#modal-overlay")?.classList.add("hidden");
    const pane = $("#gl-panel");
    if (pane) {
      openBody(true);
      pane.scrollIntoView({behavior: "smooth", block: "nearest"});
    }
    return cites.length;
  }
  function renderCites() {
    const box = $("#gl-cites");
    if (!box) return;
    if (!cites.length) {
      box.innerHTML = '<span class="hint">Nothing cited yet — use <b>cite</b> '
        + 'on a reviewed pair, or log the change without a citation.</span>';
      return;
    }
    box.innerHTML = cites.map((c, i) => {
      // a pair reads by its conflict label; everything else by what it is of —
      // one model on one segment, one annotation, or a whole segment
      const what = c.pair_key ? (c.label || "pair")
                 : c.kind === "segment" ? "segment"
                 : (shortAnnotator(c.annotator) || c.kind || "cited");
      return `<span class="gl-chip" data-i="${i}"><b>${esc(what)}</b> `
        + `${esc(c.doc_id || "")}${c.section ? " · " + esc(shortSec(c.section)) : ""}`
        + `${c.version && c.version !== "v1" ? " · " + esc(c.version) : ""}`
        + `<span class="x">✕</span></span>`;
    }).join("");
  }

  /* --------------------------------------------------------- screenshots */
  /* A citation names what the interface showed; a screenshot shows it. Some
   * reasons only exist as a picture — three annotators' spans side by side, the
   * shape of a group — and the view they were read off will not come back: the
   * conflicts get recomputed, the layout moves. So the image is uploaded and
   * kept with the version, and the entry stores the filename, not a link into a
   * view. Uploading happens on paste/drop, before the entry is logged, so a
   * shot can exist with no entry; that is the cheap direction of the trade (an
   * unused file) against the expensive one (a reason with a dead image). */
  const shotUrl = (f) => `/api/guideline_shot/${encodeURIComponent(version())}`
                       + `/${encodeURIComponent(f)}`;

  async function upload(blob) {
    const status = $("#gl-shot-status") || $("#gl-log-status");
    if (version() === "v1") {
      status.textContent = "v1 is the baseline — name a new version first";
      return null;
    }
    const fd = new FormData();
    fd.append("file", blob, blob.name || "clipboard.png");
    status.textContent = "uploading screenshot…";
    const r = await fetch(`/api/guideline_shot/${encodeURIComponent(version())}`,
                          {method: "POST", body: fd});
    const d = await r.json().catch(() => ({}));
    if (!r.ok) {
      status.textContent = "ERROR: " + (d.description || d.message || r.status);
      return null;
    }
    status.textContent = "";
    shots.push({file: d.file, caption: ""});
    renderShots(); remember();
    return d;
  }

  function renderShots() {
    const box = $("#gl-shot-list");
    if (!box) return;
    box.innerHTML = shots.map((s, i) => `<figure class="gl-shot" data-i="${i}">
        <span class="x" title="remove from this entry">✕</span>
        <img src="${esc(shotUrl(s.file))}" alt="${esc(s.caption || s.file)}"
             title="click to view full size">
        <input type="text" class="cap" data-i="${i}" value="${esc(s.caption)}"
               placeholder="what this shows"></figure>`).join("");
  }

  function lightbox(file, caption) {
    $("#gl-lightbox-img").src = shotUrl(file);
    $("#gl-lightbox-cap").textContent = caption || file;
    $("#gl-lightbox").style.display = "";
  }

  /** Every route a screenshot can arrive by, wired to the same upload. */
  function wireShots() {
    const zone = $("#gl-shots");
    if (!zone) return;
    $("#gl-shot-pick").addEventListener("click", () => $("#gl-shot-file").click());
    $("#gl-shot-file").addEventListener("change", async (ev) => {
      for (const f of ev.target.files) await upload(f);
      ev.target.value = "";                       // so the same file re-picks
    });
    // Ctrl+V anywhere in the panel: the clipboard is how a screenshot tool
    // hands over, and requiring a save-to-disk round trip first is the reason
    // nobody attaches one.
    $("#gl-panel").addEventListener("paste", async (ev) => {
      const items = [...(ev.clipboardData?.items || [])]
        .filter((i) => i.type.startsWith("image/"));
      if (!items.length) return;                  // plain text paste: leave it
      ev.preventDefault();
      for (const it of items) { const b = it.getAsFile(); if (b) await upload(b); }
    });
    zone.addEventListener("dragover", (ev) => {
      ev.preventDefault(); zone.classList.add("over");
    });
    zone.addEventListener("dragleave", () => zone.classList.remove("over"));
    zone.addEventListener("drop", async (ev) => {
      ev.preventDefault(); zone.classList.remove("over");
      for (const f of ev.dataTransfer.files || [])
        if (f.type.startsWith("image/")) await upload(f);
    });
    $("#gl-shot-list").addEventListener("click", (ev) => {
      const fig = ev.target.closest(".gl-shot");
      if (!fig) return;
      const i = parseInt(fig.dataset.i, 10);
      if (ev.target.classList.contains("x")) {    // drop from the entry only —
        shots.splice(i, 1);                       // the file stays on disk
        renderShots(); remember();
      } else if (ev.target.tagName === "IMG") {
        lightbox(shots[i].file, shots[i].caption);
      }
    });
    $("#gl-shot-list").addEventListener("input", (ev) => {
      const cap = ev.target.closest(".cap");
      if (!cap) return;
      shots[parseInt(cap.dataset.i, 10)].caption = cap.value;
      remember();
    });
    $("#gl-lightbox-close").addEventListener("click", () =>
      $("#gl-lightbox").style.display = "none");
    $("#gl-lightbox").addEventListener("click", (ev) => {
      if (ev.target.id === "gl-lightbox") $("#gl-lightbox").style.display = "none";
    });
    $("#gl-entries").addEventListener("click", (ev) => {
      const t = ev.target.closest(".gl-ent-shot");
      if (t) lightbox(t.dataset.file, t.dataset.caption);
    });
  }

  /* ------------------------------------------------- the pending note ---- */
  function remember() {
    try {
      localStorage.setItem(noteKey(), JSON.stringify({
        op: $("#gl-op")?.value, rule: $("#gl-rule")?.value,
        why: $("#gl-why")?.value, cites, shots,
      }));
    } catch (e) { /* private window, blocked storage — the form still works */ }
  }
  function forget() { try { localStorage.removeItem(noteKey()); } catch (e) {} }
  /** True when the form holds work nobody has filed yet. */
  function pending() {
    return cites.length || shots.length
        || ($("#gl-why")?.value || "").trim() || ($("#gl-rule")?.value || "").trim();
  }

  function restoreNote() {
    let raw = null;
    try { raw = localStorage.getItem(noteKey()); } catch (e) {}
    // The note is kept per version, and the version can change *after* the note
    // was started — citing a column and then naming the version you are writing
    // to is the natural order, and it used to wipe the citations on the way.
    // An unfiled note follows the author to the version they just named; it is
    // a draft, not a record, and silently dropping one is data loss.
    if (!raw && pending()) { remember(); return; }
    if (!raw) { cites = []; shots = []; renderCites(); renderShots(); return; }
    let d; try { d = JSON.parse(raw); } catch (e) { return; }
    if ($("#gl-op")) $("#gl-op").value = d.op || "modify";
    if ($("#gl-rule")) $("#gl-rule").value = d.rule || "";
    if ($("#gl-why")) $("#gl-why").value = d.why || "";
    cites = d.cites || [];
    shots = d.shots || [];
    renderCites();
    renderShots();
  }

  /* --------------------------------------------------------- the entries */
  async function loadLog() {
    try {
      const d = await fetch(`/api/guideline_log/${encodeURIComponent(version())}`)
        .then((r) => r.json());
      renderEntries(d.entries || []);
    } catch (e) { /* a version with no log yet is the normal case */ }
  }
  /** v1 is the baseline, not a revision: a change log against it would be a
   *  record of editing the thing every version is measured against. */
  function guardBaseline() {
    const base = version() === "v1";
    $("#gl-log").disabled = base;
    const t = $("#gl-target");
    if (t) t.textContent = `extract_${version()}.md`;
    if (base) {
      $("#gl-sub").textContent = "v1 is the baseline — put a new version in "
        + "'Save as' before logging a change";
    }
    return base;
  }

  function renderEntries(entries) {
    const box = $("#gl-entries");
    if (!box) return;
    const target = `extract_${version()}.md`;
    const t = $("#gl-target");
    if (t) t.textContent = target;
    // The version lives on the Annotation tab's controls, which are out of
    // sight from here — so the panel has to say what it is about to write to.
    $("#gl-sub").textContent = entries.length
      ? `${entries.length} change${entries.length > 1 ? "s" : ""} logged for ${target}`
      : `writing to ${target} — why this edit, citing what the interface showed`;
    guardBaseline();
    if (!entries.length) {
      box.innerHTML = '<div class="hint">No change logged yet. Every edit to '
        + 'the draft should leave one line here — it is what the write-up '
        + 'reports as the diagnosis, and what the control is compared with.</div>';
      return;
    }
    box.innerHTML = entries.slice().reverse().map((e) => {
      const cs = (e.cites || []).map((c) => esc(
        (c.label || c.annotator || c.kind || "cited") + " " + (c.doc_id || "")
        + (c.section ? "·" + shortSec(c.section) : ""))).join(" | ")
        || "nothing cited";
      // The pictures ride under the reason, at thumbnail size: the entry list
      // is read as a list, and a full-width screenshot per entry would bury the
      // reasons it exists to show. Click opens it.
      const shs = (e.shots || []).map((s) =>
        `<img class="gl-ent-shot" src="${esc(shotUrl(s.file))}"
              data-file="${esc(s.file)}" data-caption="${esc(s.caption || "")}"
              alt="${esc(s.caption || s.file)}" title="${esc(s.caption || s.file)}">`
      ).join("");
      // The lines this entry changed, red removed / green added. A reason
      // without them is an assertion; with them the record carries its own
      // evidence, which is what the write-up reports.
      const dif = e.diff
        ? `<details class="gl-ent-diff"${e.unchanged ? "" : " open"}>
             <summary><b class="add">+${e.add || 0}</b>
               <b class="rem">−${e.rem || 0}</b> lines · vs ${esc(e.base || "")}</summary>
             <pre class="gl-diff">${e.diff.split("\n").map((l) => {
               const c = diffClass(l);
               return c ? `<span class="${c}">${esc(l)}</span>` : esc(l);
             }).join("\n")}</pre></details>`
        : "";
      return `<div class="gl-ent">
        <span class="rm" data-id="${esc(e.id)}" title="remove">✕</span>
        <div><span class="gl-op ${esc(e.op)}">${esc(e.op)}</span>
             ${esc(e.rule || "(unnamed rule)")}</div>
        <div>${esc(e.reason)}</div>
        ${dif}
        ${shs ? `<div class="gl-ent-shots">${shs}</div>` : ""}
        <div class="hint">${cs} · ${esc(e.at)}</div></div>`;
    }).join("");
  }
  async function post(body) {
    const r = await fetch(`/api/guideline_log/${encodeURIComponent(version())}`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) {
      $("#gl-log-status").textContent = "ERROR: " + (d.description || d.error || r.status);
      return null;
    }
    renderEntries(d.entries || []);
    return d;
  }

  /* ------------------------------------------------------------- history */
  async function loadHistory() {
    const list = $("#gl-hist-list");
    list.innerHTML = '<div class="hint">loading…</div>';
    const d = await fetch(`/api/guideline_history/${encodeURIComponent(version())}`)
      .then((r) => r.json());
    $("#gl-hist-sub").textContent = `${d.drafts} checkpoints · ${d.notes} notes`
      + (d.prunable ? ` · ${d.prunable} collapsible` : "");
    $("#gl-prune").style.display = d.prunable ? "" : "none";
    const items = d.items || [];
    if (!items.length) {
      list.innerHTML = '<div class="hint">Nothing yet. A checkpoint is written '
        + 'every time this version is saved.</div>';
      return;
    }
    list.innerHTML = items.slice().reverse().map((it) => {
      if (it.kind === "note") {
        return `<div class="gl-ent note"><span class="hint">${esc(it.at)}</span>
          · <span class="gl-op ${esc(it.op)}">${esc(it.op)}</span>
          ${esc(it.rule || "(unnamed rule)")} · ${it.cites} cited`
          + (it.shots ? ` · ${it.shots} screenshot${it.shots > 1 ? "s" : ""}` : "")
          + `</div>`;
      }
      // what this save changed — not how far it is from today's text
      const what = (it.add || it.rem)
        ? `<b class="add">+${it.add}</b> <b class="rem">−${it.rem}</b> lines`
        : (it.d_chars ? `${it.d_chars > 0 ? "+" : ""}${it.d_chars} chars`
                      : "no change");
      return `<div class="gl-ent draft" data-name="${esc(it.name)}">
        <span class="hint">${esc(it.at)}</span> · ${what}
        <span class="hint">· ${it.words} words`
        + (it.notes ? ` · <b class="has-note">${it.notes} reason`
                      + `${it.notes > 1 ? "s" : ""} logged</b>` : "")
        + (it.current ? " · <b>= current file</b>" : "") + "</span></div>";
    }).join("");
  }
  /** The reasons filed against the open checkpoint.
   *
   * The diff says what changed; without this the History pane shows an edit and
   * no account of it — and the account is the half that cannot be recovered
   * later. The lines are still on disk a year from now; why they were written
   * is only here.
   */
  function paintNotes() {
    const box = $("#gl-hist-notes");
    if (!box) return;
    const notes = (shown && shown.notes) || [];
    if (!notes.length) {
      box.innerHTML = '<div class="hint">No reason logged against this save — '
        + 'it is a checkpoint, not a recorded change. Use <b>Save &amp; log this '
        + 'change</b> to file one together with the edit.</div>';
      return;
    }
    box.innerHTML = notes.map((n) => {
      const cs = (n.cites || []).map((c) => esc(
        (c.label || shortAnnotator(c.annotator) || c.kind || "cited") + " "
        + (c.doc_id || "") + (c.section ? " · " + shortSec(c.section) : "")))
        .join(" | ");
      const sh = (n.shots || []).map((x) =>
        `<img class="gl-ent-shot" src="${esc(shotUrl(x.file))}"
              data-file="${esc(x.file)}" data-caption="${esc(x.caption || "")}"
              alt="${esc(x.caption || x.file)}">`).join("");
      return `<div class="gl-note">
        <div><span class="gl-op ${esc(n.op)}">${esc(n.op)}</span>
             <b>${esc(n.rule || "(unnamed rule)")}</b>
             <span class="hint">${esc(n.at)}`
        + (n.written_late ? " · written late" : "")
        + (n.add != null ? ` · <b class="add">+${n.add}</b> <b class="rem">−${n.rem}</b>` : "")
        + `</span></div>
        <div class="gl-note-reason">${esc(n.reason)}</div>`
        + (sh ? `<div class="gl-ent-shots">${sh}</div>` : "")
        + (cs ? `<div class="hint">cites: ${cs}</div>` : "")
        + `</div>`;
    }).join("");
  }

  function paintDiff() {
    if (!shown) return;
    paintNotes();
    $("#gl-hist-which").textContent = vsCurrent
      ? `distance from ${shown.name} to the saved file`
      : `what this save changed, against ${shown.base}`;
    $("#gl-hist-vs").textContent = vsCurrent ? "what this save changed"
                                             : "vs current draft";
    $("#gl-hist-diff").innerHTML = (vsCurrent ? shown.diff_current : shown.diff)
      .split("\n").map((l) => {
        const c = diffClass(l);
        return c ? `<span class="${c}">${esc(l)}</span>` : esc(l);
      }).join("\n");
  }

  function openBody(open) {
    const body = $("#gl-body");
    if (!body) return;
    body.style.display = open ? "" : "none";
    $("#gl-toggle").textContent = open ? "▾" : "▸";
    try { localStorage.setItem("gl-open", open ? "1" : "0"); } catch (e) {}
  }
  function openHist(open) {
    $("#gl-hist").style.display = open ? "" : "none";
    $("#gl-body").style.display = open ? "none" : "";
    if (open) { $("#gl-hist-detail").style.display = "none";
                $("#gl-hist-list").style.display = ""; loadHistory(); }
  }

  /* ---------------------------------------------------------------- wire */
  function init() {
    if (!$("#gl-panel")) return;
    $("#gl-toggle").addEventListener("click", () =>
      openBody($("#gl-body").style.display === "none"));
    $("#gl-history").addEventListener("click", () =>
      openHist($("#gl-hist").style.display === "none"));
    $("#gl-hist-close").addEventListener("click", () => openHist(false));
    $("#gl-hist-back").addEventListener("click", () => {
      $("#gl-hist-detail").style.display = "none";
      $("#gl-hist-list").style.display = "";
    });
    $("#gl-hist-vs").addEventListener("click", () => {
      vsCurrent = !vsCurrent; paintDiff();
    });
    $("#gl-hist-restore").addEventListener("click", () => {
      if (!shown) return;
      $("#gm-text").value = shown.text;
      openHist(false);
      $("#gm-status").textContent = `restored ${shown.name} into the editor — `
        + "press Save to write it";
      $("#gm-status").className = "ok";
    });
    $("#gl-prune").addEventListener("click", async () => {
      if (!confirm("Collapse each burst of saves to the state it ended in? "
                 + "The newest checkpoint of every burst, and every save more "
                 + "than 90 s from its neighbour, are kept.")) return;
      const d = await fetch(`/api/guideline_history/${encodeURIComponent(version())}`, {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({prune: true}),
      }).then((r) => r.json());
      $("#gl-hist-sub").textContent = `removed ${(d.removed || []).length}, kept ${d.kept}`;
      loadHistory();
    });
    $("#gl-hist-list").addEventListener("click", async (ev) => {
      const it = ev.target.closest(".gl-ent.draft");
      if (!it) return;
      $("#gl-hist-list").style.display = "none";
      $("#gl-hist-detail").style.display = "";
      $("#gl-hist-diff").textContent = "loading…";
      shown = await fetch(`/api/guideline_history/${encodeURIComponent(version())}`
        + `?name=${encodeURIComponent(it.dataset.name)}`).then((r) => r.json());
      vsCurrent = false;
      paintDiff();
    });
    $("#gl-cites").addEventListener("click", (ev) => {
      const chip = ev.target.closest(".gl-chip");
      if (!chip) return;
      cites.splice(parseInt(chip.dataset.i, 10), 1);
      renderCites(); remember();
    });
    $("#gl-entries").addEventListener("click", async (ev) => {
      const rm = ev.target.closest(".rm");
      if (!rm || !confirm("Remove this log entry?")) return;
      await post({remove: rm.dataset.id});
    });
    $("#gl-log").addEventListener("click", async () => {
      // the version selects populate asynchronously, so the disabled state can
      // be a step behind; this is the check that actually holds
      if (guardBaseline()) {
        $("#gl-log-status").textContent =
          "v1 is the baseline — name a new version in 'Save as' first";
        return;
      }
      const why = ($("#gl-why").value || "").trim();
      if (!why) { $("#gl-log-status").textContent = "a reason is required";
                  $("#gl-why").focus(); return; }
      $("#gl-log-status").textContent = "logging…";
      // One press: the draft is saved, checkpointed, and the reason is filed
      // with the lines it changed. Logging a reason and saving the file used to
      // be two actions, and the first entry ever written here described an edit
      // the file did not carry (BUG-2026-09-20-D).
      const d = await post({
        text: $("#gm-text") ? $("#gm-text").value : undefined,
        add: {
          op: $("#gl-op").value, rule: ($("#gl-rule").value || "").trim(),
          reason: why, cites, shots,
        }});
      if (!d) return;
      // Two ways a logged reason ends up describing an edit the file does not
      // carry. Both are warnings, not refusals: a reason may be written before
      // the edit or after it — but silently recording a revision that is not
      // there is the one outcome the change log must never produce.
      const last = (d.entries || [])[(d.entries || []).length - 1] || {};
      const moved = (last.add || 0) + (last.rem || 0);
      const note = moved
        ? ` — saved, +${last.add} / −${last.rem} lines`
        : " ⚠ nothing changed in the guideline text: the entry is filed, but it "
          + "describes an edit the file does not carry";
      $("#gl-log-status").textContent = "logged " + d.saved_at + note;
      $("#gl-log-status").className = moved ? "ok" : "err";
      if (moved && $("#gm-status")) {
        window.__gmSaved = $("#gm-text").value;
        $("#gm-status").textContent = `saved ${version()} from the change log`;
        $("#gm-status").className = "ok";
      }
      $("#gl-why").value = ""; $("#gl-rule").value = "";
      cites = []; shots = []; renderCites(); renderShots(); forget();
    });
    for (const sel of ["#gl-op", "#gl-rule", "#gl-why"]) {
      $(sel).addEventListener("input", remember);
      $(sel).addEventListener("change", remember);
    }
    // the log follows the version you are authoring, not the one you loaded
    // `change` alone fires on blur, so typing a new name in "Save as" left the
    // panel still guarding the baseline — the Log button stayed disabled and
    // #gl-target still said extract_v1.md while the editor was on v2.
    for (const sel of ["#gm-new-version", "#gm-select"]) {
      for (const ev of ["change", "input"]) {
        document.querySelector(sel)?.addEventListener(ev, () => {
          restoreNote(); loadLog(); guardBaseline();
        });
      }
    }
    wireShots();
    let open = true;
    try { open = localStorage.getItem("gl-open") !== "0"; } catch (e) {}
    openBody(open);
    restoreNote();
    loadLog();
    guardBaseline();
  }

  window.GLog = {cite, reload: loadLog, version};
  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", init);
  else init();
})();
