"""Serve the baseline reader with a guideline editor beside it — the authoring
surface for the **no-feedback control** g_v2'.

`make_baseline_reader.py` produces the reading material; this adds the writing
half, because a control you have to draft in a separate editor is a control
nobody drafts. It serves exactly three things: the generated reader page, the
draft text, and a save/commit endpoint. It has no access to `data/conflicts/`,
no alignment, no graph — the isolation that makes the arm a control is a
property of this file's routes, not of the author's self-discipline.

    python scripts/serve_control_console.py          # http://localhost:8771

Draft autosaves to `prompts/extract_v2_control.md` (seeded from
`prompts/extract_v1.md` on first open). **Commit** freezes it: the text is
written, its sha256 and UTC timestamp recorded in
`prompts/extract_v2_control.lock.json`, and every later save refused. That lock
is what invariant 2 of CLAUDE.md means by "commit it timestamped and do not
revisit it".

Beside the draft sits the **change log**
(`prompts/extract_v2_control_rationale.json`): click the annotations on the left
that prompted an edit, say whether the edit modifies, deletes or adds a rule,
and write why. The citation is (doc_id, section_path, annotator, group_id), so
it resolves back into `data/facts/<model>/<doc>.json` rather than into a scroll
position. The log freezes with the draft and its digest goes into the lock — a
control whose reasons were never written down cannot be set beside g_v2's
diagnosis log, which is the comparison RQ2 turns on.

**History** puts both on one timeline: every draft checkpoint (with its word
delta and a diff against the current text, restorable) and every note, oldest
first. Autosave checkpoints inside a 90 s window roll into one; a pressed
button always keeps its own.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT_ = Path(__file__).resolve().parent.parent
if str(ROOT_) not in sys.path:
    sys.path.insert(0, str(ROOT_))

from src.guideline_log import DraftLog, normalize_newlines  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
READER = ROOT / "data" / "baseline_reader.html"
SEED_FROM = ROOT / "prompts" / "extract_v1.md"
DRAFT = ROOT / "prompts" / "extract_v2_control.md"
LOCK = ROOT / "prompts" / "extract_v2_control.lock.json"
SNAPS = ROOT / "prompts" / "_control_drafts"
RATIONALE = ROOT / "prompts" / "extract_v2_control_rationale.json"

SNAP_PREFIX = "gv2control_"

PANEL = r"""
<style>
 body.cc { margin: 0; max-width: none; padding: 16px 46vw 16px 16px; }
 body.cc header { padding-right: 0; }
 #cc { position: fixed; top: 0; right: 0; width: 44vw; height: 100vh;
       box-sizing: border-box; border-left: 2px solid #1f2328; background: #fff;
       display: flex; flex-direction: column; padding: 10px 12px; z-index: 50; }
 #cc h2 { font: 600 13px/1.4 system-ui, sans-serif; margin: 0 0 4px; }
 #cc .sub { font: 11.5px/1.5 system-ui, sans-serif; color: #57606a;
            margin-bottom: 8px; }
 #cc textarea { flex: 1; width: 100%; box-sizing: border-box; resize: none;
                font: 12.5px/1.55 ui-monospace, Consolas, monospace;
                padding: 8px; border: 1px solid #d8dee4; border-radius: 6px;
                tab-size: 2; }
 #cc .bar { display: flex; align-items: center; gap: 8px; padding-top: 8px;
            font: 11.5px/1.4 system-ui, sans-serif; color: #57606a; }
 #cc button { font: 12px system-ui, sans-serif; padding: 5px 12px;
              border-radius: 6px; border: 1px solid #d8dee4; background: #f6f8fa;
              cursor: pointer; }
 #cc button.primary { background: #cf222e; border-color: #cf222e; color: #fff; }
 #cc button:disabled { opacity: .5; cursor: default; }
 #cc .locked { background: #fdecea; border: 1px solid #f3b5ae; color: #7f1d1d;
               padding: 6px 10px; border-radius: 6px; margin-bottom: 8px;
               font: 11.5px/1.5 system-ui, sans-serif; }

 /* collapse / expand */
 body.cc.ccmin { padding-right: 46px; }
 #cc-body { display: flex; flex-direction: column; flex: 1; min-height: 0; }
 #cc.min { width: 34px; padding: 10px 2px; }
 #cc.min #cc-body, #cc.min > h2 { display: none; }
 #cc-toggle { position: absolute; top: 8px; left: -1px; padding: 3px 7px;
              line-height: 1; }
 #cc.min #cc-toggle { left: 2px; }
 #cc-tab { display: none; }
 #cc.min #cc-tab { display: block; margin: 40px auto 0; writing-mode: vertical-rl;
                   font: 600 12px/1 system-ui, sans-serif; color: #57606a;
                   letter-spacing: .02em; }
 #cc.min #cc-tab .b { color: #cf222e; }
 #cc > h2 { padding-left: 34px; }

 /* the draft shrinks so the change log has room; both scroll */
 #cc textarea#cc-text { flex: 3; }
 #cc .log { flex: 2; display: flex; flex-direction: column; min-height: 120px;
            border-top: 2px solid #1f2328; margin-top: 10px; padding-top: 8px; }
 #cc .log h3 { font: 600 12px/1.4 system-ui, sans-serif; margin: 0 0 4px; }
 #cc .cites { border: 1px solid #d8dee4; border-radius: 6px; padding: 5px 6px;
              min-height: 26px; max-height: 88px; overflow: auto;
              font: 11.5px/1.5 system-ui, sans-serif; color: #8c959f; }
 #cc .chip { display: inline-block; background: #fff8c5; border: 1px solid #d4a72c;
             border-radius: 10px; padding: 1px 7px; margin: 1px 3px 1px 0;
             color: #1f2328; cursor: pointer; }
 #cc .chip b { font-weight: 600; }
 #cc .chip .x { color: #8c959f; margin-left: 4px; }
 #cc .row { display: flex; gap: 6px; align-items: center; margin-top: 6px;
            font: 11.5px/1.4 system-ui, sans-serif; }
 #cc .row select, #cc .row input { font: 12px system-ui, sans-serif;
                                   padding: 4px 6px; border-radius: 6px;
                                   border: 1px solid #d8dee4; }
 #cc .row input { flex: 1; min-width: 0; }
 #cc textarea.why { flex: none; height: 46px; margin-top: 6px; resize: vertical;
                    font: 12px/1.5 system-ui, sans-serif; }
 #cc .entries { flex: 1; overflow: auto; margin-top: 8px; min-height: 40px; }
 #cc .ent { border-top: 1px solid #eaeef2; padding: 5px 0;
            font: 11.5px/1.5 system-ui, sans-serif; }
 #cc .ent .h { color: #1f2328; }
 #cc .op { font-weight: 600; text-transform: uppercase; font-size: 10.5px;
           letter-spacing: .04em; }
 #cc .op.modify { color: #9a6700; }
 #cc .op.delete { color: #cf222e; }
 #cc .op.new    { color: #1a7f37; }
 #cc .ent .c { color: #57606a; }
 #cc .ent .rm { float: right; color: #8c959f; cursor: pointer; }
 #cc .empty { color: #8c959f; font: 11.5px/1.5 system-ui, sans-serif; }

 #cc button.go { border-color: #0969da; color: #0969da; }
 #cc .c { color: #57606a; font: 11.5px/1.4 system-ui, sans-serif; }

 /* history overlay */
 #cc .hist { position: absolute; inset: 0; background: #fff; z-index: 60;
             padding: 10px 12px; box-sizing: border-box; display: none;
             flex-direction: column; }
 #cc.hist-open .hist { display: flex; }
 #cc .hist .list { flex: 1; overflow: auto; }
 #cc .hist .it { border-top: 1px solid #eaeef2; padding: 5px 2px;
                 font: 11.5px/1.5 system-ui, sans-serif; }
 #cc .hist .it.draft { cursor: pointer; }
 #cc .hist .it.draft:hover { background: #f6f8fa; }
 #cc .hist .it .when { color: #57606a; }
 #cc .hist .it .d { font-weight: 600; }
 #cc .hist .it .d.up { color: #1a7f37; }
 #cc .hist .it .d.down { color: #cf222e; }
 #cc .hist .it.note { background: #fbfbfd; }
 #cc .hist pre { flex: 1; overflow: auto; background: #f6f8fa; padding: 8px;
                 border-radius: 6px; margin: 0; white-space: pre-wrap;
                 font: 11.5px/1.5 ui-monospace, Consolas, monospace; }
 #cc .hist pre .a { color: #1a7f37; }
 #cc .hist pre .r { color: #cf222e; }
 #cc .hist pre .h { color: #8c959f; }
 /* the reasons filed against the open checkpoint, above its diff: a save shown
    without its account is half a record — the lines survive on disk, the
    reasoning only exists in the log */
 #cc .hist .note-box { margin: 4px 0 6px; max-height: 190px; overflow: auto; }
 #cc .hist .nt { border-left: 3px solid #0969da; background: #f6f8fa;
                 padding: 5px 8px; margin-bottom: 5px; font-size: 11.5px;
                 border-radius: 0 4px 4px 0; }
 #cc .hist .nt .rz { white-space: pre-wrap; margin: 3px 0; }
 #cc .hist .nt .op { font-weight: 600; text-transform: uppercase;
                     font-size: 10.5px; margin-right: 5px; }
 #cc .hist .has-note { color: #0969da; font-weight: 600; }

 /* the left page becomes selectable material */
 .ann { cursor: pointer; border-radius: 4px; }
 .ann:hover { background: #f6f8fa; }
 .ann.sel { background: #fff8c5; outline: 2px solid #d4a72c; }
</style>
<div id="cc">
  <button id="cc-toggle" title="collapse / expand the editor">&rsaquo;</button>
  <div id="cc-tab">g_v2&prime; editor <span class="b" id="cc-tab-n"></span></div>
  <h2>g_v2&prime; &mdash; no-feedback control</h2>
  <div id="cc-body">
  <div class="sub">Seeded from <code>prompts/extract_v1.md</code>. Saves to
    <code>prompts/extract_v2_control.md</code> as you type.
    <b>Commit</b> freezes it with a timestamp + sha256 and refuses later
    edits &mdash; that is the point of the arm, so press it when you stop
    reading, before any conflict view is opened.</div>
  <div id="cc-lock" class="locked" style="display:none"></div>
  <textarea id="cc-text" spellcheck="false"></textarea>
  <div class="bar">
    <button id="cc-save">Save draft</button>
    <button id="cc-hist">History</button>
    <button id="cc-commit" class="primary">Commit (freeze)</button>
    <span id="cc-status">loading...</span>
  </div>
  <div class="log">
    <h3>Change log &mdash; why this edit, and what it was read off</h3>
    <div class="cites" id="cc-cites">Click an annotation on the left to cite it.
      Click again to drop it.</div>
    <div class="row">
      <select id="cc-op">
        <option value="modify">modify rule</option>
        <option value="delete">delete rule</option>
        <option value="new">new rule</option>
      </select>
      <input id="cc-rule" placeholder="rule id or short name, e.g. R4 / object slot">
    </div>
    <textarea id="cc-why" class="why" spellcheck="false"
      placeholder="Why this change? What did the cited annotations do that g_v1 permits?"></textarea>
    <div class="row">
      <button id="cc-log" class="go">Log this change</button>
      <span id="cc-log-status"></span>
    </div>
    <div class="entries" id="cc-entries"></div>
  </div>
  </div>
  <div class="hist" id="cc-hist-pane">
    <div class="row" style="margin:0 0 6px">
      <b style="font:600 13px system-ui,sans-serif">History</b>
      <span id="cc-hist-sub" class="c"></span>
      <span style="flex:1"></span>
      <button id="cc-hist-prune" style="display:none">Clean up</button>
      <button id="cc-hist-close">Close</button>
    </div>
    <div class="list" id="cc-hist-list"></div>
    <div id="cc-hist-detail" style="display:none; flex:1; flex-direction:column;
         min-height:0">
      <div class="row" style="margin:6px 0 4px">
        <span id="cc-hist-which" class="c"></span>
        <span style="flex:1"></span>
        <button id="cc-hist-vs">vs current draft</button>
        <button id="cc-hist-restore">Restore into draft</button>
        <button id="cc-hist-back">Back</button>
      </div>
      <div id="cc-hist-notes"></div>
      <pre id="cc-hist-diff"></pre>
    </div>
  </div>
</div>
<script>
(function () {
  document.body.classList.add('cc');
  var ta = document.getElementById('cc-text');
  var st = document.getElementById('cc-status');
  var lockBox = document.getElementById('cc-lock');
  var saveBtn = document.getElementById('cc-save');
  var commitBtn = document.getElementById('cc-commit');
  var timer = null, locked = false, lastSent = null;

  function setLocked(lock) {
    locked = true;
    ta.readOnly = true;
    saveBtn.disabled = commitBtn.disabled = true;
    lockBox.style.display = '';
    lockBox.innerHTML = 'Committed ' + lock.committed_at + ' &middot; sha256 '
      + lock.sha256.slice(0, 12) + ' &middot; ' + lock.chars
      + ' chars. Further edits are refused. Diagnosis may now begin.';
    // the change log is frozen with the draft: it is part of the artefact
    document.getElementById('cc-log').disabled = true;
    document.getElementById('cc-op').disabled = true;
    document.getElementById('cc-rule').readOnly = true;
    document.getElementById('cc-why').readOnly = true;
    fetch('/api/rationale').then(function (r) { return r.json(); })
      .then(function (d) { renderEntries(d.entries || []); });
  }
  function words() {
    var w = ta.value.trim() ? ta.value.trim().split(/\s+/).length : 0;
    return w + ' words';
  }
  function save(kind) {
    if (locked) return;
    // A press of the button always answers. Only the 1.2 s autosave is allowed
    // to stay silent, and only when there is genuinely nothing to send.
    var explicit = (kind === 'commit' || kind === 'explicit');
    if (!explicit && ta.value === lastSent) return;
    st.textContent = kind === 'commit' ? 'committing...' : 'saving...';
    return fetch('/api/guideline', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: ta.value, commit: kind === 'commit',
                            explicit: explicit})
    }).then(function (r) { return r.json(); }).then(function (d) {
      if (d.error) { st.textContent = 'ERROR: ' + d.error; return; }
      var unchanged = (ta.value === lastSent);
      lastSent = ta.value;
      if (d.lock) { setLocked(d.lock); st.textContent = 'committed.'; return; }
      st.textContent = (unchanged ? 'no change · saved ' : 'saved ')
                     + d.saved_at + ' · ' + words()
        // the button above the note is not the button below it; say so once
        // rather than leaving a filled-in note looking saved
        + (pendingNote() ? ' · the change note below is NOT logged yet' : '');
    }).catch(function (e) { st.textContent = 'ERROR: ' + e; });
  }
  function pendingNote() {
    var w = document.getElementById('cc-why');
    var r = document.getElementById('cc-rule');
    return !!((w && w.value.trim()) || (r && r.value.trim()) || sel.length);
  }
  ta.addEventListener('input', function () {
    st.textContent = words() + ' · unsaved';
    clearTimeout(timer);
    timer = setTimeout(save, 1200);
  });
  saveBtn.onclick = function () { save('explicit'); };
  commitBtn.onclick = function () {
    if (confirm('Freeze g_v2′ now? After this it cannot be edited, and '
                + 'the conflict diagnosis is allowed to start.')) save('commit');
  };
  fetch('/api/guideline').then(function (r) { return r.json(); })
    .then(function (d) {
      ta.value = d.text; lastSent = d.text;
      st.textContent = words() + (d.seeded ? ' · seeded from v1' : '');
      if (d.lock) setLocked(d.lock);
    });

  /* ---- collapse / expand -------------------------------------------- */
  var panel = document.getElementById('cc');
  var toggle = document.getElementById('cc-toggle');
  function setMin(min) {
    panel.classList.toggle('min', min);
    document.body.classList.toggle('ccmin', min);
    toggle.innerHTML = min ? '&lsaquo;' : '&rsaquo;';
    toggle.title = (min ? 'expand' : 'collapse') + ' the editor';
    try { localStorage.setItem('cc-min', min ? '1' : '0'); } catch (e) {}
  }
  toggle.onclick = function () { setMin(!panel.classList.contains('min')); };
  var startMin = false;
  try { startMin = localStorage.getItem('cc-min') === '1'; } catch (e) {}
  setMin(startMin);

  /* ---- citing an annotation on the left ------------------------------ */
  var sel = [];                       // {doc, sec, model, gid, nl, el}
  var citeBox = document.getElementById('cc-cites');
  var opSel = document.getElementById('cc-op');
  var ruleIn = document.getElementById('cc-rule');
  var whyTa = document.getElementById('cc-why');
  var logBtn = document.getElementById('cc-log');
  var logSt = document.getElementById('cc-log-status');
  var entBox = document.getElementById('cc-entries');
  var tabN = document.getElementById('cc-tab-n');

  function esc(s) {
    return String(s === undefined || s === null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }
  function keyOf(c) { return c.doc + '|' + c.sec + '|' + c.model + '|' + c.gid; }
  function shortSec(s) { return String(s || '').replace(/^enacting\./, '')
                                               .replace(/^preamble\./, ''); }

  /* The draft autosaves; the note form did not, so a reload (or a server
     restart) emptied a half-written reason. Per browser, cleared on log. */
  function rememberNote() {
    try {
      localStorage.setItem('cc-note', JSON.stringify({
        op: opSel.value, rule: ruleIn.value, why: whyTa.value,
        cites: sel.map(function (c) {
          return {doc: c.doc, sec: c.sec, model: c.model, gid: c.gid, nl: c.nl};
        })
      }));
    } catch (e) {}
  }
  function forgetNote() { try { localStorage.removeItem('cc-note'); } catch (e) {} }
  function restoreNote() {
    var raw = null;
    try { raw = localStorage.getItem('cc-note'); } catch (e) {}
    if (!raw) return;
    var d;
    try { d = JSON.parse(raw); } catch (e) { return; }
    opSel.value = d.op || 'modify';
    ruleIn.value = d.rule || '';
    whyTa.value = d.why || '';
    var want = {};
    (d.cites || []).forEach(function (c) { want[keyOf(c)] = c; });
    var anns = document.querySelectorAll('.seg .ann[data-gid]');
    for (var i = 0; i < anns.length; i++) {
      var a = anns[i], seg = a.closest('.seg'), col = a.closest('.col');
      var c = {doc: seg ? (seg.dataset.doc || '') : '',
               sec: seg ? (seg.dataset.sec || '') : '',
               model: col ? (col.dataset.model || '') : '',
               gid: a.dataset.gid || ''};
      if (want[keyOf(c)]) {
        c.nl = want[keyOf(c)].nl || '';
        c.el = a; a.classList.add('sel'); sel.push(c);
      }
    }
    renderCites();
    if (whyTa.value || ruleIn.value || sel.length)
      logSt.textContent = 'restored, not logged yet';
  }

  function renderCites() {
    if (!sel.length) {
      citeBox.innerHTML = 'Click an annotation on the left to cite it. '
                        + 'Click it again to drop it.';
      return;
    }
    citeBox.innerHTML = sel.map(function (c, i) {
      return '<span class="chip" data-i="' + i + '"><b>' + esc(c.model)
        + '</b> ' + esc(c.doc) + ' &middot; ' + esc(shortSec(c.sec))
        + '<span class="x">&#10005;</span></span>';
    }).join('');
  }
  function drop(i) {
    if (sel[i].el) sel[i].el.classList.remove('sel');
    sel.splice(i, 1);
    renderCites();
    rememberNote();
  }
  citeBox.addEventListener('click', function (ev) {
    var chip = ev.target.closest ? ev.target.closest('.chip') : null;
    if (chip) drop(parseInt(chip.dataset.i, 10));
  });

  document.addEventListener('click', function (ev) {
    if (!ev.target.closest) return;
    var ann = ev.target.closest('.ann');
    if (!ann || ann.closest('#cc')) return;
    var seg = ann.closest('.seg'), col = ann.closest('.col');
    var c = {
      doc: seg ? (seg.dataset.doc || '') : '',
      sec: seg ? (seg.dataset.sec || '') : '',
      model: col ? (col.dataset.model || '') : '',
      gid: ann.dataset.gid || '',
      nl: ((ann.querySelector('.nl') || {}).textContent || '').trim(),
      el: ann
    };
    for (var i = 0; i < sel.length; i++) {
      if (keyOf(sel[i]) === keyOf(c)) { drop(i); return; }
    }
    sel.push(c);
    ann.classList.add('sel');
    renderCites();
    rememberNote();
    if (panel.classList.contains('min')) setMin(false);
  });
  [opSel, ruleIn, whyTa].forEach(function (el) {
    el.addEventListener('input', rememberNote);
    el.addEventListener('change', rememberNote);
  });
  restoreNote();

  /* ---- the change log ------------------------------------------------ */
  function renderEntries(entries) {
    tabN.textContent = entries.length ? '(' + entries.length + ')' : '';
    if (!entries.length) {
      entBox.innerHTML = '<div class="empty">No change logged yet. Every edit '
        + 'to the draft should leave one line here, so the control revision is '
        + 'documented the same way g_v2 will be.</div>';
      return;
    }
    entBox.innerHTML = entries.slice().reverse().map(function (e) {
      var cites = (e.cites || []).map(function (c) {
        return esc(c.annotator) + ' ' + esc(c.doc_id) + '&middot;'
             + esc(shortSec(c.section));
      }).join(' | ') || 'no annotation cited';
      return '<div class="ent">'
        + (locked ? '' : '<span class="rm" data-id="' + esc(e.id)
                       + '" title="remove">&#10005;</span>')
        + '<div class="h"><span class="op ' + esc(e.op) + '">' + esc(e.op)
        + '</span> ' + esc(e.rule || '(unnamed rule)') + '</div>'
        + '<div>' + esc(e.reason) + '</div>'
        + '<div class="c">' + cites + ' &middot; ' + esc(e.at) + '</div>'
        + '</div>';
    }).join('');
  }
  entBox.addEventListener('click', function (ev) {
    var rm = ev.target.closest ? ev.target.closest('.rm') : null;
    if (!rm) return;
    if (!confirm('Remove this log entry?')) return;
    post({remove: rm.dataset.id});
  });
  function post(body) {
    return fetch('/api/rationale', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    }).then(function (r) { return r.json(); }).then(function (d) {
      if (d.error) { logSt.textContent = 'ERROR: ' + d.error; return null; }
      renderEntries(d.entries);
      return d;
    }).catch(function (e) { logSt.textContent = 'ERROR: ' + e; return null; });
  }
  logBtn.onclick = function () {
    var why = whyTa.value.trim();
    if (!why) { logSt.textContent = 'a reason is required'; whyTa.focus(); return; }
    logSt.textContent = 'logging...';
    post({add: {
      op: opSel.value,
      rule: ruleIn.value.trim(),
      reason: why,
      cites: sel.map(function (c) {
        return {doc_id: c.doc, section: c.sec, annotator: c.model,
                group_id: c.gid, excerpt: c.nl.slice(0, 240)};
      })
    }}).then(function (d) {
      if (!d) return;
      logSt.textContent = 'logged ' + d.saved_at;
      whyTa.value = ''; ruleIn.value = '';
      while (sel.length) drop(0);
      forgetNote();
    });
  };
  fetch('/api/rationale').then(function (r) { return r.json(); })
    .then(function (d) { renderEntries(d.entries || []); });

  /* ---- history: the draft's checkpoints and the notes, one timeline ---- */
  var histPane = document.getElementById('cc-hist-pane');
  var histList = document.getElementById('cc-hist-list');
  var histSub = document.getElementById('cc-hist-sub');
  var histDetail = document.getElementById('cc-hist-detail');
  var histDiff = document.getElementById('cc-hist-diff');
  var histWhich = document.getElementById('cc-hist-which');
  var shown = null;

  function openHist(open) {
    panel.classList.toggle('hist-open', open);
    if (open) { showList(); loadHist(); }
  }
  function showList() {
    histDetail.style.display = 'none';
    histList.style.display = '';
  }
  document.getElementById('cc-hist').onclick = function () {
    openHist(!panel.classList.contains('hist-open'));
  };
  document.getElementById('cc-hist-close').onclick = function () { openHist(false); };
  document.getElementById('cc-hist-back').onclick = showList;
  document.getElementById('cc-hist-restore').onclick = function () {
    if (locked || !shown) return;
    if (!confirm('Replace the draft with ' + shown.name + '? The current text '
                 + 'is checkpointed first.')) return;
    var from = shown.name, text = shown.text;
    // strictly sequential: the checkpoint of what we are about to replace must
    // land before the replacement does, or a race decides which one survives
    Promise.resolve(save('explicit')).then(function () {
      ta.value = text;
      return save('explicit');
    }).then(function () {
      openHist(false);
      st.textContent = 'restored from ' + from + ' · ' + words();
    });
  };

  var pruneBtn = document.getElementById('cc-hist-prune');
  pruneBtn.onclick = function () {
    if (locked) return;
    if (!confirm('Collapse each burst of autosaves to the state it ended in? '
                 + 'The newest checkpoint of every burst, and every save more '
                 + 'than 90 s from its neighbour, are kept.')) return;
    fetch('/api/history', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prune: true})
    }).then(function (r) { return r.json(); }).then(function (d) {
      if (d.error) { histSub.textContent = 'ERROR: ' + d.error; return; }
      loadHist();
      st.textContent = 'history: removed ' + d.removed.length
                     + ', kept ' + d.kept;
    });
  };

  function loadHist() {
    histList.innerHTML = '<div class="empty">loading...</div>';
    fetch('/api/history').then(function (r) { return r.json(); })
      .then(function (d) {
        var items = d.items || [];
        histSub.textContent = d.drafts + ' draft checkpoints · ' + d.notes
          + ' change notes'
          + (d.prunable ? ' · ' + d.prunable + ' collapsible' : '');
        pruneBtn.style.display = (d.prunable && !locked) ? '' : 'none';
        if (!items.length) {
          histList.innerHTML = '<div class="empty">Nothing yet. A checkpoint '
            + 'is written every time the draft is saved.</div>';
          return;
        }
        histList.innerHTML = items.slice().reverse().map(function (it) {
          if (it.kind === 'note') {
            return '<div class="it note"><span class="when">' + esc(it.at)
              + '</span> &middot; <span class="op ' + esc(it.op) + '">'
              + esc(it.op) + '</span> ' + esc(it.rule || '(unnamed rule)')
              + ' &middot; ' + it.cites + ' cited</div>';
          }
          // what this save changed, not how far it is from today's text
          var what = it.add || it.rem
            ? '<span class="d up">+' + it.add + '</span> '
              + '<span class="d down">&minus;' + it.rem + '</span> lines'
            : (it.d_chars ? (it.d_chars > 0 ? '+' : '') + it.d_chars + ' chars'
                          : 'no change');
          return '<div class="it draft" data-name="' + esc(it.name) + '">'
            + '<span class="when">' + esc(it.at) + '</span> &middot; ' + what
            + ' <span class="when">&middot; ' + it.words + ' words'
            + (it.notes ? ' &middot; <span class="has-note">' + it.notes
                          + ' reason' + (it.notes > 1 ? 's' : '') + ' logged</span>' : '')
            + (it.current ? ' &middot; <b>= current draft</b>' : '') + '</span>'
            + '</div>';
        }).join('');
      });
  }
  histList.addEventListener('click', function (ev) {
    var it = ev.target.closest ? ev.target.closest('.it.draft') : null;
    if (!it) return;
    histDiff.innerHTML = 'loading...';
    histList.style.display = 'none';
    histDetail.style.display = 'flex';
    fetch('/api/history?name=' + encodeURIComponent(it.dataset.name))
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (d.error) { histDiff.textContent = 'ERROR: ' + d.error; return; }
        shown = d;
        vsCurrent = false;
        paintDiff();
        document.getElementById('cc-hist-restore').disabled = locked;
      });
  });
  var vsCurrent = false;
  var vsBtn = document.getElementById('cc-hist-vs');
  function paintNotes() {
    var box = document.getElementById('cc-hist-notes');
    if (!box) return;
    var notes = (shown && shown.notes) || [];
    if (!notes.length) {
      box.innerHTML = '<div class="c">No reason logged against this save — '
        + 'it is an autosave, not a recorded change.</div>';
      return;
    }
    box.className = 'note-box';
    box.innerHTML = notes.map(function (n) {
      var cs = (n.cites || []).map(function (c) {
        return esc((c.label || c.annotator || c.kind || 'cited') + ' '
                   + (c.doc_id || '') + (c.section ? ' · ' + c.section : ''));
      }).join(' | ');
      return '<div class="nt"><div><span class="op">' + esc(n.op) + '</span> <b>'
        + esc(n.rule || '(unnamed rule)') + '</b> <span class="when">' + esc(n.at)
        + (n.written_late ? ' · written late' : '')
        + (n.add != null ? ' · +' + n.add + ' / −' + n.rem : '')
        + '</span></div><div class="rz">' + esc(n.reason) + '</div>'
        + (cs ? '<div class="when">cites: ' + cs + '</div>' : '') + '</div>';
    }).join('');
  }

  function paintDiff() {
    if (!shown) return;
    paintNotes();
    histWhich.textContent = vsCurrent
      ? 'distance from ' + shown.name + ' to the current draft'
      : 'what this save changed, against ' + shown.base;
    vsBtn.textContent = vsCurrent ? 'what this save changed'
                                  : 'vs current draft';
    histDiff.innerHTML = (vsCurrent ? shown.diff_current : shown.diff)
      .split('\n').map(function (l) {
        var c = l.charAt(0) === '+' ? 'a' : (l.charAt(0) === '-' ? 'r'
              : (l.charAt(0) === '@' ? 'h' : ''));
        return c ? '<span class="' + c + '">' + esc(l) + '</span>' : esc(l);
      }).join('\n');
  }
  vsBtn.onclick = function () { vsCurrent = !vsCurrent; paintDiff(); };
})();
</script>
"""


def _lock():
    if LOCK.exists():
        return json.loads(LOCK.read_text(encoding="utf-8"))
    return None


def _log() -> DraftLog:
    """The shared substrate, pointed at this arm's files.

    Same entry shape, same checkpoints, same freeze as the diagnosed arm's
    logger in the VA interface — the two records have to be comparable, since
    RQ2 sets one revision against the other. What differs is only what an entry
    may cite, and that is decided by this server's routes, which serve the
    reader page and nothing from `data/conflicts/`.
    """
    return DraftLog(draft=DRAFT, seed=SEED_FROM, snaps_dir=SNAPS,
                    prefix=SNAP_PREFIX, log_path=RATIONALE,
                    artifact="g_v2_prime_change_log")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):  # quieter than the default
        sys.stderr.write("%s\n" % (fmt % a))

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            if not READER.exists():
                self._send(500, b"run scripts/make_baseline_reader.py first",
                           "text/plain; charset=utf-8")
                return
            page = READER.read_text(encoding="utf-8") + PANEL
            self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/guideline":
            seeded = not DRAFT.exists()
            src = SEED_FROM if seeded else DRAFT
            self._json(200, {"text": src.read_text(encoding="utf-8"),
                             "seeded": seeded, "lock": _lock()})
        elif path == "/api/rationale":
            payload = _log().read_log()
            payload["lock"] = _lock()
            self._json(200, payload)
        elif path == "/api/history":
            self._history(parse_qs(urlparse(self.path).query))
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        route = self.path.split("?")[0]
        if route not in ("/api/guideline", "/api/rationale", "/api/history"):
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        n = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return
        if _lock():
            self._json(409, {"error": "g_v2' is committed; edits are refused"})
            return
        if route == "/api/rationale":
            self._rationale_post(payload)
            return
        if route == "/api/history":
            self._history_post(payload)
            return
        text = payload.get("text")
        if not isinstance(text, str):
            self._json(400, {"error": "text required"})
            return
        text = normalize_newlines(text)      # a textarea hands back CRLF
        now = datetime.now(timezone.utc)
        DRAFT.write_text(text, encoding="utf-8")
        log = _log()
        log.checkpoint(text, now, bool(payload.get("explicit")
                                       or payload.get("commit")))
        if payload.get("commit"):
            lock = {
                "artifact": "g_v2_prime_no_feedback_control",
                "path": str(DRAFT.relative_to(ROOT)).replace("\\", "/"),
                "committed_at": now.strftime("%Y-%m-%d %H:%M:%SZ"),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "chars": len(text),
                "seeded_from": str(SEED_FROM.relative_to(ROOT)).replace("\\", "/"),
                "written_from": "prompts/extract_v1.md + data/baseline_reader.html "
                                "(raw g_v1 output only; no alignment, no conflict "
                                "labels, no graph)",
                "change_log": log.digest(),
            }
            LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
            self._json(200, {"lock": lock})
            return
        self._json(200, {"saved_at": now.strftime("%H:%M:%SZ")})

    # -------------------------------------------------------------- history
    def _history(self, query: dict) -> None:
        """One checkpoint / one note per line, or one checkpoint's own diff.

        Both halves of the artefact on the same timeline: the draft is what
        changed, the notes are why. Read together they are the account of the
        control revision the write-up has to give. A checkpoint is shown as
        **what that save changed** (`DraftLog._base_of`), not as its distance
        from today's text — that is the second view, behind a toggle.
        """
        log = _log()
        cur = (DRAFT if DRAFT.exists() else SEED_FROM).read_text(encoding="utf-8")
        name = (query.get("name") or [""])[0]
        if name:
            detail = log.detail(name, cur)
            if detail is None:
                self._json(404, {"error": "no such checkpoint"})
                return
            self._json(200, detail)
            return
        payload = log.history(DRAFT.read_text(encoding="utf-8")
                              if DRAFT.exists() else None)
        payload["lock"] = _lock()
        self._json(200, payload)

    def _history_post(self, payload: dict) -> None:
        """Apply the coalescing rule to checkpoints written before it existed."""
        if not payload.get("prune"):
            self._json(400, {"error": "prune required"})
            return
        log = _log()
        removed = log.prune()
        self._json(200, {"removed": removed, "kept": len(log.snaps())})

    # ------------------------------------------------------------------ log
    def _rationale_post(self, payload: dict) -> None:
        log = _log()
        if "remove" in payload:
            if not log.remove_entry(payload["remove"]):
                self._json(404, {"error": "no such entry",
                                 "entries": log.entries()})
                return
            self._json(200, {"entries": log.entries()})
            return
        entry = payload.get("add")
        if not isinstance(entry, dict):
            self._json(400, {"error": "add or remove required"})
            return
        now = datetime.now(timezone.utc)
        try:
            log.add_entry(entry, now)
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return
        self._json(200, {"entries": log.entries(),
                         "saved_at": now.strftime("%H:%M:%SZ")})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print("control console on http://%s:%d/" % (args.host, args.port))
    srv.serve_forever()


if __name__ == "__main__":
    main()
