#!/usr/bin/env bash
# One-off watchdog for the 2026-09-02 cloud dev run against the flaky DBvis
# box. Every 30 minutes: stop whatever extraction pass is in flight, scan
# its stderr log for sections that were silently SKIPPED after a network
# failure (these land as status: ok, fact_count: 0 — see
# scripts/requeue_incomplete.py's docstring) and delete those manifest
# entries so they're picked up as pending again, then relaunch. Stops once a
# full pass leaves nothing requeued and all 114 (model, doc) pairs are ok.
# Not part of the permanent pipeline; delete once this run is done.
set -u
cd "$(dirname "$0")/.."

PY=".venv/Scripts/python.exe"
MODELS="cloud:qwen3.6:35b,cloud:hf.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF:UD-Q4_K_M,cloud:hf.co/unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL"
DOC_LIST="data/doc_lists/phase1_dev_only.json"
MANIFEST="data/facts/manifest.json"
WATCHDOG_LOG="data/facts/experiment_watchdog.log"

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$WATCHDOG_LOG"; }

round=0
while true; do
  round=$((round+1))
  RUN_LOG="data/facts/run_dev_cloud_20260902_r${round}.log"
  ERR_LOG="data/facts/run_dev_cloud_20260902_r${round}.err.log"

  log "round ${round}: launching run_extraction (models=3, doc-list=${DOC_LIST})"
  "$PY" -m scripts.run_extraction run \
    --models "$MODELS" \
    --doc-list "$DOC_LIST" \
    --out data/facts \
    --manifest "$MANIFEST" \
    >> "$RUN_LOG" 2>> "$ERR_LOG" &
  pid=$!
  log "round ${round}: pid ${pid}, checking back in 30 minutes"

  # Sleep 30 min, but wake early if the process exits on its own (e.g. a
  # clean full pass finishes faster than the check interval).
  slept=0
  while [ $slept -lt 1800 ] && kill -0 "$pid" 2>/dev/null; do
    sleep 30
    slept=$((slept+30))
  done

  if kill -0 "$pid" 2>/dev/null; then
    log "round ${round}: 30 minutes up, stopping pid ${pid} for a clean manifest snapshot"
    # Plain `kill` on a git-bash job number doesn't reliably reach the
    # underlying native python.exe on this machine (observed earlier this
    # session: ps/kill from a separate shell couldn't see a process that
    # PowerShell's Stop-Process could) — use taskkill on the whole process
    # tree via the shell PID as well as a name-based fallback.
    taskkill //F //T //PID "$pid" >/dev/null 2>&1
    kill -9 "$pid" 2>/dev/null
    sleep 3
  else
    log "round ${round}: run_extraction already exited on its own"
  fi
  wait "$pid" 2>/dev/null

  requeue_out=$("$PY" scripts/requeue_incomplete.py "$ERR_LOG" 2>&1)
  echo "$requeue_out" | tee -a "$WATCHDOG_LOG"
  n_requeued=$(echo "$requeue_out" | head -1 | grep -oE '[0-9]+' | head -1)
  n_requeued=${n_requeued:-0}

  n_ok=$("$PY" -c "
import json
m = json.load(open('$MANIFEST', encoding='utf-8'))
print(len(m['runs']))
")
  log "round ${round}: manifest has ${n_ok}/114 pairs recorded; ${n_requeued} requeued this round"

  if [ "$n_requeued" -eq 0 ] && [ "$n_ok" -ge 114 ]; then
    log "all 114 (model, doc) pairs are clean and ok — watchdog done."
    break
  fi
done
