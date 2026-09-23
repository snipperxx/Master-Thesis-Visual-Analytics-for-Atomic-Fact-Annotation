#!/bin/sh
# Thesis dev extraction pass (Zilong Yan, MasterThesis) — 3 cloud annotators.
#
# Resumable: re-running skips pairs already `status: ok` in
# data/facts/manifest.json. Records flagged by scripts/audit_facts_coverage.py
# must be requeued (--requeue) before they are redone.
#
# IDENTIFYING THIS PROCESS. Another repo (fiz-kg/json2kg) runs its own
# python.exe on this machine, so `taskkill /IM python.exe` is NEVER acceptable
# — it killed that work on 2026-09-07, and json2kg is running again as of
# 2026-09-09. This script therefore:
#   * marks itself in the environment (THESIS_RUN)
#   * records the real WINDOWS pid of the python doing the work
#
# The pid file used to hold `$!`, which inside Git-Bash is an MSYS pid, not a
# Windows pid: on 2026-09-09 it read 2034 while the run was really 25904 (this
# .venv shim) and its child 25760 (the conda interpreter). taskkill takes
# Windows pids, so the old file pointed at an unrelated process — or at
# nothing. It is now resolved by command line via PowerShell (`wmic` does not
# exist on this machine; Windows 11 26200 dropped it).
#
# Stop it with ./stop_dev.sh, which re-derives the pids from the command line
# and refuses to kill anything it cannot identify.
set -e
cd "$(dirname "$0")"

export OLLAMA_CLOUD_URL=$(grep OLLAMA_CLOUD_URL .env | cut -d= -f2)
export OLLAMA_CLOUD_API_KEY=$(grep OLLAMA_CLOUD_API_KEY .env | cut -d= -f2)
export THESIS_RUN=masterthesis-dev-extraction

.venv/Scripts/python.exe -u scripts/run_extraction.py run \
  --models "cloud:qwen3.6:35b,cloud:hf.co/unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL,cloud:hf.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF:UD-Q4_K_M" \
  --doc-list data/doc_lists/phase1_dev_only.json &

msys_pid=$!

# Resolve the real Windows pid(s). The interpreter takes a moment to appear, so
# poll briefly rather than racing it.
i=0
while [ $i -lt 20 ]; do
  win_pids=$(powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*run_extraction.py*' } | ForEach-Object { \$_.ProcessId }" 2>/dev/null | tr -d '\r' | grep -E "^[0-9]+$" | tr '\n' ' ')
  [ -n "$win_pids" ] && break
  i=$((i + 1))
done

if [ -n "$win_pids" ]; then
  echo "$win_pids" | tr ' ' '\n' | grep -E "^[0-9]+$" > runs/dev_pass.pid
  echo "MASTERTHESIS dev extraction started, windows pid(s) $win_pids(runs/dev_pass.pid)"
else
  : > runs/dev_pass.pid
  echo "MASTERTHESIS dev extraction started (msys pid $msys_pid); could NOT resolve"
  echo "the Windows pid — stop_dev.sh re-derives it from the command line, so"
  echo "stopping still works, but the pid file is empty on purpose rather than wrong."
fi

wait "$msys_pid"
