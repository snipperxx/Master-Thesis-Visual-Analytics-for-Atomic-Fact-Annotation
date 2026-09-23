#!/bin/sh
# Stop the thesis dev extraction pass — and ONLY it.
#
# Never `taskkill /IM python.exe`: another repo (fiz-kg/json2kg) runs its own
# python.exe on this machine and was killed that way on 2026-09-07. As of
# 2026-09-09 it is running again (`python -m json2kg.serve`), so image-name
# kills stay permanently off the table.
#
# TWO BUGS FOUND 2026-09-09, both in the safety check itself:
#  1. `wmic` DOES NOT EXIST on this machine (Windows 11 26200 dropped it). The
#     previous version guarded its verification with `if command -v wmic`, so
#     on this box it skipped verification entirely and ran taskkill on the
#     recorded PID unchecked — the exact hazard the script exists to prevent.
#     Identification is now done with PowerShell, and a failure to identify
#     REFUSES instead of falling through.
#  2. `runs/dev_pass.pid` holds what `$!` reported inside Git-Bash, which is an
#     MSYS pid, not a Windows pid — on 2026-09-09 it said 2034 while the run
#     was really 25904 (the .venv shim) and its child 25760 (the conda
#     interpreter doing the work). So the PID file is a hint only; the command
#     line decides.
cd "$(dirname "$0")"

# Every python whose command line mentions run_extraction.py. That string
# cannot appear in the other project's processes, so this cannot hit them.
pids=$(powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*run_extraction.py*' } | ForEach-Object { \$_.ProcessId }" 2>/dev/null | tr -d '\r' | grep -E "^[0-9]+$")

if [ -z "$pids" ]; then
  if ! powershell.exe -NoProfile -Command "exit 0" >/dev/null 2>&1; then
    echo "cannot query processes (no working PowerShell) — refusing to kill anything"
    exit 1
  fi
  echo "no python running scripts/run_extraction.py — nothing to stop"
  rm -f runs/dev_pass.pid
  exit 0
fi

recorded=$(cat runs/dev_pass.pid 2>/dev/null || echo "-")
echo "stopping extraction pid(s): $(echo $pids | tr '\n' ' ')(pid file said: $recorded)"
echo "=== stopped at $(date '+%Y-%m-%d %H:%M') via stop_dev.sh (pids $(echo $pids | tr '\n' ' ')) ===" \
  >> runs/dev_pass_"$(date +%Y%m%d)".log 2>/dev/null || true

# /T so the shim takes its conda child with it; every PID was verified above.
for pid in $pids; do
  taskkill //F //T //PID "$pid" 2>&1 | tail -1
done
rm -f runs/dev_pass.pid
