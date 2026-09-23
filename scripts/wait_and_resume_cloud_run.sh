#!/usr/bin/env bash
# One-off helper: poll the DBvis cloud Ollama endpoint until /api/generate
# actually answers again (2026-09-02 it started returning connection resets
# on every request, tags still 200 — see doc/CHANGELOG.md 2026-09-02), then
# resume the cloud-only dev extraction run. Not part of the permanent
# pipeline; delete after the run finishes or once superseded.
set -u
cd "$(dirname "$0")/.."

API_KEY=$(grep OLLAMA_CLOUD_API_KEY .env | cut -d= -f2 | tr -d '\r')
URL="$(grep OLLAMA_CLOUD_URL .env | cut -d= -f2 | tr -d '')/api/generate"
LOG="data/facts/wait_and_resume_20260902.log"

echo "$(date -u +%FT%TZ) waiting for ${URL} to answer /api/generate again" | tee -a "$LOG"

attempt=0
while true; do
  attempt=$((attempt+1))
  code=$(curl -s -m 45 -o /tmp/probe_resp.json -w "%{http_code}" \
    -H "Authorization: Bearer ${API_KEY}" -H "Content-Type: application/json" \
    -d '{"model":"qwen3.6:35b","prompt":"hi","stream":false}' "$URL")
  if [ "$code" = "200" ] && grep -q '"response"' /tmp/probe_resp.json 2>/dev/null; then
    echo "$(date -u +%FT%TZ) attempt ${attempt}: endpoint answered (HTTP $code) — resuming run" | tee -a "$LOG"
    break
  fi
  echo "$(date -u +%FT%TZ) attempt ${attempt}: not ready (HTTP ${code:-timeout})" | tee -a "$LOG"
  sleep 300
done

".venv/Scripts/python.exe" -m scripts.run_extraction run \
  --models "cloud:qwen3.6:35b,cloud:hf.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF:UD-Q4_K_M,cloud:hf.co/unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL" \
  --doc-list data/doc_lists/eurlex_dev.json \
  --out data/facts \
  --manifest data/facts/manifest.json \
  >> "data/facts/run_dev_cloud_20260902.log" 2>> "data/facts/run_dev_cloud_20260902.err.log"

echo "$(date -u +%FT%TZ) run_extraction exited with code $?" | tee -a "$LOG"
