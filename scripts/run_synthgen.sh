#!/usr/bin/env bash
# Self-healing supervisor for the multi-day synthetic-data generation.
#
# WHY A SUPERVISOR. The vLLM server has shut its engine down twice under high concurrency (64 in-flight
# killed the mtp variant; 100 in-flight aborted the non-mtp one). For a run measured in DAYS we cannot
# assume the server stays up, so this loop:
#   * (re)starts the sparkrun server and waits for /v1/models to answer;
#   * runs the generator (which is RESUMABLE -- it counts pairs already on disk and continues);
#   * if the generator exits before the target, checks the server and restarts whatever died.
#
# CONCURRENCY is deliberately conservative (48). The measured throughput ceiling was ~3.7 pairs/s at
# concurrency 160, but that load level is exactly what crashed the engine. ~3.0 pairs/s at 48 that
# stays up for days beats 3.7 pairs/s that dies every hour and has to reload a 35B model each time.
set -u
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
V=~/jupyterlab/.venv/bin/python
RECIPE="@official/qwen3.6-35b-a3b-fp8-vllm"   # non-mtp: more stable than the speculative-decode variant
PORT=8000
BASE="http://127.0.0.1:${PORT}/v1"
TARGET="${TARGET:-1000000}"
CONC="${CONC:-48}"
OUT=data/pairs_synth_taxonomy

server_up() { curl -s "${BASE}/models" 2>/dev/null | grep -q '"id"'; }

start_server() {
  echo "[$(date -u +%H:%MZ)] (re)starting server $RECIPE"
  docker ps --format '{{.Names}}' | grep sparkrun | xargs -r docker stop >/dev/null 2>&1
  sleep 3
  # run the server detached inside THIS supervisor's own tmux-less background
  setsid sparkrun run "$RECIPE" --port "$PORT" >> sparkrun.log 2>&1 &
  for i in $(seq 1 40); do server_up && { echo "  server up after ~$((i*15))s"; return 0; }; sleep 15; done
  echo "  server did NOT come up in 10min"; return 1
}

pairs_on_disk() { wc -l < "$OUT/train.jsonl" 2>/dev/null || echo 0; }

while true; do
  if [ "$(pairs_on_disk)" -ge "$TARGET" ]; then
    echo "[$(date -u +%H:%MZ)] target $TARGET reached ($(pairs_on_disk) pairs). done."
    curl -s -H "Title: Synthetic generation complete" -H "Tags: tada" \
      -d "$(pairs_on_disk) synthetic pairs generated." "https://ntfy.sh/${NTFY_TOPIC:-lfmb-srg5tx}" >/dev/null || true
    break
  fi

  server_up || start_server || { echo "server unrecoverable, sleeping 60s"; sleep 60; continue; }

  echo "[$(date -u +%H:%MZ)] generating: have $(pairs_on_disk)/$TARGET at concurrency $CONC"
  $V -u scripts/gen_synthetic_taxonomy.py \
     --target-pairs "$TARGET" --queries-per-cluster 3 --concurrency "$CONC" \
     --base-url "$BASE" --out "$OUT" >> gensyn.log 2>&1

  # generator returned. either target met (loop top handles it) or the server died mid-run.
  echo "[$(date -u +%H:%MZ)] generator exited at $(pairs_on_disk) pairs; re-checking server"
  sleep 5
done
