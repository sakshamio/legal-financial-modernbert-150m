#!/usr/bin/env bash
# Gentle supervisor for the synthetic-data generation.
#
# THE BUG THIS REPLACES. The previous version called `docker stop` inside start_server whenever the
# server "looked down". A transient request failure under load was enough to trigger it, and the
# SIGTERM killed a perfectly healthy vLLM engine mid-generation. The engine logs proved it: 48 reqs
# at 230 tok/s, then `signal=SIGTERM`, then the in-flight requests 500 and the engine dies. Both the
# 35B MoE and the 27B dense "crashes" were SELF-INFLICTED -- the models were fine, my orchestration
# was shooting them.
#
# THE RULES NOW:
#   * NEVER docker-stop or signal a running server. The server is launched once and left alone.
#   * Only (re)launch a server when it has been genuinely unreachable for several consecutive checks.
#   * The generator retries transient errors internally; if it exits early it is simply re-run
#     (resumable), WITHOUT touching the server.
set -u
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
V=~/jupyterlab/.venv/bin/python
RECIPE="${RECIPE:-@official/qwen3.6-27b-fp8-vllm}"   # dense 27B; MoE was never the real problem
PORT=8000
BASE="http://127.0.0.1:${PORT}/v1"
TARGET="${TARGET:-1000000}"
CONC="${CONC:-48}"
OUT=data/pairs_synth_taxonomy

server_up() { curl -s --max-time 8 "${BASE}/models" 2>/dev/null | grep -q '"id"'; }
pairs_on_disk() { wc -l < "$OUT/train.jsonl" 2>/dev/null || echo 0; }

launch_server() {
  # Launch ONLY if nothing is serving. Do NOT stop anything -- that was the bug.
  if server_up; then return 0; fi
  echo "[$(date -u +%H:%MZ)] no server; launching $RECIPE"
  setsid sparkrun run "$RECIPE" --port "$PORT" >> sparkrun27b.log 2>&1 &
  for i in $(seq 1 80); do server_up && { echo "  up after ~$((i*15))s"; return 0; }; sleep 15; done
  echo "  did not come up in 20min"; return 1
}

down_streak=0
while true; do
  if [ "$(pairs_on_disk)" -ge "$TARGET" ]; then
    echo "[$(date -u +%H:%MZ)] target reached ($(pairs_on_disk))"
    curl -s -H "Title: Synthetic generation complete" -H "Tags: tada" \
      -d "$(pairs_on_disk) pairs." "https://ntfy.sh/${NTFY_TOPIC:-lfmb-srg5tx}" >/dev/null || true
    break
  fi

  if ! server_up; then
    down_streak=$((down_streak + 1))
    # require 4 consecutive failures (~2 min) before concluding the server is really gone -- a single
    # blip under load must NEVER trigger a restart.
    if [ "$down_streak" -ge 4 ]; then
      launch_server && down_streak=0 || { sleep 30; continue; }
    else
      echo "[$(date -u +%H:%MZ)] server blip ${down_streak}/4, waiting"
      sleep 30; continue
    fi
  fi
  down_streak=0

  echo "[$(date -u +%H:%MZ)] generating: $(pairs_on_disk)/$TARGET @ conc $CONC"
  # Go generator: bounded goroutine pool -> flat memory (the Python version leaked 60GB and tripped
  # earlyoom, which killed the server). scripts/gen_synthetic_taxonomy.py stays as the reference.
  gogen/gogen -target "$TARGET" -nq 3 -conc "$CONC" -base "$BASE" -out "$OUT" >> gensyn.log 2>&1

  # generator returned: either target met (top of loop) or it stopped early. Re-run WITHOUT touching
  # the server. A brief pause avoids a hot spin if it is failing fast.
  echo "[$(date -u +%H:%MZ)] generator returned at $(pairs_on_disk); continuing"
  sleep 10
done
