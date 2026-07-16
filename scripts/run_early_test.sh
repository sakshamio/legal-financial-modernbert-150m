#!/usr/bin/env bash
# Autonomous early Stage-2 test window. Generation is paused; the GPU is ours. Runs the full chain,
# then RESUMES generation and pings the phone with the result.
#
# Steps (each needs the GPU, so strictly sequential -- never share it):
#   0. wait for the guide-embedding precompute already running
#   1. reranker-margin precompute (MarginMSE teacher)
#   2. train hybrid Stage 2 (dense+sparse, GIST, distil, latent pooling, Matryoshka)
#   3. MTEB the hybrid dense head vs baselines
#   4. resume generation (server + supervisor) from the snapshot, notify
set -u
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
V=~/jupyterlab/.venv/bin/python
PAIRS=data/pairs_synth_dedup.jsonl
NTFY="${NTFY_TOPIC:-lfmb-srg5tx}"
log(){ echo "[$(date -u +%H:%MZ)] $*"; }

# 0. wait for guide precompute
log "waiting for guide-embedding precompute"
while pgrep -f "[p]recompute_teachers.py --mode guide" >/dev/null; do sleep 30; done
[ -f data/teacher_guide.f16 ] || { log "guide file missing, abort"; exit 1; }
log "guide done"

# 1. reranker margins
log "reranker-margin precompute"
$V scripts/precompute_teachers.py --mode margins --pairs "$PAIRS" \
   --out data/teacher_margins.f16 --device cuda --batch-size 32 --force >> precompute_margins.log 2>&1
[ -f data/teacher_margins.f16 ] || { log "margins file missing, abort"; exit 1; }
log "margins done"

# 2. train hybrid Stage 2 (1 epoch over ~105k pairs)
log "training hybrid Stage 2"
$V scripts/train_stage2_hybrid.py --pairs "$PAIRS" \
   --init checkpoints/mlm_stage1/checkpoint-14000 --device cuda \
   --batch-size 64 --encode-chunk 32 --epochs 1 --flops-lambda 0.01 \
   --guide-emb data/teacher_guide.f16 --margins data/teacher_margins.f16 \
   --output-dir checkpoints/stage2_hybrid_105k >> hybrid_train.log 2>&1
[ -f checkpoints/stage2_hybrid_105k/hybrid.pt ] || { log "hybrid model missing, abort"; exit 1; }
log "training done"

# 3. MTEB
log "MTEB eval"
$V scripts/eval_hybrid.py --model checkpoints/stage2_hybrid_105k --dims 768 256 64 \
   --device cuda --out benchmarks/hybrid_105k.json >> hybrid_eval.log 2>&1
log "eval done"

# 4. resume generation
log "resuming generation"
tmux new-session -d -s synthgen "cd $(pwd) && NTFY_TOPIC=$NTFY TARGET=1000000 CONC=48 ./scripts/run_synthgen.sh >> $(pwd)/synthgen_supervisor.log 2>&1"
sleep 120   # let the supervisor bring the server back up

avg=$($V - <<'PY'
import json,glob
try:
    d=json.load(open("benchmarks/hybrid_105k.json"))
    v=[d[t]["768"] for t in d if "768" in d[t]]
    print(f"{sum(v)/len(v):.3f}" if v else "n/a")
except Exception: print("n/a")
PY
)
log "hybrid MTEB avg @768: $avg  (baselines: MiniLM 0.575, Qwen 0.826, plain-Stage2 0.285)"
curl -s -H "Title: Early hybrid Stage-2 result" -H "Tags: bar_chart" \
  -d "Hybrid on 105k synthetic pairs: MTEB avg @768 = $avg (MiniLM 0.575, plain-S2 0.285). Generation resumed." \
  "https://ntfy.sh/$NTFY" >/dev/null || true
log "DONE"
