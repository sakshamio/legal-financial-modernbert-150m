#!/usr/bin/env bash
# The eval window: run EVERY GPU evaluation while the GPU is free, then resume Stage 1.
#
# WHY THIS EXISTS. Running a GPU eval alongside training OOM-killed the trainer at step 13,384 (on
# unified memory the two jobs together exceeded 121GB and the kernel took the largest process). That
# was the SECOND time -- a batch-128 benchmark had already OOM-killed corpus prep earlier. The lesson
# was written down and then ignored, so it is now enforced by structure rather than by memory:
#
#   * every eval runs SEQUENTIALLY, never concurrently with another GPU job;
#   * training is only resumed AFTER the last eval finishes;
#   * the script refuses to start if a trainer is already running.
#
# On the GPU these evals take minutes. On CPU, alongside training, FiQA alone ran for over an hour
# and never finished -- so this window is also the only way to get the financial benchmarks at all.
set -u
cd "$(dirname "$0")/.."
V=~/jupyterlab/.venv/bin/python
S2=checkpoints/embedding_stage2_from12k
CK1=checkpoints/mlm_stage1/checkpoint-12000
LOG=eval_window.log

banner() { echo "" | tee -a $LOG; echo "=== $* ===" | tee -a $LOG; }

if ./scripts/is_training_alive.sh; then
  echo "REFUSING: a trainer is running. This window exists precisely to avoid that overlap."
  exit 1
fi

# 1. wait for stage 2 to finish (it owns the GPU until then)
while pgrep -f "[t]rain_matryoshka.py" > /dev/null; do sleep 60; done
banner "stage 2 complete: $(ls -d $S2 2>/dev/null || echo MISSING)"

# ---------------------------------------------------------------- the actual question
banner "1/6  MTEB -- STAGE 2 MODEL (all dims, financial + legal)"
$V scripts/eval_benchmarks.py --model "$S2" --device cuda \
   --tag stage2_from12k >> $LOG 2>&1

banner "2/6  MTEB -- STAGE 1 RAW ENCODER (the 'before' picture)"
$V scripts/eval_benchmarks.py --model "$CK1" --device cuda --no-prompts --dims 768 \
   --tag stage1_ck12000 >> $LOG 2>&1

banner "3/6  MTEB -- BASELINES (same harness: the only fair comparison)"
$V scripts/eval_benchmarks.py --model sentence-transformers/all-MiniLM-L6-v2 --device cuda \
   --no-prompts --dims 384 --tag minilm >> $LOG 2>&1
$V scripts/eval_benchmarks.py --model BAAI/bge-base-en-v1.5 --device cuda \
   --no-prompts --dims 768 --tag bge_base >> $LOG 2>&1
$V scripts/eval_benchmarks.py --model Qwen/Qwen3-Embedding-0.6B --device cuda \
   --no-prompts --dims 1024 --tag qwen3_emb >> $LOG 2>&1

banner "4/6  held-out retrieval on OUR pairs, per Matryoshka dim"
$V scripts/eval_retrieval.py --model-dir "$S2" >> $LOG 2>&1

banner "5/6  LEDGAR zero-shot -- stage 2 vs stage 1 (does contrastive training help here too?)"
$V scripts/eval_zeroshot.py --checkpoint "$CK1" --n 1500 --device cuda --force >> $LOG 2>&1

banner "6/6  CLEAN throughput numbers (uncontended -- the earlier ones were distorted)"
$V scripts/bench_context.py --steps 8 >> $LOG 2>&1

# ---------------------------------------------------------------- back to work
banner "evals done -- resuming Stage 1 from checkpoint-12000"
tmux new-session -d -s train "cd $(pwd)/scripts && $V -u pretrain_mlm.py \
  --max-steps 271000 --per-device-batch-size 32 --grad-accum-steps 8 \
  --learning-rate 2e-4 --warmup-ratio 0.05 --decay-ratio 0.1 \
  --mlm-probability 0.3 --block-size 1024 \
  --save-steps 2000 --eval-steps 2000 --eval-subset-size 2000 --logging-steps 50 \
  --output-dir $(pwd)/checkpoints/mlm_stage1 \
  >> $(pwd)/train_stage1.log 2>&1"

sleep 90
if ./scripts/is_training_alive.sh; then
  echo "training resumed OK" | tee -a $LOG
  curl -s -H "Title: Eval window done, training resumed" -H "Priority: default" -H "Tags: white_check_mark" \
       -d "Stage 2 + all GPU evals complete. Stage 1 resumed from checkpoint-12000." \
       "https://ntfy.sh/${NTFY_TOPIC:-lfmb-srg5tx}" > /dev/null || true
else
  echo "!!! TRAINING FAILED TO RESUME" | tee -a $LOG
  curl -s -H "Title: TRAINING FAILED TO RESUME" -H "Priority: urgent" -H "Tags: rotating_light" \
       -d "Eval window finished but Stage 1 did not restart. Needs a human." \
       "https://ntfy.sh/${NTFY_TOPIC:-lfmb-srg5tx}" > /dev/null || true
fi
