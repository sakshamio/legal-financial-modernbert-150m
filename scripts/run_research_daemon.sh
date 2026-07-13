#!/usr/bin/env bash
# Capture the longitudinal research measurements at EVERY checkpoint, automatically.
#
# This exists because the data cannot be recovered later. The Trainer keeps only save_total_limit=3
# local checkpoints (~15h of run time), so a checkpoint that is not probed before it is pruned is gone
# for good -- and with it the step-N row of every trajectory plot. The papers on nesting emergence,
# domain differentiation, and concept-before-token are all built from rows that can only be written
# while the run is alive.
#
# Runs diagnostics.py (geometry, frequency buckets, domain tokens) and research_probes.py (nesting
# concentration, domain differentiation, concept-before-token) on each new checkpoint, then commits
# the JSON to git so the trajectory is versioned and survives the box.
#
# CPU only, nice'd. Each pass is a few minutes once per ~5h checkpoint, so the duty cycle against the
# bandwidth-bound GPU run is negligible.
set -u
cd "$(dirname "$0")/.."
V=~/jupyterlab/.venv/bin/python
INTERVAL="${INTERVAL:-600}"

while true; do
  did_work=0
  for ck in $(ls -d checkpoints/mlm_stage1/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
    step=$(basename "$ck" | cut -d- -f2)

    if [ ! -f "research_step${step}.json" ]; then
      echo "[$(date -u +%H:%MZ)] research probes @ step ${step}"
      nice -n 15 $V scripts/research_probes.py --checkpoint "$ck" \
        --passages 256 --per-domain 96 --contexts-per-token 6 >> research.log 2>&1
      did_work=1
    fi

    if [ ! -f "diagnostics_step${step}.json" ]; then
      echo "[$(date -u +%H:%MZ)] diagnostics @ step ${step}"
      nice -n 15 $V scripts/diagnostics.py --checkpoint "$ck" >> research.log 2>&1
      did_work=1
    fi
  done

  # Version the trajectory. These JSONs are a few KB each and they ARE the papers' data.
  if [ "$did_work" = "1" ]; then
    git add -f research_step*.json diagnostics_step*.json 2>/dev/null
    if ! git diff --cached --quiet 2>/dev/null; then
      git -c user.name=sakshamio -c user.email=sakshamio@outlook.com \
          commit -q -m "research: trajectory data through step $(ls research_step*.json 2>/dev/null | sed 's/[^0-9]//g' | sort -n | tail -1)" \
        && git push -q origin main 2>/dev/null \
        && echo "[$(date -u +%H:%MZ)] pushed trajectory data"
    fi
  fi

  sleep "$INTERVAL"
done
