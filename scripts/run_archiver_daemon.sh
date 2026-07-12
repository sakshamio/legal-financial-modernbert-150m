#!/usr/bin/env bash
# Archives intermediate checkpoints to local archive/ + HF branches. Decoupled from training
# so upload latency/failures can never stall the run. No cron on this box, so it self-loops.
set -u
cd "$(dirname "$0")"
~/jupyterlab/.venv/bin/python -u archive_checkpoints.py \
  --repo-id sakshamio/legal-financial-modernbert-150m \
  --max-steps 271000 --save-steps 2000 --poll-seconds 600
