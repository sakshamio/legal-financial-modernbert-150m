#!/usr/bin/env bash
# Stream the synthetic dataset to the HF Hub as it is generated.
#
# Re-pushes data/pairs_synth_taxonomy/train.jsonl to the dataset repo every INTERVAL seconds, but only
# when it has actually grown by at least MIN_NEW pairs since the last push -- so the HF dataset viewer
# tracks generation live without spamming identical revisions or re-uploading an unchanged file.
#
# The uploader itself (upload_synthetic_dataset.py) is idempotent, so a push is just "replace the file
# with the current, larger one". HF's content hashing means an unchanged file is a no-op.
set -u
cd "$(dirname "$0")/.."
V=~/jupyterlab/.venv/bin/python
DATA=data/pairs_synth_taxonomy/train.jsonl
INTERVAL="${INTERVAL:-1200}"     # 20 min
MIN_NEW="${MIN_NEW:-2000}"       # only push once 2k new pairs exist

last=0
while true; do
  now=$(wc -l < "$DATA" 2>/dev/null || echo 0)
  if [ "$((now - last))" -ge "$MIN_NEW" ]; then
    echo "[$(date -u +%H:%MZ)] pushing $now pairs to HF (was $last)"
    if $V scripts/upload_synthetic_dataset.py >> hf_stream.log 2>&1; then
      last=$now
    else
      echo "[$(date -u +%H:%MZ)] push failed, will retry next tick"
    fi
  fi
  sleep "$INTERVAL"
done
