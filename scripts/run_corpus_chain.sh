#!/usr/bin/env bash
# Wait for the in-flight domain corpus build, then pull the 20% general slice, then retrain the
# tokenizer on the MIXED corpus (a legal-only BPE would fragment the general text badly).
set -u
cd "$(dirname "$0")"
V=~/jupyterlab/.venv/bin/python

while pgrep -f "[p]repare_corpus.py" > /dev/null; do sleep 60; done
echo "=== domain corpus done; pulling general_web (FineWeb-Edu, 20%) ==="
$V -u prepare_corpus.py --only general_web

echo "=== retraining tokenizer on the MIXED corpus ==="
$V -u train_tokenizer.py --vocab-size 50368 --sample-gb 4.0

echo "=== CHAIN COMPLETE ==="
ls -la ../data/raw/*.jsonl | awk '{printf "  %6.2f GB  %s\n", $5/1e9, $9}'
