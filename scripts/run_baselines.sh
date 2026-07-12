#!/usr/bin/env bash
# Establish external baselines to compare against, on the SAME clean task set.
#
# Baselines run at their NATIVE dim: they are not Matryoshka models, so truncating them is naive
# slicing -- an unfair and uninformative comparison. The exception is nomic-embed-v1.5, which IS a
# Matryoshka model: it gets the full ladder and is the honest head-to-head for our truncation claims.
#
# Runs on CPU with capped threads so it cannot starve a concurrent GPU training run.
set -u
cd "$(dirname "$0")"
V=~/jupyterlab/.venv/bin/python
COMMON="--device cpu --threads 8 --groups legal financial"

echo "### all-MiniLM-L6-v2 (22M) @384"
$V -u eval_benchmarks.py --model sentence-transformers/all-MiniLM-L6-v2 \
   --no-prompts --dims 384 --tag baseline_minilm-l6 $COMMON

echo "### bge-small-en-v1.5 (33M) @384"
$V -u eval_benchmarks.py --model BAAI/bge-small-en-v1.5 \
   --no-prompts --dims 384 --tag baseline_bge-small $COMMON

echo "### bge-base-en-v1.5 (109M) @768  <- closest in size to ours"
$V -u eval_benchmarks.py --model BAAI/bge-base-en-v1.5 \
   --no-prompts --dims 768 --tag baseline_bge-base $COMMON

echo "### nomic-embed-text-v1.5 (137M, MATRYOSHKA) full ladder <- the real head-to-head"
$V -u eval_benchmarks.py --model nomic-ai/nomic-embed-text-v1.5 \
   --no-prompts --trust-remote-code --dims 768 512 256 128 64 \
   --tag baseline_nomic-matryoshka $COMMON

echo "### BASELINES COMPLETE"
