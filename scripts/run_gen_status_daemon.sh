#!/usr/bin/env bash
# Publish GEN_STATUS.md (synthetic-generation dashboard) to the `status` branch every INTERVAL
# seconds, force-pushed so main's history stays clean. Same pattern as run_status_daemon.sh.
#
#   phone/github view:
#   https://github.com/sakshamio/legal-financial-modernbert-150m/blob/status/GEN_STATUS.md
set -u
cd "$(dirname "$0")/.."
V=~/jupyterlab/.venv/bin/python
INTERVAL="${INTERVAL:-600}"
NTFY_TOPIC="${NTFY_TOPIC:-}"
TARGET=1000000

last_milestone=0

while true; do
  $V scripts/gen_status_page.py > /dev/null 2>&1

  if [ -f GEN_STATUS.md ]; then
    tmp=$(mktemp -d)
    cp GEN_STATUS.md "$tmp/"
    # keep STATUS.md (the training dashboard) on the branch too if it exists
    [ -f STATUS.md ] && cp STATUS.md "$tmp/"
    ( cd "$tmp" && git init -q && git add -A \
      && git -c user.name=sakshamio -c user.email=sakshamio@outlook.com commit -q -m "gen status $(date -u +%H:%MZ)" \
      && git push -q --force "$(cd - >/dev/null && git remote get-url origin)" HEAD:status ) >/dev/null 2>&1
    rm -rf "$tmp"
  fi

  # milestone alerts every 10% of the pair target
  n=$(wc -l < data/pairs_synth_taxonomy/train.jsonl 2>/dev/null || echo 0)
  if [ -n "$NTFY_TOPIC" ] && [ "$n" -gt 0 ]; then
    pct=$(( n * 100 / TARGET ))
    bucket=$(( pct / 10 ))
    if [ "$bucket" -gt "$last_milestone" ]; then
      curl -s -H "Title: Synthetic data ${pct}%" -H "Priority: low" -H "Tags: card_index_dividers" \
        -d "${n} / ${TARGET} pairs generated" "https://ntfy.sh/${NTFY_TOPIC}" >/dev/null || true
      last_milestone=$bucket
    fi
  fi

  sleep "$INTERVAL"
done
