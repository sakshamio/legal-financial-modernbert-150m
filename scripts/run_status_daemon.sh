#!/usr/bin/env bash
# Publish a phone-viewable STATUS.md every 10 minutes, and push phone alerts for events that
# actually matter (crash / milestone / completion).
#
# STATUS.md goes to a dedicated `status` branch, force-pushed each time, so `main`'s history is not
# buried under hundreds of status commits.
#
#   phone view : https://github.com/sakshamio/legal-financial-modernbert-150m/blob/status/STATUS.md
#   alerts     : set NTFY_TOPIC to a private-ish random string and subscribe in the ntfy app
#                (https://ntfy.sh -- no account needed). Leave unset to disable alerts.
set -u
cd "$(dirname "$0")/.."
V=~/jupyterlab/.venv/bin/python
INTERVAL="${INTERVAL:-600}"
NTFY_TOPIC="${NTFY_TOPIC:-}"
MAX_STEPS=271000

notify() {  # title, message, priority, tags
  [ -z "$NTFY_TOPIC" ] && return 0
  curl -s -H "Title: $1" -H "Priority: ${3:-default}" -H "Tags: ${4:-bell}" \
       -d "$2" "https://ntfy.sh/${NTFY_TOPIC}" > /dev/null || true
}

last_milestone=0
was_alive=1

while true; do
  $V scripts/status_page.py > /dev/null 2>&1

  # --- publish to the `status` branch (orphan, force-pushed: no history bloat)
  if [ -f STATUS.md ]; then
    git add -f STATUS.md > /dev/null 2>&1
    tree=$(git write-tree 2>/dev/null)
    if [ -n "$tree" ]; then
      # build a single-file commit with no parent, then force-push it
      tmp=$(mktemp -d)
      cp STATUS.md "$tmp/"
      ( cd "$tmp" && git init -q && git add STATUS.md \
        && git -c user.name=sakshamio -c user.email=sakshamio@outlook.com \
             commit -q -m "status: $(date -u +%H:%MZ)" \
        && git push -q --force "$(cd - > /dev/null && git remote get-url origin)" HEAD:status ) > /dev/null 2>&1
      rm -rf "$tmp"
    fi
    git reset -q > /dev/null 2>&1
  fi

  # --- alerts worth waking up for
  alive=$(pgrep -f "[p]retrain_mlm.py" > /dev/null && echo 1 || echo 0)
  step=$(grep -oE "[0-9]+/${MAX_STEPS}" train_stage1.log 2>/dev/null | tail -1 | cut -d/ -f1)
  step=${step:-0}
  loss=$(grep -oE "'loss': [0-9.]+" train_stage1.log 2>/dev/null | tail -1 | grep -oE "[0-9.]+$")

  if [ "$alive" = "0" ] && [ "$was_alive" = "1" ]; then
    if grep -q "train_runtime" train_stage1.log 2>/dev/null; then
      notify "Training COMPLETE" "Finished ${MAX_STEPS} steps. Final loss ${loss:-?}" "high" "tada"
    else
      notify "TRAINING DIED" "Stopped at step ${step}. It did NOT complete." "urgent" "rotating_light"
    fi
  fi
  was_alive=$alive

  # milestone every 10%
  if [ "$step" -gt 0 ]; then
    pct=$(( step * 100 / MAX_STEPS ))
    bucket=$(( pct / 10 ))
    if [ "$bucket" -gt "$last_milestone" ]; then
      notify "Training ${pct}%" "step ${step}/${MAX_STEPS} · loss ${loss:-?}" "low" "chart_with_downwards_trend"
      last_milestone=$bucket
    fi
  fi

  sleep "$INTERVAL"
done
