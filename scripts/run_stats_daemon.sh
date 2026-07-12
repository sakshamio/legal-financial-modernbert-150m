#!/usr/bin/env bash
# Runs training_stats.py every 6 hours, forever. Meant to run inside its own
# tmux session (no system cron available on this box).
set -u
cd "$(dirname "$0")"
while true; do
  ~/jupyterlab/.venv/bin/python -u training_stats.py
  sleep 21600
done
