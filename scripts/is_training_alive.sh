#!/usr/bin/env bash
# Exit 0 iff a real Python training process is running.
#
# `pgrep -f pretrain_mlm.py` is NOT sufficient: it also matches the tmux SERVER, whose command line
# contains the session's command string. That false positive made the status daemon report "healthy"
# for 26 minutes after an OOM had killed the trainer, and the ntfy "TRAINING DIED" alert never fired.
# The process comm must actually be python.
for p in $(pgrep -f pretrain_mlm.py 2>/dev/null); do
  [ "$(cat /proc/$p/comm 2>/dev/null)" = "python" ] && exit 0
done
exit 1
