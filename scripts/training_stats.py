"""Snapshot every available training-progress statistic to disk. Meant to be run
on a timer (every 6h) via a self-looping daemon (see run_stats_daemon.sh) since
this box has no cron. Pure read-only inspection -- never touches the training
process, so it can't destabilize it.
"""
import argparse
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
STATS_DIR = PROJECT_DIR / "stats"


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception as e:
        return f"<error: {e}>"


def latest_checkpoint(output_dir):
    ckpts = list(Path(output_dir).glob("checkpoint-*"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: int(p.name.split("-")[1]))


def load_trainer_state(ckpt_dir):
    state_path = Path(ckpt_dir) / "trainer_state.json"
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text())


def summarize_log_history(log_history):
    train_entries = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_entries = [e for e in log_history if "eval_loss" in e]

    eval_losses = [e["eval_loss"] for e in eval_entries]
    latest_eval = eval_losses[-1] if eval_losses else None

    # Health signals -- the point is to CATCH a plateau or divergence, not just record numbers.
    health = "unknown"
    improvement_last_5 = None
    if len(eval_losses) >= 2:
        best = min(eval_losses)
        if latest_eval > best * 1.25:
            health = "DIVERGING"  # eval loss well above its own best -> something is wrong
        elif len(eval_losses) >= 5:
            # relative improvement across the last 5 evals
            old = eval_losses[-5]
            improvement_last_5 = (old - latest_eval) / max(old, 1e-9)
            if improvement_last_5 < 0.001:
                health = "PLATEAU"  # <0.1% improvement over 5 evals
            else:
                health = "improving"
        else:
            health = "improving" if latest_eval <= best else "watch"

    return {
        "latest_train_loss": train_entries[-1]["loss"] if train_entries else None,
        "latest_grad_norm": train_entries[-1].get("grad_norm") if train_entries else None,
        "latest_learning_rate": train_entries[-1].get("learning_rate") if train_entries else None,
        "latest_eval_loss": latest_eval,
        "latest_eval_perplexity": round(math.exp(latest_eval), 2) if latest_eval and latest_eval < 20 else None,
        "best_eval_loss": min(eval_losses) if eval_losses else None,
        "health": health,
        "relative_improvement_last_5_evals": improvement_last_5,
        "recent_train_entries": train_entries[-10:],
        "eval_loss_trend": eval_losses,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=str(PROJECT_DIR / "checkpoints" / "mlm_stage1"))
    ap.add_argument("--tmux-session", default="train")
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--effective-batch-size", type=int, default=256)
    args = ap.parse_args()

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)

    snapshot = {"timestamp_utc": now.isoformat()}

    ckpt = latest_checkpoint(args.output_dir)
    snapshot["latest_checkpoint"] = str(ckpt) if ckpt else None

    state = load_trainer_state(ckpt) if ckpt else None
    if state:
        snapshot["global_step"] = state.get("global_step")
        snapshot["max_steps"] = state.get("max_steps")
        snapshot["epoch"] = state.get("epoch")
        snapshot["percent_complete"] = (
            round(100 * state["global_step"] / state["max_steps"], 2)
            if state.get("max_steps") else None
        )
        snapshot["tokens_seen_est"] = (
            state["global_step"] * args.effective_batch_size * args.block_size
            if state.get("global_step") else None
        )
        snapshot["best_metric"] = state.get("best_metric")
        snapshot.update(summarize_log_history(state.get("log_history", [])))
    else:
        snapshot["note"] = "no checkpoint/trainer_state.json found yet"

    # process / session health
    snapshot["tmux_session_alive"] = args.tmux_session in sh("tmux ls 2>/dev/null")
    snapshot["training_process"] = sh("ps aux | grep '[p]retrain_mlm.py'")

    # system health
    snapshot["memory"] = sh("free -h")
    snapshot["disk"] = sh(f"df -h {PROJECT_DIR}")
    snapshot["gpu"] = sh("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv 2>&1")
    snapshot["load_average"] = sh("uptime")
    snapshot["earlyoom_recent_events"] = sh(
        "journalctl --since '6 hours ago' 2>/dev/null | grep -i 'earlyoom.*low memory\\|killed process' | tail -10"
    )

    # hub push health
    snapshot["output_dir_checkpoints"] = sh(f"ls -la {args.output_dir} 2>&1 | grep checkpoint")

    fname = STATS_DIR / f"snapshot_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    fname.write_text(json.dumps(snapshot, indent=2, default=str))
    print(f"wrote {fname}")


if __name__ == "__main__":
    main()
