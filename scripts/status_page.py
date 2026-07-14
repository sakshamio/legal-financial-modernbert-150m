"""Generate a phone-friendly STATUS.md from the live training run.

Published to a dedicated `status` branch (force-pushed) so that `main`'s history stays clean instead of
accumulating hundreds of status commits. View it on a phone at:

    https://github.com/sakshamio/legal-financial-modernbert-150m/blob/status/STATUS.md

Reads ONLY the log and the checkpoint's trainer_state.json -- never touches the training process.
"""
import json
import math
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
LOG = PROJECT_DIR / "train_stage1.log"
CKPT_DIR = PROJECT_DIR / "checkpoints" / "mlm_stage1"
MAX_STEPS = 271000
EFF_BATCH_TOKENS = 32 * 8 * 1024
TOTAL_TOKENS = MAX_STEPS * EFF_BATCH_TOKENS

# reference points that make the loss mean something (see DIAGNOSTICS.md)
UNIFORM = math.log(50368)
UNIGRAM = 7.42


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        return ""


def latest_state():
    cks = sorted(CKPT_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if not cks:
        return None
    f = cks[-1] / "trainer_state.json"
    return json.loads(f.read_text()) if f.exists() else None


def main():
    log = LOG.read_text(errors="ignore") if LOG.exists() else ""
    prog = re.findall(r"(\d+)/271000 \[([\d:]+)<([\d:]+),\s+([\d.]+)s/it", log)
    losses = [float(x) for x in re.findall(r"'loss': ([\d.]+)", log)]
    lrs = [float(x) for x in re.findall(r"'learning_rate': ([\d.e-]+)", log)]
    evals = [float(x) for x in re.findall(r"'eval_loss': ([\d.]+)", log)]

    # NOT a bare pgrep on the script name -- that matches the tmux server too (see
    # scripts/is_training_alive.sh). A false "alive" is worse than no monitoring at all.
    alive = subprocess.run(["scripts/is_training_alive.sh"], cwd=PROJECT_DIR).returncode == 0
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if not prog:
        (PROJECT_DIR / "STATUS.md").write_text(f"# Status\n\nNo progress data yet. Updated {now}.\n")
        return

    step, elapsed, eta, sit = prog[-1]
    step = int(step)
    rates = [float(r) for *_, r in prog[-12:]]
    sit = statistics.median(rates)
    tps = EFF_BATCH_TOKENS / sit
    pct = 100 * step / MAX_STEPS
    tokens = step * EFF_BATCH_TOKENS
    eta_days = int(eta.split(":")[0]) / 24 if ":" in eta else 0

    loss = losses[-1] if losses else None
    lr = lrs[-1] if lrs else None
    eval_loss = evals[-1] if evals else None

    # progress bar
    filled = int(pct / 100 * 28)
    bar = "█" * filled + "░" * (28 - filled)

    # where the loss sits between the reference points
    if loss:
        span = UNIFORM - 1.75  # 1.75 = middle of the "well-trained encoder" band
        done = max(0.0, min(1.0, (UNIFORM - loss) / span))
        lbar = "█" * int(done * 24) + "░" * (24 - int(done * 24))

    gpu = sh("nvidia-smi --query-gpu=utilization.gpu,power.draw,temperature.gpu --format=csv,noheader")
    mem = sh("free -g | awk '/Mem:/{print $7\"GB free\"}'")
    disk = sh(f"df -h {PROJECT_DIR} | tail -1 | awk '{{print $4\" free\"}}'")
    errors = sh(f"grep -cE 'Traceback|OutOfMemory|CUDA error' {LOG} 2>/dev/null") or "0"
    ckpts = sh(f"ls {CKPT_DIR} 2>/dev/null | grep -c checkpoint") or "0"

    health = "🟢 healthy" if alive and errors == "0" else ("🔴 NOT RUNNING" if not alive else "🟠 errors in log")

    md = f"""# Training status — {health}

`{bar}` **{pct:.2f}%**

| | |
| --- | --- |
| **Step** | **{step:,}** / {MAX_STEPS:,} |
| **Tokens** | {tokens/1e9:.2f}B / {TOTAL_TOKENS/1e9:.0f}B |
| **Speed** | {sit:.2f} s/step → **{tps:,.0f} tok/s** |
| **Elapsed** | {elapsed} |
| **ETA** | **{eta_days:.1f} days** |
| Checkpoints | {ckpts} local · [step-N branches](https://huggingface.co/sakshamio/legal-financial-modernbert-150m/branches) |

## Loss

**{loss:.3f}**{f" · eval {eval_loss:.3f}" if eval_loss else ""} · perplexity **{math.exp(loss):.0f}**

`{lbar}`

| reference | loss | |
| --- | --- | --- |
| random init `ln(50,368)` | 10.83 | start |
| **unigram** (frequencies only) | **7.42** | {"✅ **beaten** — the model is using *context*" if loss < UNIGRAM else "⬅️ not yet"} |
| **current** | **{loss:.2f}** | |
| well-trained encoder | 1.5–2.0 | target |

LR **{lr:.2e}** / 2.00e-04 peak{" *(still warming up)*" if lr and lr < 1.9e-4 and step < 13550 else ""}

## Machine

| | |
| --- | --- |
| GPU | {gpu} |
| Memory | {mem} |
| Disk | {disk} |
| Errors | {errors} |

<sub>Updated {now} · [README](https://github.com/sakshamio/legal-financial-modernbert-150m) · [DIAGNOSTICS](https://github.com/sakshamio/legal-financial-modernbert-150m/blob/main/DIAGNOSTICS.md)</sub>
"""
    (PROJECT_DIR / "STATUS.md").write_text(md)
    print(f"step {step:,} ({pct:.2f}%) loss {loss:.3f} — STATUS.md written")


if __name__ == "__main__":
    main()
