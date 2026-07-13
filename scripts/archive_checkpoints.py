"""Preserve intermediate training checkpoints as a retrievable series.

Why this exists: the training run pushes to the Hub's `main` branch, which OVERWRITES the weights each
time, and locally `save_total_limit` prunes old checkpoint dirs. Without this, no intermediate snapshot
survives in any addressable form.

For each checkpoint at a target step, this:
  1. copies weights + tokenizer (no optimizer state) to `archive/step-N/` locally, and
  2. pushes that to a Hub branch `step-N`, so anyone can do:
         AutoModel.from_pretrained(REPO, revision="step-10000")

Runs as a decoupled daemon (see run_archiver_daemon.sh) rather than a Trainer callback, so a slow or
failed upload can never stall or crash the multi-day training run. It polls often enough to grab each
checkpoint well before save_total_limit prunes it.

Idempotent and crash-safe: re-running skips steps already archived and pushed.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

from huggingface_hub import HfApi

PROJECT_DIR = Path(__file__).resolve().parent.parent
CKPT_DIR = PROJECT_DIR / "checkpoints" / "mlm_stage1"
ARCHIVE_DIR = PROJECT_DIR / "archive"
STATE_FILE = ARCHIVE_DIR / "_archived.json"

# Weights + tokenizer only. Deliberately excludes optimizer.pt / scheduler.pt / rng_state.pth:
# those are for resuming, not for using the model, and would ~3x the upload size.
ARTIFACT_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def target_steps(max_steps, save_steps, n_tail=20, dense_until=30000):
    """DENSE early, log-spaced, then a regular tail.

    Three regimes, because they answer different questions:

    - DENSE (every save_steps up to `dense_until`): the representation changes fastest here, and the
      research probes measure exactly that -- when nesting starts, when finance leaves the legal cone,
      when the concept-before-token lead peaks. A 14k-step gap could hide a phase transition outright.
      Anything NOT archived is gone for good: save_total_limit=3 prunes it locally after ~15h, so this
      is the only lever that decides what stays reproducible.
    - DOUBLING: cheap coverage across the middle of the run.
    - TAIL: cadence SCALES with run length. A fixed cadence that gave 24 snapshots on an 88k-step run
      would give ~250 on a 271k-step run (~150GB of pushes).

    Only multiples of save_steps are reachable -- those are the steps training actually checkpoints at.
    """
    targets = set()

    # dense early phase -- ~15 extra branches (~9GB), where the publishable dynamics live
    for step in range(save_steps, min(dense_until, max_steps) + 1, save_steps):
        targets.add(step)

    step = save_steps
    while step <= max_steps:  # doubling: 2k, 4k, 8k, 16k, ...
        targets.add(step)
        step *= 2

    tail = max(save_steps, round(max_steps / n_tail / save_steps) * save_steps)
    for step in range(tail, max_steps + 1, tail):
        targets.add(step)

    targets.add(max_steps - (max_steps % save_steps))
    return sorted(s for s in targets if s % save_steps == 0 and s > 0)


def load_state():
    if STATE_FILE.exists():
        return set(json.loads(STATE_FILE.read_text()))
    return set()


def save_state(done):
    STATE_FILE.write_text(json.dumps(sorted(done)))


def archive_step(api, repo_id, ckpt_path, step, push):
    dest = ARCHIVE_DIR / f"step-{step}"
    dest.mkdir(parents=True, exist_ok=True)

    missing = [f for f in ARTIFACT_FILES if not (ckpt_path / f).exists()]
    if "model.safetensors" in missing or "config.json" in missing:
        print(f"  step {step}: checkpoint incomplete (missing {missing}), will retry", flush=True)
        return False
    for fname in ARTIFACT_FILES:
        src = ckpt_path / fname
        if src.exists():
            shutil.copy2(src, dest / fname)
    print(f"  step {step}: archived locally -> {dest}", flush=True)

    if not push:
        return True

    branch = f"step-{step}"
    try:
        api.create_branch(repo_id=repo_id, branch=branch, exist_ok=True)
        api.upload_folder(
            folder_path=str(dest),
            repo_id=repo_id,
            repo_type="model",
            revision=branch,
            commit_message=f"Checkpoint at step {step}",
        )
        print(f"  step {step}: pushed -> {repo_id}@{branch}", flush=True)
        return True
    except Exception as e:
        print(f"  step {step}: PUSH FAILED ({type(e).__name__}: {e}) -- kept locally, will retry", flush=True)
        return False


def scan_once(api, repo_id, targets, done, push):
    if not CKPT_DIR.exists():
        return done
    for ckpt in sorted(CKPT_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])):
        step = int(ckpt.name.split("-")[1])
        if step in done or step not in targets:
            continue
        print(f"found un-archived checkpoint at step {step}", flush=True)
        if archive_step(api, repo_id, ckpt, step, push):
            done.add(step)
            save_state(done)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="sakshamio/legal-financial-modernbert-150m")
    ap.add_argument("--max-steps", type=int, default=88000)
    ap.add_argument("--save-steps", type=int, default=1000)
    ap.add_argument("--poll-seconds", type=int, default=600)
    ap.add_argument("--no-push", action="store_true", help="archive locally only")
    ap.add_argument("--once", action="store_true", help="single pass, don't loop")
    args = ap.parse_args()

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    targets = target_steps(args.max_steps, args.save_steps)
    done = load_state()

    print(f"archiver: {len(targets)} target steps: {targets}", flush=True)
    print(f"already archived: {sorted(done) or 'none'}", flush=True)

    while True:
        try:
            done = scan_once(api, args.repo_id, targets, done, push=not args.no_push)
        except Exception as e:
            print(f"scan error ({type(e).__name__}: {e}) -- continuing", flush=True)
        if args.once or done.issuperset(targets):
            print(f"archiver: done ({len(done)}/{len(targets)} steps archived)", flush=True)
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
