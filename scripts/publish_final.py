"""Publish a FINAL model to the Hub's `main` branch.

`main` is deliberately NOT updated during training: Trainer's push_to_hub overwrites the weights on
every save, which turns `main` into a churning pointer and buries the docs under hundreds of weight
commits. Instead:

  main      -> README, PLAN, tokenizer, and (only at the end) the finished model
  step-N    -> the 28 preserved checkpoints, written by archive_checkpoints.py

Run this once stage-1 (or stage-2) finishes.
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="local checkpoint dir to publish")
    ap.add_argument("--repo-id", default="sakshamio/legal-financial-modernbert-150m")
    ap.add_argument("--message", default="Final stage-1 model")
    args = ap.parse_args()

    src = Path(args.checkpoint)
    missing = [f for f in FILES if not (src / f).exists()]
    if "model.safetensors" in missing or "config.json" in missing:
        raise SystemExit(f"checkpoint incomplete: missing {missing}")

    api = HfApi()
    for f in FILES:
        if (src / f).exists():
            api.upload_file(
                path_or_fileobj=str(src / f),
                path_in_repo=f,
                repo_id=args.repo_id,
                repo_type="model",
                commit_message=args.message,
            )
            print(f"  pushed {f}")
    print(f"\npublished {src} -> {args.repo_id}@main")


if __name__ == "__main__":
    main()
