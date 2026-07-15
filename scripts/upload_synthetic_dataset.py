"""Push the synthetic retrieval dataset to the HF Hub. Re-runnable: call it again as generation grows.

Uploads the raw train.jsonl plus the dataset card. Idempotent -- re-running replaces the files in the
repo with the current on-disk version, so it doubles as the "update it now that we have more" step.

    python scripts/upload_synthetic_dataset.py                 # push current train.jsonl
    python scripts/upload_synthetic_dataset.py --split-val 5000 # also carve a val split
"""
import argparse
import json
import random
from pathlib import Path

from huggingface_hub import HfApi

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA = PROJECT_DIR / "data" / "pairs_synth_taxonomy" / "train.jsonl"
CARD = PROJECT_DIR / "data_card_synthetic.md"
REPO = "sakshamio/financial-legal-synthetic-retrieval"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--split-val", type=int, default=0, help="hold out N rows as a val split")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not DATA.exists():
        raise SystemExit(f"{DATA} missing -- generate first")

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)

    stage = PROJECT_DIR / "data" / "_hf_upload"
    stage.mkdir(exist_ok=True)

    # CLEAN SNAPSHOT. gogen APPENDS to train.jsonl continuously; uploading it live can capture a
    # half-written final line, which makes the HF dataset viewer choke ("Unexpected token '<'"). So we
    # snapshot into staging, drop any incomplete trailing line, and drop any line that does not parse.
    clean = stage / "_clean.jsonl"
    kept = dropped = 0
    with open(DATA, "rb") as src, open(clean, "w") as dst:
        data = src.read()
        # ignore anything after the last newline (a partial in-progress append)
        if not data.endswith(b"\n"):
            data = data[: data.rfind(b"\n") + 1]
        for line in data.decode("utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                json.loads(line)
            except Exception:
                dropped += 1
                continue
            dst.write(line + "\n")
            kept += 1
    DATA_CLEAN = clean
    n = kept
    print(f"{n:,} valid pairs snapshotted ({dropped} malformed dropped)")

    if args.split_val > 0:
        # deterministic val carve so the split is reproducible and disjoint
        rng = random.Random(args.seed)
        rows = DATA_CLEAN.read_text().splitlines()
        rng.shuffle(rows)
        val, train = rows[: args.split_val], rows[args.split_val :]
        (stage / "train.jsonl").write_text("\n".join(train) + "\n")
        (stage / "val.jsonl").write_text("\n".join(val) + "\n")
        print(f"  split -> train {len(train):,} / val {len(val):,}")
        files = ["train.jsonl", "val.jsonl"]
    else:
        (stage / "train.jsonl").write_bytes(DATA_CLEAN.read_bytes())
        files = ["train.jsonl"]

    for f in files:
        api.upload_file(path_or_fileobj=str(stage / f), path_in_repo=f,
                        repo_id=args.repo, repo_type="dataset")
        print(f"  uploaded {f}")

    api.upload_file(path_or_fileobj=str(CARD), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="dataset")
    print(f"  uploaded README.md (dataset card)")
    print(f"\nhttps://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
