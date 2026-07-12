"""Tokenize + pack the corpus into a flat uint16 memmap.

WHY NOT HF datasets/Arrow: the previous packer round-tripped 130GB through Arrow three times
(tokenize map -> group_texts map -> save_to_disk), which projects to ~7 HOURS at this corpus size. But
our vocab is 50,368 < 65,536, so every token id fits in a uint16. That means the packed corpus is just
a flat array of 2-byte ints, and "packing into blocks" is pure arithmetic -- block i is simply
mmap[i*block : (i+1)*block]. No grouping pass, no Arrow, no serialization. (This is the nanoGPT/litgpt
layout.) It is also FASTER TO READ during training: mmap + OS page cache, zero-copy, versus Arrow.

WHY NOT GO: tokenization is already Rust (HF `tokenizers`) running parallel across cores, so Go adds
nothing -- and a Go tokenizer binding that diverged even slightly from our 156 AddedTokens would
silently corrupt a 30-day run. Go went to dedup, which was genuinely single-threaded Python.

Output:
  data/packed/train.bin   uint16 token stream
  data/packed/val.bin
  data/packed/meta.json   {vocab_size, block_size, n_train_tokens, ...}
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

PROJECT_DIR = Path(__file__).resolve().parent.parent
TOKENIZER_DIR = PROJECT_DIR / "tokenizer"
OUT_DIR = PROJECT_DIR / "data" / "packed"

MAX_DOC_CHARS = 50_000  # outlier docs (up to 16.2M chars) blow worker memory; split, don't truncate

_tok = None


def _init():
    global _tok
    _tok = Tokenizer.from_file(str(TOKENIZER_DIR / "tokenizer.json"))


def _split_long(text):
    if len(text) <= MAX_DOC_CHARS:
        return [text]
    return [text[i : i + MAX_DOC_CHARS] for i in range(0, len(text), MAX_DOC_CHARS)]


def _worker(task):
    """Tokenize one byte-range of one file; write a uint16 shard. Returns (idx, path, n_tokens)."""
    idx, path, start, end, shard_path = task
    texts = []
    with open(path, "rb") as f:
        f.seek(start)
        if start > 0:
            f.readline()  # we may have landed mid-line; the previous shard owns it
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            try:
                texts.extend(_split_long(json.loads(line)["text"]))
            except (json.JSONDecodeError, KeyError):
                continue

    ids = []
    B = 1000
    for i in range(0, len(texts), B):
        for enc in _tok.encode_batch(texts[i : i + B]):
            ids.extend(enc.ids)  # tokenizer.json carries the [CLS]...[SEP] post-processor

    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(shard_path)
    return idx, shard_path, len(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(PROJECT_DIR / "data" / "raw_dedup"))
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--num-proc", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--val-blocks", type=int, default=40_000)
    ap.add_argument("--shard-mb", type=int, default=256, help="input bytes per parallel task")
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(TOKENIZER_DIR / "tokenizer.json"))
    vocab = tok.get_vocab_size()
    assert vocab < 65536, f"vocab {vocab} does not fit in uint16"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shard_dir = OUT_DIR / "_shards"
    shard_dir.mkdir(exist_ok=True)

    files = sorted(glob.glob(str(Path(args.raw_dir) / "*.jsonl")))
    if not files:
        raise SystemExit(f"no corpus in {args.raw_dir}")

    # Split every file into byte-ranges so all cores stay busy even on one huge file.
    tasks, idx = [], 0
    chunk = args.shard_mb * 1024 * 1024
    for path in files:
        size = os.path.getsize(path)
        for start in range(0, size, chunk):
            tasks.append((idx, path, start, min(start + chunk, size), str(shard_dir / f"{idx:06d}.bin")))
            idx += 1
    total_gb = sum(os.path.getsize(f) for f in files) / 1e9
    print(f"corpus {total_gb:.1f}GB -> {len(tasks)} tasks on {args.num_proc} procs (vocab {vocab}, uint16)", flush=True)

    results = []
    with mp.Pool(args.num_proc, initializer=_init) as pool:
        for n, (i, sp, ntok) in enumerate(pool.imap_unordered(_worker, tasks), 1):
            results.append((i, sp, ntok))
            if n % 25 == 0 or n == len(tasks):
                done = sum(r[2] for r in results)
                print(f"  {n}/{len(tasks)} tasks | {done/1e9:.2f}B tokens", flush=True)

    results.sort()  # restore corpus order
    total = sum(r[2] for r in results)
    n_blocks = total // args.block_size
    val_tokens = args.val_blocks * args.block_size
    train_tokens = (n_blocks * args.block_size) - val_tokens
    print(f"\ntotal {total/1e9:.3f}B tokens -> {n_blocks:,} blocks of {args.block_size}", flush=True)

    # Concatenate shards into train.bin / val.bin (val = the tail, held out).
    train_path, val_path = OUT_DIR / "train.bin", OUT_DIR / "val.bin"
    written = 0
    with open(train_path, "wb") as ftr, open(val_path, "wb") as fva:
        for _, sp, ntok in results:
            data = np.fromfile(sp, dtype=np.uint16)
            if written + len(data) <= train_tokens:
                data.tofile(ftr)
            elif written >= train_tokens:
                data.tofile(fva)
            else:  # this shard straddles the boundary
                cut = train_tokens - written
                data[:cut].tofile(ftr)
                data[cut:].tofile(fva)
            written += len(data)
            os.remove(sp)
    shard_dir.rmdir()

    meta = {
        "vocab_size": vocab,
        "block_size": args.block_size,
        "dtype": "uint16",
        "total_tokens": int(total),
        "train_tokens": int(train_tokens),
        "val_tokens": int(written - train_tokens),
        "train_blocks": int(train_tokens // args.block_size),
        "val_blocks": int((written - train_tokens) // args.block_size),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

    # sanity: a packed block must decode back to readable text
    m = np.memmap(train_path, dtype=np.uint16, mode="r")
    print("\nsample block:", tok.decode(m[:80].tolist())[:200])


if __name__ == "__main__":
    main()
