"""Does using more of the 119GB unified memory buy any throughput?

Holds the EFFECTIVE batch fixed at 256 sequences and shifts work from gradient-accumulation steps into
the per-device micro-batch. If the box is bandwidth-bound, per-token cost is constant and this changes
nothing except memory use. If there's meaningful kernel-launch / fixed-overhead amortization, bigger
micro-batches win. Run with sparse_prediction=True (the config we actually ship), which freed ~32% of
memory and makes the larger micro-batches reachable at all.
"""
import argparse

import torch
from transformers import PreTrainedTokenizerFast

from bench_sparse import run

TOKENIZER_DIR = "/home/sakshamio/legal_fin_embedder/tokenizer"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--effective-batch", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 128])
    args = ap.parse_args()

    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
    results = []

    for bs in args.batch_sizes:
        if args.effective_batch % bs != 0:
            continue
        ga = args.effective_batch // bs
        print(f"\n===== per-device batch {bs} x grad_accum {ga} (effective {args.effective_batch}) =====", flush=True)
        sub = argparse.Namespace(
            block_size=args.block_size, batch_size=bs, grad_accum=ga, steps=args.steps, warmup=args.warmup
        )
        try:
            step_time, tps, mem = run(True, sub, tokenizer)
            results.append((bs, ga, tps, mem))
            print(f"  {tps:,.0f} tok/s | peak {mem:.1f}GB", flush=True)
        except torch.cuda.OutOfMemoryError:
            print("  OOM", flush=True)
            results.append((bs, ga, None, None))
        torch.cuda.empty_cache()

    print("\n================ RESULT (effective batch fixed) ================")
    print(f"{'micro-batch':>12} {'grad_accum':>11} {'tok/s':>10} {'peak GB':>9} {'vs bs=32':>10}")
    base = next((r[2] for r in results if r[0] == args.batch_sizes[0] and r[2]), None)
    for bs, ga, tps, mem in results:
        if tps is None:
            print(f"{bs:>12} {ga:>11} {'OOM':>10} {'-':>9} {'-':>10}")
        else:
            rel = f"{(tps/base-1)*100:+.1f}%" if base else "-"
            print(f"{bs:>12} {ga:>11} {tps:>10,.0f} {mem:>9.1f} {rel:>10}")


if __name__ == "__main__":
    main()
