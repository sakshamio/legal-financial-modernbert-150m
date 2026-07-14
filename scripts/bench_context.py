"""Does longer context cost wall-clock on THIS box, or is it nearly free?

The question is not academic. Only the 7 GLOBAL layers scale with sequence length -- the other 15 are
pinned at a 128-token window -- so at 16k, attention is 62% of FLOPs (vs 11% at 1k) and total
FLOPs/token rise 2.3x. But attention is COMPUTE, which this bandwidth-starved chip has in surplus,
while weight-loading is what it lacks. So two effects pull in opposite directions:

    FLOPs/token   UP  2.3x        (more work per token)
    MFU           UP  ?           (work shifts to high-intensity attention)

If MFU rises as fast as FLOPs do, long context is FREE in wall-clock per token. That would be a
striking result and it decides whether we extend to 8k or 16k. Theory cannot answer it; only a
measurement can.

DESIGN: hold TOKENS PER MICRO-BATCH CONSTANT (32,768 -- exactly the live config, 32 x 1024) and vary
only the sequence length. Batch shrinks as S grows. This isolates context length instead of
confounding it with batch size, and keeps activation memory roughly comparable.

SAFETY: this contends with a live 27-day training run on UNIFIED memory. An OOM here does not just
fail the benchmark -- the kernel OOM killer may take down TRAINING (it killed corpus prep once
already). So we check free memory before every config, keep a hard reserve, and back off on OOM
rather than pushing.

    python scripts/bench_context.py --force
"""
import argparse
import gc
import json
import subprocess
import time
from pathlib import Path

import torch
from transformers import ModernBertConfig, ModernBertForMaskedLM

PROJECT_DIR = Path(__file__).resolve().parent.parent
CKPT_DIR = PROJECT_DIR / "checkpoints" / "mlm_stage1"

TOKENS_PER_MICRO = 32 * 1024        # the live config: micro-batch 32 at seq 1024
RESERVE_GB = 25                     # never let free memory fall below this; training needs headroom
PEAK_TFLOPS = 200.0


def free_gb():
    out = subprocess.run("free -g | awk '/Mem:/{print $7}'", shell=True,
                         capture_output=True, text=True).stdout.strip()
    return int(out or 0)


def flops_per_token(cfg, S):
    d, L, inter = cfg.hidden_size, cfg.num_hidden_layers, cfg.intermediate_size
    glob = L // cfg.global_attn_every_n_layers
    loc = L - glob
    w = cfg.local_attention if isinstance(cfg.local_attention, int) else 128
    n_nonemb = L * (4 * d * d + 3 * d * inter)
    attn = 12 * d * (glob * S + loc * w)     # only GLOBAL layers scale with S
    return 6 * n_nonemb + attn


def bench(model, cfg, S, batch, steps, device):
    V = cfg.vocab_size
    ids = torch.randint(0, V, (batch, S), device=device)
    labels = ids.clone()
    labels[torch.rand_like(labels, dtype=torch.float) > 0.3] = -100  # 30% masking, as in training
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for i in range(steps + 2):
        if i == 2:                                   # warmup done
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    el = time.perf_counter() - t0
    return (steps * batch * S) / el


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--seqs", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384])
    args = ap.parse_args()

    alive = subprocess.run("pgrep -f '[p]retrain_mlm.py'", shell=True, capture_output=True).stdout
    if alive and not args.force:
        raise SystemExit("training is running; this benchmark contends. Pass --force if intended.")

    cks = sorted(CKPT_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    cfg = ModernBertConfig.from_pretrained(str(cks[-1]))
    cfg.reference_compile = False
    cfg.sparse_prediction = True                     # as in training: +23.6% and -32% memory

    print(f"config: {cfg.num_hidden_layers}L d={cfg.hidden_size} "
          f"max_pos={cfg.max_position_embeddings} global_every={cfg.global_attn_every_n_layers}")
    print(f"holding tokens/micro-batch constant at {TOKENS_PER_MICRO:,}\n")
    print(f"  {'seq':>7} {'batch':>6} {'tok/s':>9} {'TFLOPS':>8} {'MFU':>6} {'vs 1k':>7}  {'free GB':>8}")

    rows, base = [], None
    for S in args.seqs:
        if S > cfg.max_position_embeddings:
            print(f"  {S:>7,}  SKIP -- exceeds max_position_embeddings={cfg.max_position_embeddings}")
            continue
        batch = max(1, TOKENS_PER_MICRO // S)

        fg = free_gb()
        if fg < RESERVE_GB:
            print(f"  {S:>7,}  SKIP -- only {fg}GB free, reserve is {RESERVE_GB}GB (protecting training)")
            continue

        model = ModernBertForMaskedLM(cfg).to("cuda")
        try:
            tps = bench(model, cfg, S, batch, args.steps, "cuda")
            tf = tps * flops_per_token(cfg, S) / 1e12
            mfu = 100 * tf / PEAK_TFLOPS
            if base is None:
                base = tps
            print(f"  {S:>7,} {batch:>6} {tps:>9,.0f} {tf:>8.1f} {mfu:>5.1f}% {tps/base:>6.2f}x  {free_gb():>7}G")
            rows.append({"seq": S, "batch": batch, "tok_s": tps, "tflops": tf, "mfu": mfu})
        except torch.cuda.OutOfMemoryError:
            print(f"  {S:>7,} {batch:>6}  OOM -- backing off (training protected)")
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    if rows:
        (PROJECT_DIR / "bench_context.json").write_text(json.dumps(rows, indent=2))
        print("\n  tok/s is what determines wall-clock. If it falls SLOWER than FLOPs/token rise,")
        print("  long context is buying compute efficiency -- i.e. it is partly free on this box.")


if __name__ == "__main__":
    main()
