"""Does FP8 training beat bf16 on GB10?

The workload is memory-bandwidth-bound (measured MFU ~9-19%), so the lever that matters is BYTES MOVED
per token, not FLOPs. FP8 halves the bytes for linear-layer weights and activations, which is exactly
the right kind of win here -- unlike bigger batches, which move MORE bytes for no gain.

Runs in the torch-2.13 venv: torch._scaled_mm (native FP8) works on sm_121, and torchao's float8
TRAINING path rides on it rather than on torchao's own (aarch64-missing) kernels.

Manual training loop on purpose: keeps the benchmark independent of Trainer/transformers API drift.
"""
import argparse
import time

import torch
from transformers import ModernBertConfig, ModernBertForMaskedLM

VOCAB = 50368
MASK_ID = 4
PAD_ID = 0


def build(sparse=True, compile_=True):
    cfg = ModernBertConfig(
        vocab_size=VOCAB,
        pad_token_id=PAD_ID,
        bos_token_id=2,
        eos_token_id=3,
        cls_token_id=2,
        sep_token_id=3,
        max_position_embeddings=16384,
        hidden_size=768,
        num_hidden_layers=22,
        num_attention_heads=12,
        intermediate_size=1152,
        reference_compile=compile_,
        sparse_prediction=sparse,
    )
    return ModernBertForMaskedLM(cfg).cuda()


def to_fp8(model):
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    def filt(mod, fqn):
        # FP8 GEMMs need both dims divisible by 16. Skip the decoder/LM head: with sparse_prediction it
        # sees only ~30% of positions, and its vocab-sized output is the numerically touchiest part.
        if "decoder" in fqn or "head" in fqn:
            return False
        return all(d % 16 == 0 for d in (mod.in_features, mod.out_features))

    convert_to_float8_training(model, config=Float8LinearConfig(), module_filter_fn=filt)
    n = sum(1 for m in model.modules() if type(m).__name__ == "Float8Linear")
    print(f"  converted {n} Linear layers -> Float8Linear", flush=True)
    return model


def bench(model, args, label):
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, fused=True)
    g = torch.Generator(device="cuda").manual_seed(0)
    times = []
    torch.cuda.reset_peak_memory_stats()

    for step in range(args.warmup + args.steps):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            ids = torch.randint(5, VOCAB, (args.batch_size, args.block_size), device="cuda", generator=g)
            labels = ids.clone()
            keep = torch.rand(ids.shape, device="cuda", generator=g) < args.mlm_prob
            labels[~keep] = -100          # only masked positions carry loss (~30%)
            ids[keep] = MASK_ID
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, labels=labels).loss
            (loss / args.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        if step >= args.warmup:
            times.append(time.perf_counter() - t0)

    times.sort()
    median = times[len(times) // 2]
    tok_per_step = args.batch_size * args.grad_accum * args.block_size
    tps = tok_per_step / median
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  {label:<10} {tps:>9,.0f} tok/s | step {median*1000:>7.0f}ms | peak {mem:>5.1f}GB | loss {loss.item():.3f}", flush=True)
    return tps, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--mlm-prob", type=float, default=0.3)
    args = ap.parse_args()

    print(f"torch {torch.__version__} | sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}\n")

    results = {}

    print("=== A: bf16 + reference_compile (current production config) ===")
    m = build(compile_=True)
    results["bf16 (prod)"] = bench(m, args, "bf16")
    del m; torch.cuda.empty_cache()

    # torchao float8 only pays off when torch.compile fuses the scaling/cast ops it inserts around
    # each Linear. In eager they are pure overhead. Full-graph compile is torchao's documented path.
    print("\n=== B: bf16 + FULL torch.compile ===")
    m = build(compile_=False)
    m = torch.compile(m)
    results["bf16 (full compile)"] = bench(m, args, "bf16+c")
    del m; torch.cuda.empty_cache()

    print("\n=== C: FP8 + FULL torch.compile (torchao's supported path) ===")
    m = build(compile_=False)
    m = to_fp8(m)
    m = torch.compile(m)
    results["fp8 (full compile)"] = bench(m, args, "fp8+c")
    del m; torch.cuda.empty_cache()

    print("\n================ RESULT ================")
    base = results["bf16 (prod)"][0]
    for label, (tps, mem) in results.items():
        toks30 = tps * 30 * 86400
        print(f"  {label:<22} {tps:>9,.0f} tok/s | {mem:>5.1f}GB | {(tps/base-1)*100:>+6.1f}% | 30d: {toks30/1e9:>5.1f}B tok, {toks30/149.7e6:>4.0f} tok/param")


if __name__ == "__main__":
    main()
