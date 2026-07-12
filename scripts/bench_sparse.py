"""A/B the ModernBERT `sparse_prediction` flag under a REALISTIC MLM setup.

Must use the real DataCollatorForLanguageModeling so ~70% of label positions are -100 -- that's the
whole point of sparse_prediction (gather masked positions before the LM head). Benchmarking with dense
labels would measure nothing.
"""
import argparse
import time

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    DataCollatorForLanguageModeling,
    ModernBertConfig,
    ModernBertForMaskedLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

TOKENIZER_DIR = "/home/sakshamio/legal_fin_embedder/tokenizer"


class StepTimer(TrainerCallback):
    def __init__(self, warmup):
        self.warmup, self.times, self._last = warmup, [], None

    def on_step_end(self, args, state, control, **kwargs):
        now = time.perf_counter()
        if self._last is not None and state.global_step > self.warmup:
            self.times.append(now - self._last)
        self._last = now


def run(sparse, args, tokenizer):
    cfg = ModernBertConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        max_position_embeddings=16384,
        reference_compile=True,
        hidden_size=768,
        num_hidden_layers=22,
        num_attention_heads=12,
        intermediate_size=1152,
        sparse_prediction=sparse,
    )
    model = ModernBertForMaskedLM(cfg)

    total_steps = args.steps + args.warmup
    n = total_steps * args.batch_size * args.grad_accum
    rng = np.random.default_rng(0)
    ids = rng.integers(5, tokenizer.vocab_size, size=(n, args.block_size), dtype=np.int64)
    ds = Dataset.from_dict({"input_ids": ids.tolist()})

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.3)

    targs = TrainingArguments(
        output_dir="/tmp/bench_sparse",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=total_steps,
        learning_rate=2e-4,
        bf16=True,
        optim="adamw_torch_fused",
        logging_steps=10**9,
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=4,
        disable_tqdm=True,
    )
    timer = StepTimer(args.warmup)
    Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator, callbacks=[timer]).train()

    median = float(np.median(timer.times))
    tokens_per_step = args.batch_size * args.grad_accum * args.block_size
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    return median, tokens_per_step / median, peak_mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=8)
    args = ap.parse_args()

    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)

    results = {}
    for sparse in [False, True]:
        print(f"\n===== sparse_prediction={sparse} =====", flush=True)
        step_time, tps, mem = run(sparse, args, tokenizer)
        results[sparse] = (step_time, tps, mem)
        print(f"  step: {step_time*1000:.0f}ms | {tps:,.0f} tok/s | peak mem {mem:.1f}GB", flush=True)

    (t0, tps0, m0), (t1, tps1, m1) = results[False], results[True]
    speedup = tps1 / tps0
    print("\n================ RESULT ================")
    print(f"  sparse=False : {tps0:>7,.0f} tok/s  peak {m0:.1f}GB")
    print(f"  sparse=True  : {tps1:>7,.0f} tok/s  peak {m1:.1f}GB")
    print(f"  speedup      : {speedup:.3f}x  ({(speedup-1)*100:+.1f}%)")
    corpus = 8162899 * 1024
    for label, tps in [("False", tps0), ("True", tps1)]:
        days = 17.0
        toks = tps * days * 86400
        print(f"  sparse={label:<5}: in 17 days -> {toks/1e9:.1f}B tokens | {toks/corpus:.2f} epochs | {toks/149.7e6:.0f} tok/param")


if __name__ == "__main__":
    main()
