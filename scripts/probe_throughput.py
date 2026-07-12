"""Measure real MLM training throughput on this GPU using synthetic data at the
target model/architecture size, before committing to the full multi-day stage-1 run.

Uses random token ids rather than the real corpus/tokenizer -- throughput is
governed by model forward/backward + optimizer cost, not token content, so this
gives an accurate tokens/sec estimate without waiting on corpus prep to finish.
"""
import argparse
import time

import numpy as np
from datasets import Dataset
from transformers import (
    ModernBertConfig,
    ModernBertForMaskedLM,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)

VOCAB_SIZE = 50368


def make_synthetic_dataset(num_samples, block_size, vocab_size, seed=0):
    rng = np.random.default_rng(seed)
    input_ids = rng.integers(low=5, high=vocab_size, size=(num_samples, block_size), dtype=np.int64)
    return Dataset.from_dict({"input_ids": input_ids.tolist(), "labels": input_ids.tolist()})


class StepTimer(TrainerCallback):
    def __init__(self, warmup_steps):
        self.warmup_steps = warmup_steps
        self.times = []
        self._last = None

    def on_step_end(self, args, state, control, **kwargs):
        now = time.perf_counter()
        if self._last is not None and state.global_step > self.warmup_steps:
            self.times.append(now - self._last)
        self._last = now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--per-device-batch-size", type=int, default=32)
    ap.add_argument("--grad-accum-steps", type=int, default=8)
    ap.add_argument("--probe-steps", type=int, default=100)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--target-days", type=float, default=2.5)
    ap.add_argument("--compile", action="store_true", help="enable ModernBERT's reference_compile (torch.compile) path")
    ap.add_argument("--hidden-size", type=int, default=768)
    ap.add_argument("--num-layers", type=int, default=22)
    ap.add_argument("--num-heads", type=int, default=12)
    ap.add_argument("--intermediate-size", type=int, default=1152)
    args = ap.parse_args()

    config = ModernBertConfig(
        vocab_size=VOCAB_SIZE,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        cls_token_id=2,
        sep_token_id=3,
        max_position_embeddings=max(args.block_size, 8192),
        reference_compile=args.compile,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
    )
    model = ModernBertForMaskedLM(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.1f}M")

    total_steps = args.probe_steps + args.warmup_steps
    total_samples = total_steps * args.per_device_batch_size * args.grad_accum_steps
    ds = make_synthetic_dataset(total_samples, args.block_size, VOCAB_SIZE)

    training_args = TrainingArguments(
        output_dir="/tmp/throughput_probe",
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        max_steps=total_steps,
        learning_rate=1e-4,
        bf16=True,
        optim="adamw_torch_fused",
        logging_steps=1_000_000,
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=4,
        disable_tqdm=True,
    )

    timer = StepTimer(args.warmup_steps)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=default_data_collator,
        callbacks=[timer],
    )
    trainer.train()

    step_times = np.array(timer.times)
    median_step_time = float(np.median(step_times))
    effective_batch = args.per_device_batch_size * args.grad_accum_steps
    tokens_per_step = effective_batch * args.block_size
    tokens_per_sec = tokens_per_step / median_step_time

    target_seconds = args.target_days * 86400
    steps_in_budget = int(target_seconds / median_step_time)
    tokens_in_budget = steps_in_budget * tokens_per_step

    print("\n=== throughput probe results ===")
    print(f"median step time: {median_step_time*1000:.1f}ms  (n={len(step_times)} steps measured, after {args.warmup_steps} warmup)")
    print(f"effective batch size: {effective_batch} sequences x {args.block_size} tokens = {tokens_per_step} tokens/step")
    print(f"throughput: {tokens_per_sec:,.0f} tokens/sec")
    print(f"\nat this rate, a {args.target_days}-day budget = {steps_in_budget:,} steps = {tokens_in_budget/1e9:.2f}B tokens seen")


if __name__ == "__main__":
    main()
