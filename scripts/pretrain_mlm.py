"""Stage 1: MLM-pretrain a randomly-initialized ModernBERT-style encoder from scratch.

Resumable: re-running with the same --output-dir will auto-resume from the last
checkpoint if one exists (standard `transformers.Trainer` behavior).
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import (
    DataCollatorForLanguageModeling,
    ModernBertConfig,
    ModernBertForMaskedLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
TOKENIZER_DIR = PROJECT_DIR / "tokenizer"
PACKED_DIR = PROJECT_DIR / "data" / "packed"
CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"


class PackedDataset(torch.utils.data.Dataset):
    """Fixed-size blocks read straight out of a flat uint16 memmap.

    The corpus is one contiguous token stream (see pack_fast.py), so a "block" is just a slice --
    no Arrow, no per-example deserialization. Reads are zero-copy through the OS page cache, which
    matters on a bandwidth-bound box: the dataloader must never become the bottleneck.
    """

    def __init__(self, path, block_size):
        self.path = str(path)
        self.block_size = block_size
        self.n_blocks = os.path.getsize(self.path) // 2 // block_size  # uint16 = 2 bytes
        self._data = None  # opened lazily: a memmap cannot be forked into dataloader workers

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, i):
        if self._data is None:
            self._data = np.memmap(self.path, dtype=np.uint16, mode="r")
        s = i * self.block_size
        block = self._data[s : s + self.block_size].astype(np.int64)
        return {"input_ids": torch.from_numpy(block)}


def build_model(
    tokenizer, max_position_embeddings, hidden_size, num_layers, num_heads, intermediate_size, compile_, sparse
):
    config = ModernBertConfig(
        # len(tokenizer), NOT tokenizer.vocab_size. `.vocab_size` reports ONLY the learned BPE (50,196)
        # and EXCLUDES added tokens -- so the embedding matrix would be 172 rows too small and every
        # domain/reserved token (ids 50,196..50,367) would index out of bounds. The MLM collator also
        # emits random ids up to len(tokenizer) for its random-replacement step.
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        max_position_embeddings=max_position_embeddings,
        reference_compile=compile_,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=intermediate_size,
        # Gather masked positions BEFORE the LM head. Only ~30% of positions carry loss, so the dense
        # path projects 32k positions x 50,368 vocab and discards ~70% of it. On this bandwidth-bound
        # box that head is the heaviest op: measured +23.6% tok/s and -32% peak memory.
        sparse_prediction=sparse,
    )
    return ModernBertForMaskedLM(config)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--output-dir", default=str(CHECKPOINTS_DIR / "mlm_stage1"))
    ap.add_argument("--init-from", default=None, help="checkpoint dir to init weights from (for context-extension phase)")
    ap.add_argument("--per-device-batch-size", type=int, default=32)
    ap.add_argument("--grad-accum-steps", type=int, default=8)
    ap.add_argument("--max-steps", type=int, required=True)
    # ModernBERT-base used ~8e-4 at a ~4.7M-token batch. Ours is ~262K tokens (18x smaller);
    # sqrt-scaling gives 8e-4/sqrt(18) ~= 1.9e-4. Watch the first few hundred steps for divergence.
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--decay-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--mlm-probability", type=float, default=0.3)
    ap.add_argument("--logging-steps", type=int, default=20)
    ap.add_argument("--save-steps", type=int, default=1000)
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument(
        "--eval-subset-size",
        type=int,
        default=2000,
        help="evaluate on a fixed random subset of the val blocks rather than all of them. The full "
        "val split is ~40k blocks (~42M tokens); evaluating all of it every eval_steps would cost "
        "~25min per eval and burn >1 day of the run on evals alone. 2k blocks (~2M tokens) is a "
        "plenty-tight loss estimate. Full val set is still on disk for a final end-of-run eval.",
    )
    ap.add_argument("--dataloader-num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    # Full-graph torch.compile beats ModernBERT's built-in `reference_compile` (which only compiles the
    # MLP) by +22.7% measured -- it fuses far more and cuts memory round-trips, which is the only thing
    # that matters on this bandwidth-bound box. So: reference_compile OFF, Trainer's torch_compile ON.
    ap.add_argument("--compile", action="store_true", default=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--sparse-prediction", action="store_true", default=True)
    ap.add_argument("--no-sparse-prediction", dest="sparse_prediction", action="store_false")
    # ModernBERT-base geometry (~150M). We are token-limited, not capacity-limited: at our budget a
    # 395M model would see only ~32 tokens/param, vs ~154 here -- far closer to a converged encoder.
    ap.add_argument("--hidden-size", type=int, default=768)
    ap.add_argument("--num-layers", type=int, default=22)
    ap.add_argument("--num-heads", type=int, default=12)
    ap.add_argument("--intermediate-size", type=int, default=1152)
    ap.add_argument("--push-to-hub", action="store_true")
    ap.add_argument("--hub-model-id", default="sakshamio/legal-financial-modernbert-150m")
    ap.add_argument(
        "--max-position-embeddings",
        type=int,
        default=16384,
        help="RoPE position buffer ceiling -- set higher than any block_size we'll actually train on, "
        "so the model can extrapolate somewhat beyond its trained context length at inference "
        "(this is ModernBERT's own documented approach, aided by its high global_rope_theta default).",
    )
    args = ap.parse_args()

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(TOKENIZER_DIR))

    meta = json.loads((PACKED_DIR / "meta.json").read_text())
    assert meta["vocab_size"] == len(tokenizer), (
        f"packed corpus was built with vocab {meta['vocab_size']} but the tokenizer now has "
        f"{len(tokenizer)} -- the token ids would be meaningless. Re-run pack_fast.py."
    )
    assert meta["block_size"] == args.block_size, f"packed at block {meta['block_size']}, asked for {args.block_size}"

    train_ds = PackedDataset(PACKED_DIR / "train.bin", args.block_size)
    val_full = PackedDataset(PACKED_DIR / "val.bin", args.block_size)
    print(f"train blocks: {len(train_ds):,} ({meta['train_tokens']/1e9:.2f}B tokens), val blocks: {len(val_full):,}")

    # Evaluating the whole val split every eval_steps would cost ~20min a time (~30h over the run).
    # A fixed subset is a plenty-tight loss estimate; the full split stays on disk for a final eval.
    val_ds = val_full
    if args.eval_subset_size and args.eval_subset_size < len(val_full):
        val_ds = torch.utils.data.Subset(val_full, range(args.eval_subset_size))
        print(f"using fixed {len(val_ds)}-block eval subset (full val set retained on disk)")

    if args.init_from:
        model = ModernBertForMaskedLM.from_pretrained(args.init_from)
        model.config.max_position_embeddings = max(model.config.max_position_embeddings, args.max_position_embeddings)
        print(f"initialized weights from {args.init_from} (context-extension phase)")
    else:
        model = build_model(
            tokenizer,
            max_position_embeddings=max(args.block_size, args.max_position_embeddings),
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            intermediate_size=args.intermediate_size,
            compile_=False,  # superseded by full-graph torch.compile below
            sparse=args.sparse_prediction,
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"random-init model: {n_params/1e6:.1f}M params, sparse_prediction={args.sparse_prediction}")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability)

    decay_steps = max(1, int(args.max_steps * args.decay_ratio))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="warmup_stable_decay",
        lr_scheduler_kwargs={"num_decay_steps": decay_steps},
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        optim="adamw_torch_fused",
        torch_compile=args.compile,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        report_to=[],
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
        # "every_save" pushes model weights + config + tokenizer; "checkpoint" would also push
        # optimizer state (~2x the bytes) on every save, which we don't need on the Hub -- local
        # checkpoints are what we resume from.
        hub_strategy="every_save",
        hub_private_repo=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        # Registering the tokenizer makes Trainer save AND push it with every checkpoint. Without
        # this, Hub checkpoints ship weights with no tokenizer -- and ours is a bespoke 50,368-vocab
        # domain BPE that exists nowhere else, so those checkpoints would be unusable by anyone.
        processing_class=tokenizer,
    )

    resume = any(Path(args.output_dir).glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=resume if resume else None)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
