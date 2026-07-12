"""Stage 2: wrap the stage-1 pretrained encoder into a SentenceTransformer and
fine-tune it for embeddings with MultipleNegativesRankingLoss wrapped in MatryoshkaLoss.
"""
import argparse
from pathlib import Path

from datasets import load_from_disk
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.models import Pooling, Transformer

PROJECT_DIR = Path(__file__).resolve().parent.parent
PAIRS_DIR = PROJECT_DIR / "data" / "pairs"
CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"

def matryoshka_dims(full_dim):
    """Nested dims for MRL, largest first: the standard MRL ladder (…512, 256, 128, 64), capped at the
    model's actual width, with the full width always included.

    Derived rather than hardcoded -- a hardcoded 768-first list against a 1024-wide model silently
    discards the top 256 dims and never trains them. Naive halving (768→384→192→96) is also wrong: it
    never lands on 64, the dim people actually want for cheap first-stage retrieval.
    """
    ladder = [d for d in (1024, 768, 512, 256, 128, 64) if d <= full_dim]
    if full_dim not in ladder:
        ladder.insert(0, full_dim)
    return sorted(set(ladder), reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-checkpoint", default=str(CHECKPOINTS_DIR / "mlm_stage1"))
    ap.add_argument("--output-dir", default=str(CHECKPOINTS_DIR / "embedding_stage2"))
    ap.add_argument("--per-device-batch-size", type=int, default=64)
    ap.add_argument("--grad-accum-steps", type=int, default=4)
    ap.add_argument("--num-train-epochs", type=float, default=3.0)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    args = ap.parse_args()

    transformer = Transformer(args.stage1_checkpoint)
    embedding_dim = transformer.get_word_embedding_dimension()
    pooling = Pooling(embedding_dim, pooling_mode="mean")
    model = SentenceTransformer(modules=[transformer, pooling])

    dims = matryoshka_dims(embedding_dim)
    print(f"embedding dim: {embedding_dim}, matryoshka dims: {dims}")

    train_ds = load_from_disk(str(PAIRS_DIR / "train"))
    val_ds = load_from_disk(str(PAIRS_DIR / "val"))
    print(f"train pairs: {len(train_ds)}, val pairs: {len(val_ds)}")

    base_loss = MultipleNegativesRankingLoss(model)
    loss = MatryoshkaLoss(model, base_loss, matryoshka_dims=dims)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        logging_steps=20,
        report_to=[],
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        loss=loss,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    print(f"saved embedding model to {args.output_dir}")


if __name__ == "__main__":
    main()
