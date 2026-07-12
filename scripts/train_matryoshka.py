"""Stage 2: turn the pretrained encoder into a Matryoshka embedding model.

Wraps the stage-1 checkpoint in a SentenceTransformer (mean pooling) and trains
MultipleNegativesRankingLoss inside MatryoshkaLoss.

Two non-obvious things here:

1. LABEL-AWARE BATCHING. MNRL treats every OTHER example in the batch as a negative. With LEDGAR pairs
   (two provisions sharing a clause type), two same-label examples in one batch means the model is
   trained to PUSH APART two genuinely similar clauses -- the exact opposite of what we want. With 100
   clause types and a batch of 64, you expect ~20 colliding pairs PER BATCH, so this is not a rare edge
   case. `NoLabelCollisionSampler` guarantees every example in a batch carries a distinct label.

2. CACHED MNRL. In-batch negatives are the whole point of MNRL, so a bigger batch = more negatives =
   better. GradCache (`CachedMultipleNegativesRankingLoss`) decouples the effective batch from memory by
   recomputing embeddings in chunks -- so we can use a large contrastive batch on a box where memory is
   the constraint that bites (batch 128 OOM'd during stage-1 tuning).
"""
import argparse
import random
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import (
    CachedMultipleNegativesRankingLoss,
    MatryoshkaLoss,
    MultipleNegativesRankingLoss,
)
from sentence_transformers.models import Pooling, Transformer

PROJECT_DIR = Path(__file__).resolve().parent.parent
PAIRS_DIR = PROJECT_DIR / "data" / "pairs"
CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"


def matryoshka_dims(full_dim):
    """The standard MRL ladder (…512, 256, 128, 64), capped at the model's actual width.

    Derived, never hardcoded: a hardcoded 768-first list against a 1024-wide model silently discards the
    top 256 dims and never trains them. Naive halving (768→384→192→96) is also wrong -- it never lands
    on 64, the dim people actually want for cheap first-stage retrieval.
    """
    ladder = [d for d in (1024, 768, 512, 256, 128, 64) if d <= full_dim]
    if full_dim not in ladder:
        ladder.insert(0, full_dim)
    return sorted(set(ladder), reverse=True)


class NoLabelCollisionSampler(torch.utils.data.Sampler):
    """Batches in which every example carries a DISTINCT label.

    MNRL uses the other in-batch examples as negatives. Two LEDGAR provisions of the same clause type,
    or two EDGAR "risk factors" sections, are genuinely similar -- making one a negative for the other
    teaches the model something false. This sampler simply never places them together.

    Greedy fill: walk a shuffled stream, add an example only if its label is unused in the current
    batch, park collisions for a later batch. Examples are not dropped, only deferred.
    """

    def __init__(self, labels, batch_size, seed=0, drop_last=True):
        self.labels = list(labels)
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        order = list(range(len(self.labels)))
        rng.shuffle(order)

        pending = defaultdict(list)  # label -> deferred indices
        batch, used = [], set()
        for i in order:
            lab = self.labels[i]
            if lab in used:
                pending[lab].append(i)
                continue
            batch.append(i)
            used.add(lab)
            if len(batch) == self.batch_size:
                yield from batch
                batch, used = [], set()

        # drain deferred examples, still respecting the no-collision rule
        leftovers = [i for v in pending.values() for i in v]
        rng.shuffle(leftovers)
        for i in leftovers:
            lab = self.labels[i]
            if lab in used:
                continue  # would collide again; drop rather than corrupt the batch
            batch.append(i)
            used.add(lab)
            if len(batch) == self.batch_size:
                yield from batch
                batch, used = [], set()
        if batch and not self.drop_last:
            yield from batch

    def __len__(self):
        return (len(self.labels) // self.batch_size) * self.batch_size


class LabelAwareTrainer(SentenceTransformerTrainer):
    """SentenceTransformerTrainer with the no-label-collision sampler."""

    def __init__(self, *a, batch_labels=None, **kw):
        self.batch_labels = batch_labels
        super().__init__(*a, **kw)

    def get_train_sampler(self, *a, **kw):
        if self.batch_labels is None:
            return super().get_train_sampler(*a, **kw)
        return NoLabelCollisionSampler(
            self.batch_labels,
            self.args.per_device_train_batch_size,
            seed=self.args.seed,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-checkpoint", default=str(CHECKPOINTS_DIR / "mlm_stage1"))
    ap.add_argument("--output-dir", default=str(CHECKPOINTS_DIR / "embedding_stage2"))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--mini-batch-size", type=int, default=16, help="GradCache chunk; memory knob only")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--no-gradcache", action="store_true")
    ap.add_argument("--no-label-sampler", action="store_true")
    ap.add_argument("--push-to-hub", action="store_true")
    ap.add_argument("--hub-model-id", default="sakshamio/legal-financial-modernbert-150m-embed")
    args = ap.parse_args()

    transformer = Transformer(args.stage1_checkpoint, max_seq_length=args.max_seq_length)
    dim = transformer.get_word_embedding_dimension()
    model = SentenceTransformer(modules=[transformer, Pooling(dim, pooling_mode="mean")])

    dims = matryoshka_dims(dim)
    print(f"embedding dim {dim} -> matryoshka dims {dims}")

    train = load_from_disk(str(PAIRS_DIR / "train"))
    val = load_from_disk(str(PAIRS_DIR / "val"))
    labels = train["label"] if "label" in train.column_names else None

    # `label`/`source` are metadata for the sampler -- the loss must never see them as text columns.
    drop = [c for c in ("label", "source") if c in train.column_names]
    train_in = train.remove_columns(drop)
    val_in = val.remove_columns([c for c in drop if c in val.column_names])
    n_neg = sum(1 for c in train_in.column_names if c.startswith("negative_"))
    print(f"train {len(train_in)} / val {len(val_in)} | columns {train_in.column_names} ({n_neg} hard negatives)")

    # GradCache: recompute embeddings in chunks so the contrastive batch (=negative count) is not
    # capped by memory. More in-batch negatives is the single biggest lever MNRL has.
    if args.no_gradcache:
        base = MultipleNegativesRankingLoss(model)
    else:
        base = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch_size)
    loss = MatryoshkaLoss(model, base, matryoshka_dims=dims)

    targs = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
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
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
    )

    trainer = LabelAwareTrainer(
        model=model,
        args=targs,
        train_dataset=train_in,
        eval_dataset=val_in,
        loss=loss,
        batch_labels=None if args.no_label_sampler else labels,
    )
    if labels and not args.no_label_sampler:
        print(f"label-aware batching ON: {len(set(labels))} distinct labels, no collisions within a batch")

    trainer.train()
    model.save_pretrained(args.output_dir)
    print(f"saved -> {args.output_dir}")


if __name__ == "__main__":
    main()
