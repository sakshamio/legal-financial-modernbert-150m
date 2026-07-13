"""Make the MLM loss interpretable.

A cross-entropy of "4.62" means nothing on its own. What makes it meaningful is where it sits between
the reference points:

  ln(vocab) = 10.83   a uniform guess over all 50,368 tokens -- what random init produces
  unigram H = ?       what you get by predicting TOKEN FREQUENCIES and nothing else. This is the real
                      bar: beating it is the first proof the model uses CONTEXT rather than just
                      learning that "the" is common. Computed here from the actual packed corpus.
  ~1.5-2.0            roughly where well-trained BERT-class encoders land on in-domain text.

We also report:
  - perplexity: exp(loss) = the effective number of tokens the model is choosing between.
  - top-1 / top-5 accuracy on masked positions -- far more legible than a loss value.
  - PER-DOMAIN loss (legal / financial / general), which tells us WHERE it is learning. The packed
    corpus is laid out in source order, so we can slice regions of the memmap by domain.
  - Qualitative mask-fills on real domain text -- the only thing that shows what it actually knows.

Runs on CPU by default: on this unified-memory box, CPU work steals ~2% from a bandwidth-bound GPU
training run, and a GPU eval would contend far worse. A few hundred forward passes on CPU is cheap.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import DataCollatorForLanguageModeling, ModernBertForMaskedLM, PreTrainedTokenizerFast

PROJECT_DIR = Path(__file__).resolve().parent.parent
TOKENIZER_DIR = PROJECT_DIR / "tokenizer"
PACKED_DIR = PROJECT_DIR / "data" / "packed"
CKPT_DIR = PROJECT_DIR / "checkpoints" / "mlm_stage1"

# The packed corpus is written in alphabetical file order, so each domain occupies a contiguous region
# of the token stream. Fractions of train.bin, measured from the deduped source sizes.
DOMAIN_REGIONS = {
    "financial_edgar": (0.00, 0.24),
    "financial_pol": (0.24, 0.31),
    "general_web": (0.31, 0.55),
    "legal_caselaw": (0.55, 0.84),
    "legal_contracts": (0.84, 0.96),
    "legal_regulations": (0.96, 1.00),
}

PROBES = [
    ("legal",     "The [MASK] shall indemnify and hold harmless the Company from any claims."),
    ("legal",     "Notwithstanding the [MASK], this Agreement shall remain in full force and effect."),
    ("citation",  "See Roe v. [MASK], 410 U.S. 113 (1973)."),
    ("citation",  "Pursuant to 15 [MASK] § 78j(b), the defendant is liable."),
    ("financial", "Item 1A. Risk [MASK]. Our business is subject to numerous risks."),
    ("financial", "Revenue increased 12% to $3.4 [MASK] for the fiscal year ended December 31."),
    ("financial", "The Company filed its annual report on Form [MASK] with the SEC."),
    ("general",   "The capital of France is [MASK]."),
    ("general",   "Water boils at 100 degrees [MASK]."),
]


def latest_checkpoint():
    cks = sorted(CKPT_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if not cks:
        raise SystemExit(f"no checkpoint in {CKPT_DIR}")
    return cks[-1]


def unigram_entropy(path, tok, sample_tokens=200_000_000):
    """Entropy of the token distribution = the loss you'd get by predicting frequencies alone.

    This is THE reference point. A model below it is using context; a model at it has only learned
    which tokens are common. Sampled across the whole stream so every domain is represented.
    """
    m = np.memmap(path, dtype=np.uint16, mode="r")
    stride = max(1, len(m) // sample_tokens)
    sample = np.asarray(m[::stride][:sample_tokens])
    counts = np.bincount(sample, minlength=len(tok)).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    H = float(-(p * np.log(p)).sum())
    return H, len(sample)


@torch.no_grad()
def eval_region(model, collator, m, lo, hi, block, n_blocks, seed=0):
    """Masked-LM loss + top-k accuracy over blocks drawn from one region of the token stream."""
    rng = np.random.default_rng(seed)
    n_total = (hi - lo) // block
    if n_total < 1:
        return None
    idx = rng.choice(n_total, size=min(n_blocks, n_total), replace=False)

    tot_loss = tot_tok = top1 = top5 = 0
    for i in idx:
        s = lo + int(i) * block
        ids = torch.from_numpy(np.asarray(m[s : s + block]).astype(np.int64))
        batch = collator([{"input_ids": ids}])
        out = model(input_ids=batch["input_ids"], labels=batch["labels"])

        labels = batch["labels"][0]
        mask = labels != -100
        if mask.sum() == 0:
            continue
        # sparse_prediction returns logits ONLY for masked positions, already gathered
        logits = out.logits
        gold = labels[mask]
        if logits.shape[0] != gold.shape[0]:
            logits = logits[0][mask]
        tot_loss += out.loss.item() * gold.numel()
        tot_tok += gold.numel()
        top = logits.topk(5, dim=-1).indices
        top1 += (top[:, 0] == gold).sum().item()
        top5 += (top == gold[:, None]).any(-1).sum().item()

    return {
        "loss": tot_loss / tot_tok,
        "ppl": math.exp(tot_loss / tot_tok),
        "top1": top1 / tot_tok,
        "top5": top5 / tot_tok,
        "tokens": tot_tok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cpu", help="cpu keeps the GPU free for training")
    ap.add_argument("--blocks-per-domain", type=int, default=24)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--skip-unigram", action="store_true")
    args = ap.parse_args()

    tok = PreTrainedTokenizerFast.from_pretrained(str(TOKENIZER_DIR))
    ckpt = Path(args.checkpoint) if args.checkpoint else latest_checkpoint()
    step = int(ckpt.name.split("-")[1])
    V = len(tok)

    print(f"checkpoint: {ckpt.name}  (step {step:,})\n")

    # ---- reference points -------------------------------------------------------------------------
    uniform = math.log(V)
    print("=" * 74)
    print("REFERENCE POINTS -- what the loss number actually means")
    print("=" * 74)
    print(f"  uniform over vocab   ln({V:,}) = {uniform:.2f}   (a random-init model; perplexity {V:,})")
    if not args.skip_unigram:
        H, n = unigram_entropy(PACKED_DIR / "train.bin", tok)
        print(f"  unigram entropy      {H:.2f}   <- THE BAR: predicting token frequencies alone")
        print(f"                              (perplexity {math.exp(H):,.0f}; from {n/1e6:.0f}M sampled tokens)")
        print(f"                              Below this = the model is USING CONTEXT, not just frequency.")
    print(f"  well-trained encoder ~1.5-2.0 on in-domain text")

    # ---- current model ----------------------------------------------------------------------------
    model = ModernBertForMaskedLM.from_pretrained(str(ckpt)).to(args.device).eval()
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=True, mlm_probability=0.3)
    m = np.memmap(PACKED_DIR / "train.bin", dtype=np.uint16, mode="r")
    total = len(m)

    print("\n" + "=" * 74)
    print("PER-DOMAIN -- where is it learning?")
    print("=" * 74)
    print(f"  {'domain':<20} {'loss':>7} {'ppl':>9} {'top-1':>8} {'top-5':>8}")
    rows = {}
    for name, (a, b) in DOMAIN_REGIONS.items():
        r = eval_region(model, collator, m, int(a * total), int(b * total), args.block_size,
                        args.blocks_per_domain)
        if r:
            rows[name] = r
            print(f"  {name:<20} {r['loss']:>7.3f} {r['ppl']:>9.1f} {r['top1']:>7.1%} {r['top5']:>7.1%}")
    if rows:
        avg = sum(r["loss"] for r in rows.values()) / len(rows)
        print(f"\n  overall loss {avg:.3f} -> perplexity {math.exp(avg):.1f}")
        print(f"  i.e. the model is effectively choosing between ~{math.exp(avg):.0f} tokens per masked")
        print(f"  position, down from {V:,} at random init.")

    # ---- qualitative ------------------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("WHAT IT ACTUALLY PREDICTS (top-5 for each [MASK])")
    print("=" * 74)
    for kind, text in PROBES:
        ids = tok(text, return_tensors="pt")
        ids = {k: v for k, v in ids.items() if k in ("input_ids", "attention_mask")}  # no token_type_ids
        pos = (ids["input_ids"][0] == tok.mask_token_id).nonzero()
        if len(pos) == 0:
            continue
        with torch.no_grad():
            logits = model(**{k: v.to(args.device) for k, v in ids.items()}).logits
        # with sparse_prediction and no labels, logits are dense [B, T, V]
        lg = logits[0][pos[0, 0]] if logits.dim() == 3 else logits[pos[0, 0]]
        top = lg.topk(5).indices.tolist()
        preds = [tok.decode([t]).strip() or "␣" for t in top]
        print(f"  [{kind:<9}] {text}")
        print(f"              -> {preds}")

    out = {"step": step, "uniform": uniform, "per_domain": rows}
    (PROJECT_DIR / "loss_analysis.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
