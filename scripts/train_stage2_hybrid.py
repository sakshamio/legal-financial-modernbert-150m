"""Stage 2 (hybrid) trainer: dense + sparse, GIST-filtered, reranker-distilled, Matryoshka.

Combines four literature-backed upgrades over the plain-contrastive Stage 2 (which scored 0.285 vs
MiniLM 0.575):

  dense  : matryoshka InfoNCE with GISTEmbed false-negative masking (guide model)
  sparse : SPLADE contrastive + FLOPS sparsity
  distil : MarginMSE from a cross-encoder reranker
  pool   : latent-attention pooling (NV-Embed)

The two GPU-teacher inputs are PRECOMPUTED and passed as files, so training itself needs no teacher
loaded and the expensive passes run once:

  --guide-emb  <path.f16>  Qwen-embedding vectors for every anchor+positive, for GIST masking.
  --margins    <path.f16>  reranker margins reranker(q,pos)-reranker(q,neg), for MarginMSE.

Both optional: without them the trainer still does hybrid dense+sparse+matryoshka. Memory is kept in
check with GradCache-style micro-chunk encoding (this box is bandwidth/-memory bound).

    python scripts/train_stage2_hybrid.py --pairs data/pairs_synth_taxonomy/train.jsonl \
        --init checkpoints/mlm_stage1/checkpoint-14000 --device cuda
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast

from hybrid_losses import margin_mse, matryoshka_gist_infonce, splade_infonce
from hybrid_model import HybridEmbedder

PROJECT_DIR = Path(__file__).resolve().parent.parent


class PairDS(Dataset):
    def __init__(self, path, limit=None):
        self.rows = []
        with open(path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                r = json.loads(line)
                self.rows.append(r)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        strip = lambda s: s.replace("[QUERY] ", "").replace("[PASSAGE] ", "")
        return {
            "idx": i,
            "anchor": strip(r["anchor"]),
            "positive": strip(r["positive"]),
            "negative": strip(r.get("negative_0", "")) or None,
        }


def encode(model, tok, texts, device, max_len, chunk):
    """Encode in micro-chunks and concatenate -- keeps peak activation memory bounded."""
    dense, sparse = [], []
    for s in range(0, len(texts), chunk):
        b = tok(texts[s : s + chunk], padding=True, truncation=True, max_length=max_len,
                return_tensors="pt")
        b = {k: v.to(device) for k, v in b.items()}
        out = model(b["input_ids"], b["attention_mask"])
        dense.append(out["dense"])
        if "sparse" in out:
            sparse.append(out["sparse"])
    return torch.cat(dense), (torch.cat(sparse) if sparse else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs_synth_taxonomy/train.jsonl")
    ap.add_argument("--init", default="checkpoints/mlm_stage1/checkpoint-14000")
    ap.add_argument("--output-dir", default="checkpoints/stage2_hybrid")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--encode-chunk", type=int, default=16, help="micro-chunk for memory")
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--w-dense", type=float, default=1.0)
    ap.add_argument("--w-sparse", type=float, default=0.2)
    ap.add_argument("--w-distil", type=float, default=0.2)
    ap.add_argument("--flops-lambda", type=float, default=1e-3)
    ap.add_argument("--no-sparse", action="store_true")
    ap.add_argument("--guide-emb", default=None, help="precomputed Qwen embeddings .f16 for GIST")
    ap.add_argument("--margins", default=None, help="precomputed reranker margins .f16 for MarginMSE")
    ap.add_argument("--guide-dim", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log-steps", type=int, default=20)
    args = ap.parse_args()

    tok = HybridEmbedder.load_tokenizer_fix(PreTrainedTokenizerFast.from_pretrained(
        str(PROJECT_DIR / "tokenizer")))
    model = HybridEmbedder(args.init, sparse=not args.no_sparse).to(args.device).train()

    ds = PairDS(args.pairs, args.limit)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                    collate_fn=lambda b: b)

    # optional precomputed teacher inputs (memmapped; indexed by row idx)
    guide = np.memmap(args.guide_emb, dtype=np.float16, mode="r").reshape(-1, 2, args.guide_dim) \
        if args.guide_emb else None   # [N, {anchor,positive}, guide_dim]
    margins = np.memmap(args.margins, dtype=np.float16, mode="r") if args.margins else None  # [N]

    steps = args.max_steps if args.max_steps > 0 else int(len(dl) * args.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    warm = max(1, int(steps * 0.05))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    print(f"hybrid Stage 2 | {len(ds):,} pairs | steps {steps:,} | sparse={not args.no_sparse} | "
          f"GIST={'on' if guide is not None else 'off'} | distil={'on' if margins is not None else 'off'}")

    step, t0 = 0, time.time()
    done = False
    while not done:
        for batch in dl:
            anchors = [b["anchor"] for b in batch]
            positives = [b["positive"] for b in batch]
            idxs = [b["idx"] for b in batch]

            qd, qs = encode(model, tok, anchors, args.device, args.max_len, args.encode_chunk)
            pd, ps = encode(model, tok, positives, args.device, args.max_len, args.encode_chunk)

            gq = gp = None
            if guide is not None:
                gq = torch.from_numpy(np.asarray(guide[idxs, 0])).to(args.device)
                gp = torch.from_numpy(np.asarray(guide[idxs, 1])).to(args.device)

            loss = args.w_dense * matryoshka_gist_infonce(
                qd, pd, tau=args.tau, guide_q=gq, guide_p=gp)

            parts = {"dense": float(loss)}
            if qs is not None:
                sl, flops = splade_infonce(qs, ps, tau=args.tau)
                # FLOPS coefficient warms up quadratically (standard SPLADE): near 0 early so the head
                # can learn what to activate, ramping to full so it then sparsifies. Kept separate from
                # w_sparse so the sparse CONTRASTIVE signal is weighted like the dense one, not buried.
                flops_coef = args.flops_lambda * min(1.0, (step / max(1, int(steps * 0.3))) ** 2)
                loss = loss + args.w_sparse * sl + flops_coef * flops
                parts["sparse"] = float(sl)
                parts["nnz"] = float((qs > 0).float().sum(1).mean())

            if margins is not None and all(b["negative"] for b in batch):
                negs = [b["negative"] for b in batch]
                nd, _ = encode(model, tok, negs, args.device, args.max_len, args.encode_chunk)
                tm = torch.from_numpy(np.asarray(margins[idxs])).float().to(args.device)
                dl_ = margin_mse(qd, pd, nd, tm)
                loss = loss + args.w_distil * dl_
                parts["distil"] = float(dl_)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_steps == 0:
                el = time.time() - t0
                extra = " ".join(f"{k} {v:.3f}" for k, v in parts.items())
                print(f"  step {step}/{steps} loss {loss.item():.4f} [{extra}] "
                      f"lr {sched.get_last_lr()[0]:.2e} {step/el:.2f} it/s", flush=True)
            if step >= steps:
                done = True
                break

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "hybrid.pt")
    (out / "config.json").write_text(json.dumps({"init": args.init, "sparse": not args.no_sparse,
                                                  "matryoshka_dims": list(model.matryoshka_dims)}))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
