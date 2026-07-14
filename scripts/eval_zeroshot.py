"""Zero-shot retrieval: the only metric that measures what we are ACTUALLY building.

Everything else we track -- MLM loss, effective rank, anisotropy, nesting -- is a PROXY for "will
this thing retrieve." Right now those proxies disagree with each other: MLM loss is excellent and
falling, while mean pairwise cosine has started RISING again (0.553 at step 6k -> 0.623 at step 10k)
and law/finance are growing MORE similar. Only a real retrieval measurement can adjudicate that.

TASK: LEDGAR (10,000 contract clauses, 100 clause types). Each clause is a query; a result is
relevant iff it has the SAME clause type. No training, no fine-tuning -- we mean-pool the raw
encoder, exactly as stage 2 will, and ask whether the representation is becoming useful.

Scored at every Matryoshka dim [768...64], because "does truncation hurt?" is the actual product
question, and variance-in-first-k-coords is only a proxy for it.

REFERENCE POINTS (a bare nDCG is not information):
  random ranking  -- chance, given the label distribution. The floor.
  our model       -- at each archived checkpoint.
  teacher         -- Qwen3-Embedding-0.6B, the ceiling we distil toward (--teacher, needs GPU).

CPU by default so it can run inside the research daemon alongside training.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import ModernBertModel, PreTrainedTokenizerFast

PROJECT_DIR = Path(__file__).resolve().parent.parent
TOKENIZER_DIR = PROJECT_DIR / "tokenizer"
CKPT_DIR = PROJECT_DIR / "checkpoints" / "mlm_stage1"
DIMS = [768, 512, 256, 128, 64]


def checkpoints():
    return sorted(CKPT_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))


def load_ledgar(n, seed=0):
    from datasets import load_dataset
    d = load_dataset("coastalcph/lex_glue", "ledgar", split="test")
    texts, labels = d["text"], d["label"]
    # keep only labels with >=2 examples -- a query whose type is unique has no retrievable positive
    from collections import Counter
    c = Counter(labels)
    keep = [i for i, l in enumerate(labels) if c[l] >= 2]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(keep)[:n]
    return [texts[i] for i in idx], np.array([labels[i] for i in idx])


@torch.no_grad()
def encode(enc, tok, texts, device, max_len=256, bs=32):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i : i + bs], padding=True, truncation=True, max_length=max_len,
                return_tensors="pt")
        b = {k: v.to(device) for k, v in b.items() if k in ("input_ids", "attention_mask")}
        h = enc(**b).last_hidden_state                       # [B, T, D]
        m = b["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).float().cpu())  # mean pooling, mask-aware
    return torch.cat(out)


def retrieval_metrics(E, labels, k=10):
    """P@k, nDCG@k, MRR. Query = each clause; relevant = same clause type; self excluded."""
    E = F.normalize(E, dim=-1)
    S = E @ E.T
    N = len(labels)
    S.fill_diagonal_(-2.0)                                   # never retrieve yourself

    lab = torch.from_numpy(labels)
    rel = (lab[:, None] == lab[None, :])                     # [N, N] boolean relevance
    rel.fill_diagonal_(False)
    n_rel = rel.sum(1).clamp(min=1)

    topk = S.topk(k, dim=1).indices                          # [N, k]
    hits = torch.gather(rel, 1, topk).float()                # [N, k]

    p_at_k = hits.mean(1).mean().item()

    disc = 1.0 / torch.log2(torch.arange(2, k + 2).float())  # [k]
    dcg = (hits * disc).sum(1)
    ideal_hits = torch.minimum(n_rel, torch.tensor(k)).float()
    idcg = torch.stack([disc[: int(i)].sum() for i in ideal_hits])
    ndcg = (dcg / idcg.clamp(min=1e-9)).mean().item()

    # MRR over the full ranking, not just top-k
    order = S.argsort(dim=1, descending=True)
    first = torch.gather(rel, 1, order).float().argmax(1) + 1
    mrr = (1.0 / first.float()).mean().item()

    # chance: expected P@k if we ranked at random
    chance = (n_rel.float() / (N - 1)).mean().item()
    return {"p_at_10": p_at_k, "ndcg_at_10": ndcg, "mrr": mrr, "chance_p_at_10": chance}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--teacher", action="store_true", help="score Qwen3-Embedding-0.6B (the ceiling)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    texts, labels = load_ledgar(args.n)
    print(f"LEDGAR: {len(texts):,} clauses, {len(set(labels.tolist()))} clause types\n")

    if args.teacher:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B",
                                model_kwargs={"dtype": torch.bfloat16}, device=args.device)
        E = torch.from_numpy(m.encode(texts, batch_size=32, show_progress_bar=False,
                                      normalize_embeddings=False))
        r = retrieval_metrics(E, labels)
        print(f"TEACHER Qwen3-Embedding-0.6B  P@10 {r['p_at_10']:.3f}  "
              f"nDCG@10 {r['ndcg_at_10']:.3f}  MRR {r['mrr']:.3f}")
        return

    ckpt = Path(args.checkpoint) if args.checkpoint else checkpoints()[-1]
    step = int(ckpt.name.split("-")[1])
    out_path = PROJECT_DIR / f"retrieval_step{step}.json"
    if out_path.exists() and not args.force:
        print(f"{out_path.name} exists; skipping")
        return

    tok = PreTrainedTokenizerFast.from_pretrained(str(TOKENIZER_DIR))
    enc = ModernBertModel.from_pretrained(str(ckpt)).to(args.device).eval()
    E = encode(enc, tok, texts, args.device)

    rows = {}
    print(f"checkpoint {ckpt.name} (step {step:,})")
    print(f"  {'dim':>5} {'P@10':>7} {'nDCG@10':>8} {'MRR':>7}   vs chance")
    chance = None
    for d in DIMS:
        r = retrieval_metrics(E[:, :d], labels)
        rows[d] = r
        chance = r["chance_p_at_10"]
        print(f"  {d:>5} {r['p_at_10']:>7.3f} {r['ndcg_at_10']:>8.3f} {r['mrr']:>7.3f}   "
              f"{r['p_at_10']/chance:>5.1f}x")
    print(f"\n  chance P@10 = {chance:.3f}   (random ranking, given the label distribution)")
    print(f"  retention at 64 dims: {100*rows[64]['ndcg_at_10']/rows[768]['ndcg_at_10']:.0f}% of nDCG@768")

    out_path.write_text(json.dumps({"step": step, "chance": chance,
                                    "by_dim": {str(d): rows[d] for d in DIMS}}, indent=2))
    print(f"\nsaved -> {out_path.name}")


if __name__ == "__main__":
    main()
