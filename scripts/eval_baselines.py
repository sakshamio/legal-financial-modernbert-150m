"""Run OTHER models through the exact same LEDGAR retrieval harness.

There are no published numbers for this task. LEDGAR in LexGLUE is a 100-class CLASSIFICATION
benchmark scored by F1; the retrieval framing here (same clause type = relevant, nDCG@10) is one we
constructed. So "how do other models do" can only be answered by running them, not by quoting.

The baselines are chosen to bracket our result from every side:

  random init      -- our exact architecture, untrained. The floor. Anything above this is learning.
  BM25             -- lexical, no neural anything. In LEGAL retrieval BM25 is notoriously strong and
                      routinely beats neural models. If we cannot clear BM25, the model is pointless.
  legal-bert-base  -- THE apples-to-apples comparison: a legal-domain MLM encoder, mean-pooled,
                      exactly our recipe. Same setup, different pretraining. 110M params.
  bge-base-en-v1.5 -- a fully-trained general embedder at OUR parameter count (109M).
  all-MiniLM-L6-v2 -- the ubiquitous small baseline (22M).
  Qwen3-Emb-0.6B   -- our distillation teacher, and the ceiling we were aiming at.

Ours at step 12,000 (4.4% trained, no retrieval training): nDCG@10 = 0.546
"""
import argparse

import numpy as np
import torch

from eval_zeroshot import load_ledgar, retrieval_metrics

ST_MODELS = [
    ("all-MiniLM-L6-v2 (22M)", "sentence-transformers/all-MiniLM-L6-v2"),
    ("bge-small-en-v1.5 (33M)", "BAAI/bge-small-en-v1.5"),
    ("bge-base-en-v1.5 (109M)", "BAAI/bge-base-en-v1.5"),
    ("Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-0.6B"),
]
# Raw MLM encoders, mean-pooled -- the same recipe we use, so these are the fair comparison.
MLM_MODELS = [
    ("legal-bert-base (110M) *", "nlpaueb/legal-bert-base-uncased"),
    ("bert-base-uncased (110M) *", "bert-base-uncased"),
]


def bm25_metrics(texts, labels, k=10):
    from rank_bm25 import BM25Okapi
    corpus = [t.lower().split() for t in texts]
    bm = BM25Okapi(corpus)
    N = len(texts)
    lab = torch.from_numpy(labels)
    rel = (lab[:, None] == lab[None, :])
    rel.fill_diagonal_(False)
    n_rel = rel.sum(1).clamp(min=1)

    S = torch.zeros(N, N)
    for i, q in enumerate(corpus):
        S[i] = torch.from_numpy(bm.get_scores(q)).float()
    S.fill_diagonal_(-1e9)

    topk = S.topk(k, 1).indices
    hits = torch.gather(rel, 1, topk).float()
    disc = 1.0 / torch.log2(torch.arange(2, k + 2).float())
    dcg = (hits * disc).sum(1)
    ideal = torch.minimum(n_rel, torch.tensor(k)).float()
    idcg = torch.stack([disc[: int(i)].sum() for i in ideal])
    order = S.argsort(1, descending=True)
    first = torch.gather(rel, 1, order).float().argmax(1) + 1
    return {"p_at_10": hits.mean(1).mean().item(),
            "ndcg_at_10": (dcg / idcg.clamp(min=1e-9)).mean().item(),
            "mrr": (1.0 / first.float()).mean().item()}


@torch.no_grad()
def meanpool_encode(name, texts, device, bs=32, max_len=256):
    from transformers import AutoModel, AutoTokenizer
    tk = AutoTokenizer.from_pretrained(name)
    md = AutoModel.from_pretrained(name).to(device).eval()
    out = []
    for i in range(0, len(texts), bs):
        b = tk(texts[i : i + bs], padding=True, truncation=True, max_length=max_len,
               return_tensors="pt")
        b = {k: v.to(device) for k, v in b.items()}
        h = md(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).float().cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    texts, labels = load_ledgar(args.n)
    print(f"LEDGAR retrieval: {len(texts):,} clauses, {len(set(labels.tolist()))} clause types")
    print("(same clause type = relevant; nDCG@10. NOT a published benchmark -- all numbers measured here.)\n")
    rows = []

    # floor: our architecture, untrained
    from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast
    from eval_zeroshot import CKPT_DIR, TOKENIZER_DIR, checkpoints
    cfg = ModernBertConfig.from_pretrained(str(checkpoints()[-1]))
    cfg.reference_compile = False
    tok = PreTrainedTokenizerFast.from_pretrained(str(TOKENIZER_DIR))
    torch.manual_seed(0)
    rnd = ModernBertModel(cfg).to(args.device).eval()
    from eval_zeroshot import encode
    E = encode(rnd, tok, texts, args.device)
    r = retrieval_metrics(E, labels)
    rows.append(("random init (ours, untrained)", r["ndcg_at_10"], r["p_at_10"]))
    print(f"  {'random init (ours, untrained)':<32} nDCG@10 {r['ndcg_at_10']:.3f}")
    del rnd

    print(f"  {'BM25 (lexical, no neural)':<32} ", end="", flush=True)
    r = bm25_metrics(texts, labels)
    rows.append(("BM25 (lexical)", r["ndcg_at_10"], r["p_at_10"]))
    print(f"nDCG@10 {r['ndcg_at_10']:.3f}")

    for label, name in MLM_MODELS:
        try:
            E = meanpool_encode(name, texts, args.device)
            r = retrieval_metrics(E, labels)
            rows.append((label, r["ndcg_at_10"], r["p_at_10"]))
            print(f"  {label:<32} nDCG@10 {r['ndcg_at_10']:.3f}")
        except Exception as e:
            print(f"  {label:<32} FAILED: {type(e).__name__}")

    from sentence_transformers import SentenceTransformer
    for label, name in ST_MODELS:
        try:
            m = SentenceTransformer(name, device=args.device)
            E = torch.from_numpy(m.encode(texts, batch_size=32, show_progress_bar=False))
            r = retrieval_metrics(E, labels)
            rows.append((label, r["ndcg_at_10"], r["p_at_10"]))
            print(f"  {label:<32} nDCG@10 {r['ndcg_at_10']:.3f}")
            del m
        except Exception as e:
            print(f"  {label:<32} FAILED: {type(e).__name__}")

    print("\n" + "=" * 62)
    print(f"  {'MODEL':<32} {'nDCG@10':>8}")
    print("=" * 62)
    for label, nd, _ in sorted(rows, key=lambda x: -x[1]):
        print(f"  {label:<32} {nd:>8.3f}")
    print(f"  {'>>> OURS @ step 12,000 (4.4%)':<32} {0.546:>8.3f}  <- no retrieval training")
    print("\n  * = raw MLM encoder, mean-pooled: the same recipe as ours (apples-to-apples)")


if __name__ == "__main__":
    main()
