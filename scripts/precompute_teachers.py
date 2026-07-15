"""Precompute the two GPU-teacher inputs for the hybrid Stage-2 trainer, once.

Decoupling these from training means the teachers are never loaded during the training loop (which
keeps that loop small and fast) and the expensive passes run exactly once per pair set.

  --mode guide    Qwen3-Embedding-0.6B encodes every anchor and positive -> guide_emb.f16 [N, 2, 1024]
                  Used by GISTEmbed to detect in-batch false negatives.

  --mode margins  A cross-encoder reranker scores (query, positive) and (query, negative); we store the
                  margin s(q,pos) - s(q,neg) -> margins.f16 [N]. Used by MarginMSE distillation.
                  Qwen3-Reranker emits a relevance judgement as the logit of the "yes" token; the score
                  is sigmoid(logit_yes - logit_no).

Both are GPU jobs and must not share the GPU with another (generation/training). Run when the box is
free.

    python scripts/precompute_teachers.py --mode guide --pairs data/pairs_synth_taxonomy/train.jsonl
    python scripts/precompute_teachers.py --mode margins --pairs data/pairs_synth_taxonomy/train.jsonl
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
EMB_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"   # small reranker; upsize if a stronger one is worth the time


def guard_gpu(force):
    busy = subprocess.run("pgrep -f '[g]ogen/gogen'", shell=True, capture_output=True).stdout
    alive = subprocess.run(["scripts/is_training_alive.sh"], cwd=PROJECT_DIR).returncode == 0
    if (busy or alive) and not force:
        raise SystemExit("GPU busy (generation/training). These are GPU passes -- wait or pass --force.")


def load_pairs(path, limit):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            r = json.loads(line)
            strip = lambda s: s.replace("[QUERY] ", "").replace("[PASSAGE] ", "")
            rows.append((strip(r["anchor"]), strip(r["positive"]),
                         strip(r.get("negative_0", "")) or None))
    return rows


def do_guide(rows, out, device, bs):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMB_MODEL, model_kwargs={"dtype": torch.bfloat16}, device=device)
    mm = np.memmap(out, dtype=np.float16, mode="w+", shape=(len(rows), 2, 1024))
    for s in range(0, len(rows), bs):
        chunk = rows[s : s + bs]
        a = m.encode([r[0] for r in chunk], normalize_embeddings=True, convert_to_numpy=True,
                     show_progress_bar=False)
        p = m.encode([r[1] for r in chunk], normalize_embeddings=True, convert_to_numpy=True,
                     show_progress_bar=False)
        mm[s : s + len(chunk), 0] = a.astype(np.float16)
        mm[s : s + len(chunk), 1] = p.astype(np.float16)
        if s % (bs * 50) == 0:
            print(f"  guide {s:,}/{len(rows):,}", flush=True)
    mm.flush()
    print(f"guide embeddings -> {out}")


@torch.no_grad()
def do_margins(rows, out, device, bs):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(RERANK_MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(RERANK_MODEL, dtype=torch.bfloat16,
                                                 device_map=device).eval()
    yes_id = tok.convert_tokens_to_ids("yes")
    no_id = tok.convert_tokens_to_ids("no")

    def score(queries, docs):
        # Qwen3-Reranker prompt: judge whether the document answers the query; read the yes/no logit.
        prompts = [f"<Instruct>: Judge whether the Document answers the Query.\n<Query>: {q}\n"
                   f"<Document>: {d[:1200]}\n<Judgement>:" for q, d in zip(queries, docs)]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        logits = model(**enc).logits[:, -1, :]
        return torch.sigmoid(logits[:, yes_id] - logits[:, no_id])

    have_neg = [r for r in rows if r[2]]
    margins = np.memmap(out, dtype=np.float16, mode="w+", shape=(len(rows),))
    for s in range(0, len(rows), bs):
        chunk = rows[s : s + bs]
        q = [r[0] for r in chunk]
        s_pos = score(q, [r[1] for r in chunk])
        # rows without a negative get margin 0 (MarginMSE skips them at train time via the batch guard)
        negs = [r[2] if r[2] else r[1] for r in chunk]
        s_neg = score(q, negs)
        margins[s : s + len(chunk)] = (s_pos - s_neg).float().cpu().numpy().astype(np.float16)
        if s % (bs * 50) == 0:
            print(f"  margins {s:,}/{len(rows):,}", flush=True)
    margins.flush()
    print(f"reranker margins -> {out}  (over {len(have_neg):,} pairs with negatives)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["guide", "margins"], required=True)
    ap.add_argument("--pairs", default="data/pairs_synth_taxonomy/train.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    guard_gpu(args.force)
    rows = load_pairs(args.pairs, args.limit)
    out = args.out or str(PROJECT_DIR / "data" / f"teacher_{args.mode}.f16")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    print(f"{args.mode}: {len(rows):,} pairs -> {out}")

    if args.mode == "guide":
        do_guide(rows, out, args.device, args.batch_size)
    else:
        do_margins(rows, out, args.device, args.batch_size)


if __name__ == "__main__":
    main()
