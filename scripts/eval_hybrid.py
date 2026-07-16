"""Evaluate the hybrid Stage-2 model on MTEB (dense head) and report vs the baselines.

The hybrid model is a custom nn.Module, not a sentence-transformers model, so MTEB can't load it by
name. This wraps its DENSE head in the minimal encode() interface MTEB needs. Matryoshka dims are
evaluated by truncating the dense vector. (The sparse head is scored separately by a hybrid retrieval
eval; MTEB here measures the dense embedding, which is the primary output.)

    python scripts/eval_hybrid.py --model checkpoints/stage2_hybrid --dims 768 256 64
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

from hybrid_model import HybridEmbedder

PROJECT_DIR = Path(__file__).resolve().parent.parent

# same legal+financial tasks as eval_benchmarks.py; FinQARetrieval excluded (Stage-2 trains on FinQA)
TASKS = ["FiQA2018", "FinanceBenchRetrieval", "ESGReports",
         "LegalBenchConsumerContractsQA", "LegalBenchCorporateLobbying", "AILAStatutes", "AILACasedocs"]


class HybridSTWrapper:
    """Minimal MTEB-compatible encoder over the hybrid model's dense head at a fixed truncation dim."""

    def __init__(self, model, tok, device, dim, max_len=256):
        self.model, self.tok, self.device, self.dim, self.max_len = model, tok, device, dim, max_len

    def encode(self, sentences, batch_size=64, **kw):
        out = []
        for s in range(0, len(sentences), batch_size):
            b = self.tok(list(sentences[s : s + batch_size]), padding=True, truncation=True,
                         max_length=self.max_len, return_tensors="pt")
            b = {k: v.to(self.device) for k, v in b.items()}
            with torch.no_grad():
                d = self.model(b["input_ids"], b["attention_mask"])["dense"][:, : self.dim]
            out.append(F.normalize(d, dim=-1).cpu().float().numpy())
        return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/stage2_hybrid")
    ap.add_argument("--dims", type=int, nargs="+", default=[768, 256, 64])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--out", default="benchmarks/hybrid_result.json")
    args = ap.parse_args()

    cfg = json.loads((Path(args.model) / "config.json").read_text())
    tok = HybridEmbedder.load_tokenizer_fix(PreTrainedTokenizerFast.from_pretrained(
        str(PROJECT_DIR / "tokenizer")))
    model = HybridEmbedder(cfg["init"], sparse=cfg["sparse"]).to(args.device).eval()
    sd = torch.load(Path(args.model) / "hybrid.pt", map_location=args.device)
    model.load_state_dict(sd)
    print(f"loaded hybrid model from {args.model}")

    import mteb
    results = {}
    for dim in args.dims:
        wrapper = HybridSTWrapper(model, tok, args.device, dim)
        for task_name in args.tasks:
            try:
                task = mteb.get_tasks(tasks=[task_name])
                res = mteb.MTEB(tasks=task).run(wrapper, output_folder=None, verbosity=0)
                score = res[0].scores["test"][0]["main_score"]
                results.setdefault(task_name, {})[dim] = score
                print(f"  dim {dim:>3} {task_name:<32} nDCG@10 {score:.4f}", flush=True)
            except Exception as e:
                print(f"  dim {dim:>3} {task_name:<32} FAILED: {type(e).__name__}: {e}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, default=float))

    # summary vs known baselines (same harness, from earlier committed runs)
    base = {"OURS-stage1": 0.367, "legal-bert": 0.317, "MiniLM-22M": 0.575,
            "bge-base": 0.618, "Qwen3-Emb-0.6B": 0.826, "Stage2-plain": 0.285}
    print("\n=== HYBRID vs baselines (legal+financial avg nDCG@10) ===")
    for dim in args.dims:
        vals = [results[t][dim] for t in results if dim in results[t]]
        if vals:
            print(f"  hybrid @dim {dim}: {sum(vals)/len(vals):.3f}  (over {len(vals)} tasks)")
    print("  ---")
    for k, v in sorted(base.items(), key=lambda x: x[1]):
        print(f"  {k:<16} {v:.3f}")


if __name__ == "__main__":
    main()
