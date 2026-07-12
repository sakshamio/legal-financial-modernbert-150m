"""Stage 3: external benchmarks (MTEB) at every Matryoshka truncation dim.

WHY THIS EXISTS. Our own held-out eval (eval_retrieval.py) proves *graceful degradation* -- that the
representation is genuinely nested, 768 -> 64. It cannot prove the model is any GOOD, because we built
that eval set ourselves from the same sources we trained on. External benchmarks are the only way to
make a comparable claim.

CONTAMINATION. Stage 2 trains on LEDGAR, CUAD, FinQA and EDGAR sections. MTEB's `FinQARetrieval` IS
FinQA -- evaluating on it would be train/test contamination and would produce a flattering, meaningless
number. It is excluded from the headline set and only ever reported under a loud warning.

Pretraining contamination is a softer issue: our corpus contains US bills, so `BillSumUS` overlaps the
*pretraining* text (though not the retrieval supervision). That is normal for any pretrained model --
noted, not excluded.

PROMPTS. The model is trained with [QUERY]/[PASSAGE] prefixes, so external eval MUST apply them too --
otherwise we would be measuring a different model than the one we trained.
"""
import argparse
import json
from pathlib import Path

import mteb
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"
RESULTS_DIR = PROJECT_DIR / "benchmarks"

# Clean tasks: none of these supply stage-2 training supervision.
TASKS = {
    "financial": [
        "FiQA2018",                       # the standard financial retrieval benchmark (BEIR)
        "FinanceBenchRetrieval",
        "ESGReports",
    ],
    "legal": [
        "LegalBenchConsumerContractsQA",  # contracts QA -- closest to what we care about
        "LegalBenchCorporateLobbying",
        "AILAStatutes",                   # statute retrieval (non-US, but genuinely legal retrieval)
        "AILACasedocs",
    ],
}

# Trains on the same data -> any score here is contaminated and must not be quoted.
CONTAMINATED = ["FinQARetrieval"]


def matryoshka_dims(full_dim):
    ladder = [d for d in (1024, 768, 512, 256, 128, 64) if d <= full_dim]
    if full_dim not in ladder:
        ladder.insert(0, full_dim)
    return sorted(set(ladder), reverse=True)


def load_model(path, truncate_dim=None, device=None, use_prompts=True, trust=False):
    """Load with the SAME prefixes it was trained with.

    MTEB asks for prompt_name="query" for queries and "document" for the corpus; without these the
    model would see bare text at eval and prefixed text in training -- a silent train/eval mismatch.
    """
    kw = dict(truncate_dim=truncate_dim, device=device, trust_remote_code=trust)
    if use_prompts:
        kw["prompts"] = {"query": "[QUERY] ", "document": "[PASSAGE] "}
    return SentenceTransformer(path, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(CHECKPOINTS_DIR / "embedding_stage2"))
    ap.add_argument("--dims", type=int, nargs="*", default=None, help="default: the full Matryoshka ladder")
    ap.add_argument("--groups", nargs="*", default=["financial", "legal"])
    ap.add_argument("--include-contaminated", action="store_true", help="report FinQA too, clearly flagged")
    ap.add_argument("--baselines", nargs="*", default=[], help="e.g. BAAI/bge-small-en-v1.5 sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default=None, help="'cpu' to leave the GPU free for a training run")
    ap.add_argument("--no-prompts", action="store_true", help="for baselines that were not trained with our prefixes")
    ap.add_argument("--threads", type=int, default=0, help="cap CPU threads so a concurrent training run keeps its cores")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--tag", default=None, help="label for the results file (defaults to the model name)")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.threads:
        import torch
        torch.set_num_threads(args.threads)  # do not starve a concurrent GPU training run of its dataloader

    probe = SentenceTransformer(args.model, device=args.device, trust_remote_code=args.trust_remote_code)
    full_dim = probe.get_sentence_embedding_dimension()
    del probe
    dims = args.dims or matryoshka_dims(full_dim)

    task_names = [t for g in args.groups for t in TASKS[g]]
    if args.include_contaminated:
        print("!! including CONTAMINATED tasks -- these overlap stage-2 training data, do not quote them")
        task_names += CONTAMINATED

    print(f"model: {args.model} (dim {full_dim})")
    print(f"dims : {dims}")
    print(f"tasks: {task_names}\n")

    results = {}
    for dim in dims:
        model = load_model(args.model, truncate_dim=dim, device=args.device,
                           use_prompts=not args.no_prompts, trust=args.trust_remote_code)
        tasks = mteb.get_tasks(tasks=task_names)
        out = mteb.MTEB(tasks=tasks).run(
            model, output_folder=str(RESULTS_DIR / f"dim{dim}"), verbosity=0, overwrite_results=True
        )
        for r in out:
            scores = r.scores.get("test") or r.scores.get("dev") or []
            if not scores:
                continue
            s = scores[0]
            results.setdefault(r.task_name, {})[dim] = {
                "ndcg@10": s.get("ndcg_at_10"),
                "recall@100": s.get("recall_at_100"),
            }
        del model

    # nDCG@10 tells you ranking quality; Recall@100 tells you whether the gold passage even survives
    # into a reranker's candidate pool -- the number that matters if this feeds a RAG first stage.
    for metric in ["ndcg@10", "recall@100"]:
        print(f"\n=== {metric} ===")
        print(f"{'task':<34} " + "".join(f"{d:>9}" for d in dims))
        for task in task_names:
            if task not in results:
                continue
            row = "".join(f"{(results[task].get(d) or {}).get(metric) or 0:>9.4f}" for d in dims)
            flag = "  [CONTAMINATED]" if task in CONTAMINATED else ""
            print(f"{task:<34} {row}{flag}")

    tag = args.tag or Path(args.model).name.replace("/", "_")
    out_path = RESULTS_DIR / f"{tag}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out_path}")
    print("\nThe MRL signal to look for: nDCG should degrade GRACEFULLY 768 -> 64.")
    print("A cliff means the representation is not genuinely nested, only good at full width.")


if __name__ == "__main__":
    main()
