"""Evaluate the trained embedding model's retrieval quality at each Matryoshka
truncation dimension on held-out (anchor, positive) pairs, to confirm quality
degrades gracefully rather than collapsing as dimension shrinks.
"""
import argparse
from pathlib import Path

from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator

PROJECT_DIR = Path(__file__).resolve().parent.parent
PAIRS_DIR = PROJECT_DIR / "data" / "pairs"
CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"

def matryoshka_dims(full_dim):
    """Must mirror train_matryoshka.matryoshka_dims -- evaluating at a dim the model was never trained
    to nest at would understate quality."""
    ladder = [d for d in (1024, 768, 512, 256, 128, 64) if d <= full_dim]
    if full_dim not in ladder:
        ladder.insert(0, full_dim)
    return sorted(set(ladder), reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(CHECKPOINTS_DIR / "embedding_stage2"))
    args = ap.parse_args()

    val_ds = load_from_disk(str(PAIRS_DIR / "val"))
    queries = {f"q{i}": r["anchor"] for i, r in enumerate(val_ds)}
    corpus = {f"c{i}": r["positive"] for i, r in enumerate(val_ds)}
    relevant_docs = {f"q{i}": {f"c{i}"} for i in range(len(val_ds))}
    print(f"eval set: {len(queries)} queries over a {len(corpus)}-doc corpus")

    model = SentenceTransformer(args.model_dir)
    dims = matryoshka_dims(model.get_sentence_embedding_dimension())

    # Recall@100 matters as much as nDCG@10 here: if this feeds a RAG first stage, what counts is
    # whether the gold passage survives into the reranker's candidate pool at all.
    print(f"\n{'dim':>6}  {'nDCG@10':>8}  {'MRR@10':>8}  {'Recall@10':>10}  {'Recall@100':>11}")
    for dim in dims:
        evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            truncate_dim=dim,
            name=f"dim{dim}",
            accuracy_at_k=[1, 10],
            precision_recall_at_k=[10, 100],
            show_progress_bar=False,
        )
        results = evaluator(model)
        p = f"dim{dim}_cosine"
        print(
            f"{dim:>6}  {results[f'{p}_ndcg@10']:>8.4f}  {results[f'{p}_mrr@10']:>8.4f}  "
            f"{results[f'{p}_recall@10']:>10.4f}  {results[f'{p}_recall@100']:>11.4f}"
        )


if __name__ == "__main__":
    main()
