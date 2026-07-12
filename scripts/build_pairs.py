"""Build (anchor, positive) contrastive pairs for stage-2 embedding training,
mined entirely from existing labels/QA structure -- no LLM synthetic generation.

Sources:
  - LEDGAR (coastalcph/lex_glue, config "ledgar"): same-clause-type provisions
    paired as positives; in-batch negatives come from other pairs during training.
  - CUAD-QA (theatticusproject/cuad-qa): clause-type question -> answer span.
  - FinQA (ChanceFocus/flare-finqa): question -> extracted evidence context.
  - EDGAR-CORPUS (eloukas/edgar-corpus): natural-language section query -> section text.
"""
import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, hf_hub_download

PROJECT_DIR = Path(__file__).resolve().parent.parent
PAIRS_DIR = PROJECT_DIR / "data" / "pairs"

EDGAR_SECTION_QUERIES = {
    "section_1": "What is the description of this company's business?",
    "section_1A": "What are the risk factors disclosed in this company's SEC 10-K filing?",
    "section_2": "What properties does this company own or lease?",
    "section_3": "What legal proceedings is this company involved in?",
    "section_5": "What is the market for this company's common equity and related stockholder matters?",
    "section_7": "What is management's discussion and analysis of financial condition and results of operations?",
    "section_7A": "What are this company's quantitative and qualitative disclosures about market risk?",
    "section_8": "What are this company's financial statements and supplementary data?",
    "section_9A": "What are this company's disclosure controls and procedures?",
    "section_11": "What is this company's executive compensation?",
}


def build_ledgar_pairs(n_pairs, seed=0):
    ds = load_dataset("coastalcph/lex_glue", "ledgar", split="train")
    by_label = defaultdict(list)
    for r in ds:
        text = r["text"].strip()
        if len(text) > 50:
            by_label[r["label"]].append(text)
    rng = random.Random(seed)
    pairs = []
    labels = [l for l, texts in by_label.items() if len(texts) >= 2]
    while len(pairs) < n_pairs and labels:
        label = rng.choice(labels)
        a, b = rng.sample(by_label[label], 2)
        pairs.append((a, b))
    print(f"  LEDGAR: {len(pairs)} pairs from {len(labels)} clause-type labels")
    return pairs


def build_cuad_pairs(n_pairs, seed=0):
    ds = load_dataset("theatticusproject/cuad-qa", revision="refs/convert/parquet", split="train")
    pairs = []
    for r in ds:
        if len(pairs) >= n_pairs:
            break
        answers = r["answers"]["text"]
        if not answers:
            continue
        question = re.sub(r"^Highlight the parts.*?Details:\s*", "", r["question"]).strip()
        answer = " ".join(answers).strip()
        if len(answer) > 20:
            pairs.append((question, answer))
    print(f"  CUAD-QA: {len(pairs)} pairs")
    return pairs


def build_finqa_pairs(n_pairs, seed=0):
    ds = load_dataset("ChanceFocus/flare-finqa", split="train")
    pairs = []
    for r in ds:
        if len(pairs) >= n_pairs:
            break
        m = re.search(r"Context:\s*(.*?)\nQuestion:", r["query"], re.DOTALL)
        if not m:
            continue
        context = m.group(1).strip()
        question = r["text"].strip()
        if len(context) > 50 and len(question) > 5:
            pairs.append((question, context))
    print(f"  FinQA: {len(pairs)} pairs")
    return pairs


def build_edgar_section_pairs(n_pairs, seed=0):
    api = HfApi()
    info = api.dataset_info("eloukas/edgar-corpus", files_metadata=True)
    train_files = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith("train.jsonl"))
    rng = random.Random(seed)
    sample_files = rng.sample(train_files, min(3, len(train_files)))
    pairs = []
    for fname in sample_files:
        if len(pairs) >= n_pairs:
            break
        path = hf_hub_download("eloukas/edgar-corpus", fname, repo_type="dataset")
        ds = load_dataset("json", data_files=path, split="train")
        for r in ds:
            if len(pairs) >= n_pairs:
                break
            for key, query in EDGAR_SECTION_QUERIES.items():
                text = (r.get(key) or "").strip()
                if len(text) > 100:
                    pairs.append((query, text))
    print(f"  EDGAR-CORPUS sections: {len(pairs)} pairs")
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-source", type=int, default=15000)
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    PAIRS_DIR.mkdir(parents=True, exist_ok=True)

    print("Building stage-2 contrastive pairs...")
    all_pairs = []
    all_pairs += build_ledgar_pairs(args.n_per_source, args.seed)
    all_pairs += build_cuad_pairs(args.n_per_source, args.seed)
    all_pairs += build_finqa_pairs(args.n_per_source, args.seed)
    all_pairs += build_edgar_section_pairs(args.n_per_source, args.seed)

    rng = random.Random(args.seed)
    rng.shuffle(all_pairs)

    ds = Dataset.from_dict({
        "anchor": [p[0] for p in all_pairs],
        "positive": [p[1] for p in all_pairs],
    })
    split = ds.train_test_split(test_size=args.val_fraction, seed=args.seed)
    split["train"].save_to_disk(str(PAIRS_DIR / "train"))
    split["test"].save_to_disk(str(PAIRS_DIR / "val"))
    print(f"\ntotal pairs: {len(ds)} (train={len(split['train'])}, val={len(split['test'])})")
    print(f"saved to {PAIRS_DIR}/train and {PAIRS_DIR}/val")


if __name__ == "__main__":
    main()
