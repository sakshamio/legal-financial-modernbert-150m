"""Stage 2: build contrastive training data -- (anchor, positive, hard_negative...) triplets.

Pairs are mined entirely from EXISTING labels and QA structure. No LLM synthetic generation: no
generation cost, no synthetic-data licensing questions, and the supervision is human-annotated.

Three things here matter more for final nDCG than any pretraining tweak:

1. HARD NEGATIVES (via BM25). `MultipleNegativesRankingLoss` with only in-batch (random) negatives is
   the weak version of contrastive training. A random negative is trivially separable, so the gradient
   is small and uninformative. A BM25-mined negative is lexically similar but semantically wrong --
   exactly the confusion the model must learn to resolve. This matters MORE on legal/financial text,
   where surface overlap is enormous (every contract says "shall", "party", "hereunder") and the real
   distinctions are subtle.

2. ASYMMETRIC PREFIXES. Most pairs are short-query -> long-passage, which is an asymmetric task: the
   query and the passage should NOT be encoded the same way. We prepend the dedicated [QUERY] /
   [PASSAGE] tokens reserved in the tokenizer (so no embedding resize is ever needed).
   NOTE: LEDGAR clause<->clause pairs are SYMMETRIC -- both sides are passages. Prefixing one side as a
   query would teach the model something false. Handled per-source.

3. LABEL METADATA. LEDGAR pairs carry their clause-type label so the batch sampler can avoid putting
   two same-label examples in one batch -- otherwise MNRL treats one as a NEGATIVE for the other, and
   we actively train the model to push apart two genuinely similar clauses. See train_matryoshka.py.
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

QUERY_PREFIX = "[QUERY] "
PASSAGE_PREFIX = "[PASSAGE] "

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


def q(text):
    return QUERY_PREFIX + " ".join(text.split())


def p(text):
    return PASSAGE_PREFIX + " ".join(text.split())


# --------------------------------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------------------------------
def build_ledgar(n_pairs, seed=0):
    """Same-clause-type provisions. SYMMETRIC (passage<->passage) -- both sides are contract text.

    Carries `label` so the batch sampler can keep same-label examples out of the same batch.
    """
    ds = load_dataset("coastalcph/lex_glue", "ledgar", split="train")
    by_label = defaultdict(list)
    for r in ds:
        t = " ".join(r["text"].split())
        if len(t) > 50:
            by_label[r["label"]].append(t)

    rng = random.Random(seed)
    labels = [l for l, v in by_label.items() if len(v) >= 2]
    out = []
    while len(out) < n_pairs and labels:
        lab = rng.choice(labels)
        a, b = rng.sample(by_label[lab], 2)
        out.append({"anchor": p(a), "positive": p(b), "label": f"ledgar-{lab}", "source": "ledgar"})
    print(f"  LEDGAR:  {len(out)} symmetric pairs across {len(labels)} clause types")
    return out


def build_cuad(n_pairs, seed=0):
    """Clause-type question -> annotated contract span. ASYMMETRIC."""
    ds = load_dataset("theatticusproject/cuad-qa", revision="refs/convert/parquet", split="train")
    out = []
    for r in ds:
        if len(out) >= n_pairs:
            break
        ans = r["answers"]["text"]
        if not ans:
            continue
        question = re.sub(r"^Highlight the parts.*?Details:\s*", "", r["question"]).strip()
        span = " ".join(" ".join(ans).split())
        if len(span) > 20:
            # the clause TYPE is the natural label: two CUAD examples asking the same clause-type
            # question are near-duplicates and must not be each other's negatives.
            out.append({"anchor": q(question), "positive": p(span), "label": f"cuad-{question[:60]}", "source": "cuad"})
    print(f"  CUAD:    {len(out)} asymmetric pairs")
    return out


def build_finqa(n_pairs, seed=0):
    """Financial question -> evidence context. ASYMMETRIC."""
    ds = load_dataset("ChanceFocus/flare-finqa", split="train")
    out = []
    for i, r in enumerate(ds):
        if len(out) >= n_pairs:
            break
        m = re.search(r"Context:\s*(.*?)\nQuestion:", r["query"], re.DOTALL)
        if not m:
            continue
        ctx, question = m.group(1).strip(), r["text"].strip()
        if len(ctx) > 50 and len(question) > 5:
            out.append({"anchor": q(question), "positive": p(ctx), "label": f"finqa-{i}", "source": "finqa"})
    print(f"  FinQA:   {len(out)} asymmetric pairs")
    return out


def build_edgar(n_pairs, seed=0):
    """Section query -> that filing's section text. ASYMMETRIC.

    The section TYPE is the label: every "risk factors" query shares one label, so the sampler will not
    let two risk-factor examples become negatives for each other (they legitimately look alike).
    """
    api = HfApi()
    info = api.dataset_info("eloukas/edgar-corpus", files_metadata=True)
    files = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith("train.jsonl"))
    rng = random.Random(seed)
    out = []
    for fname in rng.sample(files, min(4, len(files))):
        if len(out) >= n_pairs:
            break
        path = hf_hub_download("eloukas/edgar-corpus", fname, repo_type="dataset")
        ds = load_dataset("json", data_files=path, split="train")
        for r in ds:
            if len(out) >= n_pairs:
                break
            for key, query in EDGAR_SECTION_QUERIES.items():
                txt = (r.get(key) or "").strip()
                if len(txt) > 100:
                    out.append(
                        {"anchor": q(query), "positive": p(txt[:3000]), "label": f"edgar-{key}", "source": "edgar"}
                    )
    print(f"  EDGAR:   {len(out)} asymmetric pairs")
    return out


# --------------------------------------------------------------------------------------------------
# hard negatives
# --------------------------------------------------------------------------------------------------
def mine_hard_negatives(pairs, n_neg=2, seed=0):
    """BM25 hard negatives: lexically similar to the anchor, but NOT its positive.

    Why BM25 and not a dense model: we have no trained embedding model yet (that is the point of this
    stage), and legal text is exactly the regime where lexical overlap is high -- so BM25 surfaces
    genuinely confusable passages. Mined WITHIN each source, since a contract clause is not a
    meaningful negative for a financial-QA question (it would be trivially separable, i.e. useless).

    Guards against FALSE negatives: a candidate is skipped if it shares the anchor's label (same clause
    type / same section type), because such a passage may be a legitimate answer, and training the
    model to push it away is actively wrong.
    """
    from rank_bm25 import BM25Okapi

    rng = random.Random(seed)
    by_source = defaultdict(list)
    for i, ex in enumerate(pairs):
        by_source[ex["source"]].append(i)

    # TEXT-level guard, not label-level. Comparing the two examples' labels is NOT enough: the same
    # clause text can appear under more than one label, so a "different-label" candidate can still be
    # text that is a legitimate positive for this anchor's label. Rejecting it would train the model to
    # push away a true positive -- the exact failure mode this guard exists to prevent.
    texts_of_label = defaultdict(set)
    for ex in pairs:
        texts_of_label[ex["label"]].add(ex["positive"])
        texts_of_label[ex["label"]].add(ex["anchor"])

    for src, idxs in by_source.items():
        corpus = [pairs[i]["positive"] for i in idxs]
        bm25 = BM25Okapi([c.lower().split() for c in corpus])
        print(f"  mining {src}: {len(idxs)} passages...", flush=True)

        for pos_in_src, i in enumerate(idxs):
            ex = pairs[i]
            banned = texts_of_label[ex["label"]]
            scores = bm25.get_scores(ex["anchor"].lower().split())
            ranked = sorted(range(len(scores)), key=lambda j: -scores[j])

            negs = []
            for j in ranked:
                if len(negs) >= n_neg:
                    break
                if j == pos_in_src:
                    continue
                cand = pairs[idxs[j]]
                if cand["label"] == ex["label"] or cand["positive"] in banned:
                    continue  # a true positive in disguise
                negs.append(cand["positive"])

            # Pad with random draws if BM25 could not find enough clean candidates.
            tries = 0
            while len(negs) < n_neg and tries < 50:
                tries += 1
                cand = pairs[rng.choice(idxs)]
                if cand["label"] != ex["label"] and cand["positive"] not in banned and cand["positive"] not in negs:
                    negs.append(cand["positive"])
            for k, neg in enumerate(negs):
                ex[f"negative_{k}"] = neg
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-source", type=int, default=20000)
    ap.add_argument("--n-negatives", type=int, default=2)
    ap.add_argument("--no-hard-negatives", action="store_true")
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    PAIRS_DIR.mkdir(parents=True, exist_ok=True)

    print("building pairs (labels/QA structure only -- no LLM generation)")
    pairs = []
    pairs += build_ledgar(args.n_per_source, args.seed)
    pairs += build_cuad(args.n_per_source, args.seed)
    pairs += build_finqa(args.n_per_source, args.seed)
    pairs += build_edgar(args.n_per_source, args.seed)

    if not args.no_hard_negatives:
        print(f"\nmining {args.n_negatives} BM25 hard negatives per pair (within-source, label-guarded)")
        pairs = mine_hard_negatives(pairs, args.n_negatives, args.seed)

    random.Random(args.seed).shuffle(pairs)

    cols = ["anchor", "positive"] + [f"negative_{k}" for k in range(args.n_negatives)]
    cols = [c for c in cols if all(c in ex for ex in pairs)]
    data = {c: [ex[c] for ex in pairs] for c in cols}
    data["label"] = [ex["label"] for ex in pairs]  # consumed by the label-aware batch sampler
    data["source"] = [ex["source"] for ex in pairs]

    ds = Dataset.from_dict(data)
    split = ds.train_test_split(test_size=args.val_fraction, seed=args.seed)
    split["train"].save_to_disk(str(PAIRS_DIR / "train"))
    split["test"].save_to_disk(str(PAIRS_DIR / "val"))

    print(f"\ntotal {len(ds)} examples | columns: {cols + ['label', 'source']}")
    print(f"train {len(split['train'])} / val {len(split['test'])} -> {PAIRS_DIR}")
    ex = split["train"][0]
    print(f"\nsample:\n  anchor  : {ex['anchor'][:90]}")
    print(f"  positive: {ex['positive'][:90]}")
    if "negative_0" in ex:
        print(f"  hard neg: {ex['negative_0'][:90]}")


if __name__ == "__main__":
    main()
