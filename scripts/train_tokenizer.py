"""Train a domain-specific byte-level BPE tokenizer on the legal/financial corpus.

Special tokens are placed at fixed low ids and are NOT the ModernBERT defaults
(those assume ModernBERT's own OLMo-derived tokenizer) -- pretrain_mlm.py reads
the ids back out of this tokenizer and sets them explicitly on the model config.
"""
import argparse
import glob
import json
import random
from pathlib import Path

from tokenizers import AddedToken, ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
TOKENIZER_DIR = PROJECT_DIR / "tokenizer"

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

# Reserved now so stage 2 (asymmetric retrieval) can use dedicated tokens without ever resizing the
# embedding matrix mid-project -- resizing a from-scratch model's embeddings after pretraining means
# randomly-initialised rows in an otherwise-trained matrix.
RESERVED_TOKENS = ["[QUERY]", "[PASSAGE]"] + [f"[RESERVED_{i}]" for i in range(14)]

# Domain tokens are added ONLY where BPE demonstrably fails. Measured on the domain-trained BPE:
# single legal words already tokenize to 1 token (`indemnification`, `notwithstanding`, `thereunder`,
# `EBITDA`, `GAAP`), so a legal word-list would be redundant. What shatters is CITATION MACHINERY --
# `Fed. R. Civ. P.` cost 8 tokens, `U.S.C.` and `C.F.R.` 6 each, `e.g.` 4 -- and these are both
# high-frequency in statutes/case law AND highly discriminative retrieval keys.
#
# Deliberately NOT doing digit-splitting (Llama/GPT-4 style): digits are only ~2% of characters and
# ~5% of tokens here, so splitting them would inflate total tokens ~9% -- cancelling the tokenizer's
# entire fertility win -- to buy numeracy, which matters far more for generation than for retrieval.
DOMAIN_TOKENS = [
    # citation machinery
    "U.S.C.", "C.F.R.", "U.S.", "F.2d", "F.3d", "F. Supp.", "S. Ct.", "L. Ed.",
    "Fed. R. Civ. P.", "Fed. R. Evid.", "et seq.", "id.", "infra", "e.g.", "i.e.", "cf.", "v.",
    "§§", "¶¶",
    # corporate entity suffixes
    "Inc.", "Corp.", "LLP", "L.P.", "Ltd.", "Co.", "N.A.", "plc",
    # SEC form types
    "10-K", "10-Q", "8-K", "20-F", "6-K", "S-1", "DEF 14A",
    # finance shorthand
    "IFRS", "ROE", "ROI", "P/E", "YoY", "QoQ", "Q1", "Q2", "Q3", "Q4",
    # frequent legal connectives that BPE split
    "whereas",
    # --- found by scanning the corpus for high-frequency punctuation-internal abbreviations ---
    # (a data-driven sweep, not guesswork: e.g. "P.A." occurs 665k times in a 5.7GB sample and costs
    #  4 tokens; "K.S.A." 119k times at 6 tokens; "Sp. Sess. P.A." at 8.) All are statutory-citation
    #  machinery from the state codes. Only terms costing >=3 tokens are included -- the 2-token ones
    #  ("Sec.", "No.", month abbreviations) would save a single token each, which is not worth a slot.
    "P.A.", "P.L.", "R.S.", "G.S.", "K.S.A.", "H.B.", "S.B.", "W.S.", "G.L.", "A.L.", "C.S.", "R.C.M.",
    "Sp. Sess.", "Reg. Sess.", "Ex. Sess.",
    "Pt.", "Subsec.", "Subsecs.", "Subd.", "Subdiv.", "Secs.", "Amend.",
    # regional reporters (case-law citations)
    "N.W.", "N.E.", "S.W.", "S.E.", "N.W.2d", "N.E.2d", "S.W.2d", "So.2d", "A.2d", "P.2d",
    "D.C.",
]


def iter_sample_lines(sample_bytes, seed=0):
    """Stream text lines from all raw shards, proportionally sampled up to sample_bytes total."""
    files = sorted(glob.glob(str(RAW_DIR / "*.jsonl")))
    if not files:
        raise SystemExit(f"No raw corpus files found in {RAW_DIR} -- run prepare_corpus.py first")
    sizes = {f: Path(f).stat().st_size for f in files}
    total_size = sum(sizes.values())
    rng = random.Random(seed)
    written = 0
    # round-robin over files, each contributing roughly proportional to its share of total_size
    handles = {f: open(f, "r", encoding="utf-8") for f in files}
    budgets = {f: sample_bytes * (sizes[f] / total_size) for f in files}
    used = {f: 0 for f in files}
    active = list(files)
    while active and written < sample_bytes:
        for f in list(active):
            if used[f] >= budgets[f]:
                active.remove(f)
                continue
            line = handles[f].readline()
            if not line:
                active.remove(f)
                continue
            try:
                text = json.loads(line)["text"]
            except (json.JSONDecodeError, KeyError):
                continue
            used[f] += len(line.encode("utf-8"))
            written += len(text.encode("utf-8"))
            yield text
    for fh in handles.values():
        fh.close()
    print(f"tokenizer training sample: {written/1024**3:.2f}GB across {len(files)} sources")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=50368)
    ap.add_argument("--sample-gb", type=float, default=3.0, help="approx text volume to train the tokenizer on")
    args = ap.parse_args()

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    # Keep the FINAL vocab at exactly --vocab-size (50,368 = 787 x 64). It must stay a multiple of 64
    # for tensor-core alignment, so we shrink the learned BPE by however many tokens we add back.
    n_added = 2 * len(DOMAIN_TOKENS) + len(RESERVED_TOKENS)  # bare + space-prefixed variant each
    bpe_vocab = args.vocab_size - n_added

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        iter_sample_lines(int(args.sample_gb * 1024**3)),
        vocab_size=bpe_vocab,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
    )

    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    # Add BOTH a bare and a space-prefixed variant of every domain token (the GPT-2 approach).
    #
    # The obvious alternative -- one token with lstrip=True -- silently EATS the preceding space and
    # never restores it on decode: "Roe v. Wade" round-trips as "Roev. Wade". That is not cosmetic; it
    # means the model cannot distinguish "Roe v." from "Roev.". Using lstrip=False alone round-trips
    # correctly but leaves a stray whitespace token, giving up a third of the compression.
    # Measured on citation-heavy text: lstrip=True -34.8% but BROKEN; lstrip=False -24.6% OK;
    # both-variants -34.8% AND lossless.
    added = []
    for t in DOMAIN_TOKENS:
        added.append(AddedToken(t, lstrip=False, rstrip=False, normalized=False))
        added.append(AddedToken(" " + t, lstrip=False, rstrip=False, normalized=False))
    fast_tokenizer.add_tokens(added)
    fast_tokenizer.add_special_tokens({"additional_special_tokens": RESERVED_TOKENS})

    total = len(fast_tokenizer)
    print(f"BPE vocab {bpe_vocab} + {len(added)} domain (bare+spaced) + {len(RESERVED_TOKENS)} reserved = {total}")
    assert total % 64 == 0, f"vocab {total} must stay a multiple of 64 for tensor-core alignment"

    fast_tokenizer.save_pretrained(str(TOKENIZER_DIR))
    print(f"saved tokenizer to {TOKENIZER_DIR}, vocab_size={total}")
    print(
        "special ids:",
        {
            "pad": fast_tokenizer.pad_token_id,
            "unk": fast_tokenizer.unk_token_id,
            "cls": fast_tokenizer.cls_token_id,
            "sep": fast_tokenizer.sep_token_id,
            "mask": fast_tokenizer.mask_token_id,
        },
    )


if __name__ == "__main__":
    main()
