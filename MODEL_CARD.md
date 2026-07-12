---
license: cc-by-nc-sa-4.0
language:
- en
library_name: transformers
pipeline_tag: fill-mask
tags:
- modernbert
- legal
- finance
- retrieval
- sentence-transformers
- matryoshka
- pretraining-from-scratch
datasets:
- common-pile/caselaw_access_project
- pile-of-law/pile-of-law
- eloukas/edgar-corpus
- albertvillanova/legal_contracts
- HuggingFaceFW/fineweb-edu
---

# legal-financial-modernbert-150m

A **from-scratch** ModernBERT encoder — random init, no pretrained checkpoint, no distillation — trained
on public **legal and financial** documents, becoming a **Matryoshka** embedding model whose vectors
truncate to smaller dimensions without retraining.

Trained on a single **NVIDIA DGX Spark (GB10)**.

> ## ⚠️ Training in progress — these weights are not usable yet
>
> Stage-1 pretraining is a **30-day** run. Checkpoints appear here as it proceeds (see
> [Checkpoint series](#checkpoint-series)). The contrastive/Matryoshka stage has not run at all.
>
> This card documents the plan, the measurements, and the dead ends. **No quality claims are made.**

---

## Configuration

| | |
| --- | --- |
| Params | **149.7M** (ModernBERT-base geometry, random init) |
| Corpus | **124 GB deduplicated → 25.79B tokens** |
| Mix | **80% legal/financial · 20% general (FineWeb-Edu)** |
| Vocab | **50,368** = 50,196 BPE + 156 domain tokens + 16 reserved |
| Training | **271,000 steps → 71B tokens → 2.76 epochs → 475 tokens/param** |
| Context | 1024 (long-context extension to 8192 planned) |
| Embedding dims | `[768, 512, 256, 128, 64]` (Matryoshka) |

---

## Checkpoint series

Intermediate checkpoints are preserved as **git branches** — load any point in training:

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

REPO = "sakshamio/legal-financial-modernbert-150m"
model = AutoModelForMaskedLM.from_pretrained(REPO, revision="step-56000")
tok   = AutoTokenizer.from_pretrained(REPO, revision="step-56000")
```

28 snapshots, **log-spaced early** (where learning dynamics move fastest) then regular. `main` tracks
the latest weights.

---

## The GPU is bandwidth-bound, and it changes every decision

The DGX Spark pairs a capable GPU with **273 GB/s** of unified LPDDR5X. Measured **Model FLOP
Utilization: ~9–19%**.

`nvidia-smi` reports **96–100% "utilization"** the entire time. That only means *a kernel is resident* —
it says nothing about whether the math units are working. They aren't: they stall ~85% of cycles waiting
on memory. **The GPU is "fully utilized" and simultaneously doing almost nothing.**

Consequence: **only optimizations that move fewer BYTES help. Nothing that adds work does.**

| Change | Result |
| --- | --- |
| `sparse_prediction=True` | **+23.6%**, −32% memory |
| Full-graph `torch.compile` | **+22.7%** |
| Micro-batch 32 → 64 (2× memory) | **−0.3%** |
| Micro-batch 128 | **OOM — killed the machine** |

`sparse_prediction` gathers masked positions *before* the LM head. Only ~30% of positions carry MLM
loss, so the dense path projected 32k positions × 50,368 vocab and discarded ~70% of the result — about
3.3 GB of pointless logit traffic per micro-batch. Biggest single win in the project.

Batch size is a **pure memory knob**: with the effective batch fixed by gradient accumulation the
gradients are mathematically identical, so it changes neither speed nor quality. Doubling memory bought
−0.3%. Quadrupling it OOM'd a 121 GB machine.

**Net: +70% throughput (16,715 → ~28,400 tok/s), entirely from moving fewer bytes.**

### FP8 works on this chip — and is still slower than bf16

Recorded carefully, because we got it wrong twice.

**First conclusion (wrong):** *"FP8 is impossible here — `torchao` ships no aarch64 wheels, and NVIDIA's
`transformer_engine` builds but dies at import with `undefined symbol:
cublasLtGroupedMatrixLayoutInit_internal`."*

That was a **venv artifact.** We tested under torch 2.9, where torchao silently disables itself
(*"requires torch >= 2.11"*). Under **torch 2.13**, `torch._scaled_mm` runs FP8 natively on sm_121 and
torchao's float8 training path works fine.

**FP8 is available. It just loses:**

| config | tok/s |
| --- | --- |
| bf16 + `reference_compile` | 17,715 |
| **bf16 + full `torch.compile`** | **21,740** |
| FP8 + full `torch.compile` | 20,254 |
| FP8 eager (no compile) | 8,311 |

At 150M params the per-linear scaling/cast overhead exceeds the bandwidth it saves. (FP8 in eager mode
is *2× slower* than bf16 — those scaling ops must be fused by `torch.compile` or they're pure cost.)

**Verdict: bf16 + full compile.** A measured loss, not a toolchain excuse.

---

## Why 150M and not 400M — the Chinchilla trap

This project started aimed at 395M. That was a mistake, and an easy one.

Chinchilla says ~20 tokens/param is compute-optimal. 395M × 20 ≈ our token budget, so 395M *looks*
right-sized. **But Chinchilla is a result for causal LMs.** Masked language modeling is far less
sample-efficient — only the ~30% masked positions produce a loss signal — and real encoders are trained
*orders of magnitude* past Chinchilla:

| Model | Params | Tokens seen | tokens/param |
| --- | --- | --- | --- |
| BERT-base | 110M | ~128B | ~1,160 |
| ModernBERT-base | 150M | ~2T | ~13,300 |
| 395M @ our budget | 395M | 12.7B | **~32** ❌ |
| **150M @ our budget** | **150M** | **71B** | **~475** ✅ |

The binding constraint is **tokens, not capacity**. Under a *fixed wall-clock budget* a smaller model
runs faster, sees more data, and lands closer to convergence — a 395M model here would simply be a
bigger, more undertrained artifact.

It also matches practice: essentially every strong production embedding model is **22M–150M** (MiniLM
22M, bge-small 33M, gte-base 110M, ModernBERT-base 150M). 400M encoders exist, but they were trained on
trillions of tokens.

---

## Corpus (124 GB deduplicated → 25.79B tokens)

| Source | Size | |
| --- | --- | --- |
| Case law — `common-pile/caselaw_access_project` | 36.5 GB | CC0 |
| SEC 10-K filings — `eloukas/edgar-corpus` | 30.1 GB | **maxed** |
| **General — `HuggingFaceFW/fineweb-edu`** | **29.4 GB** | **20% mix** |
| Contracts — `pile-of-law` Atticus + `albertvillanova/legal_contracts` | 15.1 GB | **maxed** |
| SEC proceedings / tax rulings — `pile-of-law` | 8.4 GB | **maxed** |
| Regulations & statutes — `pile-of-law` | 4.5 GB | **maxed** |

### Why 20% general data

Training from scratch means there is **no pretrained worldview to inherit** — and real retrieval queries
arrive in plain English, not contract prose. A model that has only ever seen contracts may embed a
casual question poorly.

20% FineWeb-Edu buys ~13B general tokens (**more than GPT-2 ever saw**) at the cost of ~13B domain
tokens. Precedent: **BloombergGPT deliberately trained ~50/50** financial/general. We stay domain-heavy
at 80/20 because this is a *retrieval* model evaluated purely on legal/financial tasks. Adding data also
*lowers* epochs, which is a bonus.

Honest caveat: 71B tokens is ~30× less than ModernBERT's 2T. You cannot buy a real worldview at this
scale — only general linguistic competence.

### Public financial and contract text is genuinely exhausted

Both eval-critical domains are pulled to their public ceilings. **Rejected after inspection:**
- `JanosAudran/financial-reports-sec` — sentence-split **10-Ks we already have**.
- `lexlms/lex_files` — its contracts split duplicates `albertvillanova`, its CourtListener split overlaps
  the Caselaw Access Project, and the remainder is UK/EU/Canada/India law. Our evaluation is entirely US.
- **Patents** (`HUPD`, 125 GB) — genuinely enterprise documents, but a genre we never evaluate.

### Deduplication removed 12.88 GB (9.4%)

Normalized near-dedup (lowercase, digits→`#`, strip punctuation — catches boilerplate differing only by
date, dollar amount, or section number). 84M unique paragraphs. **No upsampling anywhere** — regulations
got a real share by pulling *unique* statutory text (state codes, US bills, US Code, Federal Register),
not by repeating CFR six times.

| source | duplicate text |
| --- | --- |
| **Regulations** | **36.8%** |
| **Contracts** | **30.9%** |
| EDGAR 10-Ks | 6.5% |
| General web | 3.0% |
| Case law | 1.2% |

Contracts and regulations were *far* more boilerplate-heavy than a 20k-doc sample suggested. Undeduped,
a nominal "2.76 epochs" would be many more passes over risk-factor templates and forward-looking-statement
disclaimers, and fewer over substance — exactly how a token-limited encoder overfits to boilerplate.

### Three landmines worth sharing

**1. A 168 MB single line silently truncated the corpus.** Dedup was rewritten in Go (25 MB/s on 1 core →
604 MB/s on 12; ~87 min → ~4 min, verified byte-identical to the Python version). But Go's
`bufio.Scanner` has a max token size, and on a longer line it **returns false with no error** — it does
not fail, it just *stops*. The state codes contain a 168 MB document, so that source was truncated to
**36 of 88,804 documents and reported success**. We would have trained on a corpus missing 99.96% of the
regulatory data we had gone out of our way to source. Fixed with an unbounded reader plus a doc-count
guard against the raw corpus.

**2. A 16.2M-character document** blew individual tokenizer workers to **20 GB RSS**. On a
*unified-memory* machine that starves the whole system — throughput collapsed from ~3,000 to ~15
docs/sec. Fix: **split** (not truncate) documents over 50k chars; lossless, since everything is
re-chunked into fixed blocks anyway.

**3. `datasets` streaming can segfault the interpreter.** Abandoning a streaming iterator mid-shard —
which happens every time a source hits its byte budget — leaves background threads broken and Python
dies at exit with `PyGILState_Release`. Harmless in a probe; fatal in a 10-hour unattended corpus build.
Fix: read parquet directly with pyarrow.

---

## Tokenizer

Custom byte-level BPE, **50,368 vocab** (multiple of 64 for tensor-core alignment), trained on the mixed
corpus. **Fertility vs ModernBERT's general BPE: 6.8% fewer tokens** — contracts **14.7%**, case law
7.7%, and general web **0.3% (parity — we didn't get worse at general text)**.

### Custom tokens: added only where BPE structurally *cannot* learn

BPE's pre-tokenizer splits on punctuation, so `U.S.C.` becomes `["U",".","S",".","C","."]` and **can
never be merged, however frequent it is** — and it occurs **4.9 million times**. That is a *structural*
failure, not a frequency one, and it is the entire criterion.

| term | before | after |
| --- | --- | --- |
| `Fed. R. Civ. P.` | 8 tokens | **1** |
| `U.S.C.` / `C.F.R.` | 6 tokens | **1** |
| `K.S.A.`, `Sp. Sess. P.A.` | 6–8 tokens | **1** |
| `e.g.` / `i.e.` | 4 tokens | **1** |
| `P.A.` (665k occurrences) | 4 tokens | **1** |

**Result: 38.7% fewer tokens on citation-heavy legal text**, lossless round-trip, for **0.34% of vocab**.
The 78 terms were found by *scanning the corpus* for high-frequency punctuation-internal abbreviations —
not by brainstorming a list.

**Deliberately NOT added:**

- **Proper nouns (cities, states, companies).** BPE already learned every one that matters — `Delaware`,
  `California`, `Texas`, `Chicago` are all 1 token *because they're frequent*. What fragments is
  `Poughkeepsie`, and that is **correct**: its embedding row would receive ~13k gradient updates while
  its subwords each get millions. **A rare token's embedding is worse than composing from well-trained
  subwords**, and vocab is zero-sum — 19,000 US cities would eat 38% of the vocab.
- **A legal word list.** `indemnification`, `notwithstanding`, `thereunder`, `EBITDA`, `GAAP` are already
  single tokens.
- **Digit-splitting** (Llama/GPT-4 style, the textbook fix for financial text). Digits are only ~2% of
  characters and ~5% of tokens here, so splitting them would inflate total tokens ~9% — **cancelling the
  tokenizer's entire fertility win** — to buy numeracy, which matters far more for generation than for
  retrieval matching.

> **The bug this nearly shipped.** The natural implementation (`AddedToken(lstrip=True)`) gave the best
> compression but silently **ate the preceding space**: `"Roe v. Wade"` round-tripped as `"Roev. Wade"`.
> The model could not have distinguished `Roe v.` from `Roev.` on *every citation in the corpus*. Fixed
> by adding **both bare and space-prefixed variants** (the GPT-2 approach) — full compression *and*
> lossless. Caught only because the test asserted round-trip equality, not just token counts.

`[QUERY]`, `[PASSAGE]` and 14 spare tokens are **reserved now**, so stage 2 never has to resize the
embedding matrix (which would inject random rows into an otherwise-trained matrix).

---

## Data layout: a flat uint16 memmap, not Arrow

Vocab 50,368 < 65,536, so **every token id fits in a uint16**. The packed corpus is therefore just a flat
51.5 GB byte array, and "packing into 1024-token blocks" is pure arithmetic —
`block[i] = mmap[i*1024 : (i+1)*1024]`. No grouping pass, no `save_to_disk`, no Arrow. (The nanoGPT
layout.) Round-tripping 124 GB through Arrow three times projected to **~7 hours**; this took ~2. It is
also faster to *read* during training — mmap + OS page cache, zero-copy.

> **The vocab bug this caught.** `PreTrainedTokenizerFast.vocab_size` returns **only the learned BPE
> (50,196)** — it **excludes added tokens**. Real ids reach **50,367**. The model was being built with a
> 50,196-row embedding matrix, so every domain token and `[QUERY]`/`[PASSAGE]` would have **indexed past
> the end of it**. Use `len(tokenizer)`. Surfaced only because the dataset test asserted
> `max_id < vocab_size` instead of merely checking that it ran.

**A note on Go:** dedup was rewritten in Go (24× faster) because it was genuinely single-threaded Python.
Packing was **not**, because tokenization is already Rust (HF `tokenizers`) running parallel across
cores — Go would add nothing, and a Go tokenizer binding that diverged even slightly from our 156
AddedTokens would silently corrupt a 30-day run. *Use Go where the work is single-threaded Python, not
where it is already Rust.*

---

## Training recipe

| | |
| --- | --- |
| Objective | MLM, **30% masking** (beats BERT's classic 15%) |
| Precision | bf16 + full-graph `torch.compile` + `sparse_prediction` |
| Optimizer | fused AdamW, wd 1e-5, grad-clip 1.0 |
| Schedule | warmup-stable-decay (5% / 10%) — the long stable phase means mid-run checkpoints are usable |
| Peak LR | **2e-4** — *derived*: ModernBERT-base used ~8e-4 at a ~4.7M-token batch; ours is 18× smaller, so `8e-4/√18 ≈ 1.9e-4` |
| Effective batch | 256 seq × 1024 = **262,144 tokens/step** |
| Steps | **271,000** |

Verified rather than assumed: dynamic masking (fresh mask each pass), global shuffle (domains
interleave), RoPE thetas (local 10k / global 160k), and **full-state resume** — kill-and-resume was
tested end-to-end and loss continued from 9.93 rather than restarting at 10.9.

---

## Planned: Matryoshka embedding stage

`SentenceTransformer` (mean pooling) trained with `MultipleNegativesRankingLoss` inside `MatryoshkaLoss`,
dims **`[768, 512, 256, 128, 64]`**.

Contrastive pairs are **mined from existing labels — no LLM-generated data**: LEDGAR (clause-type),
CUAD (clause question → contract span), FinQA (question → evidence), EDGAR (section query → section).

Planned improvements: **BM25 hard negatives** (in-batch-only negatives is the weak version of MNRL); a
**label-aware batch sampler** (LEDGAR same-label pairs create *false negatives* — two genuinely similar
clauses in one batch train the model to push them apart); and asymmetric `[QUERY]`/`[PASSAGE]` prefixes
(noting LEDGAR clause↔clause pairs are **symmetric** and need passage/passage).

Evaluation reports **nDCG@10 *and* Recall@100** at every truncation dim — Recall@100 matters because if
this feeds a RAG first stage, what counts is whether the gold passage survives into the reranker's
candidate pool at all. The success signal for MRL is *graceful degradation* from 768 → 64: that is what
proves the representation is genuinely nested, not merely good at full width.

---

## Intended use & limitations

**Intended:** research on domain-specific retrieval over legal/financial documents; a reference point for
from-scratch encoder training on modest single-node hardware.

**Not intended:** legal or financial advice, or any decision about real matters. This is an experiment.

**Limitations, plainly:**
- **Still undertrained by encoder standards.** ~475 tokens/param vs ~1,160 for BERT-base and ~13,300 for
  ModernBERT-base. Expect it to trail well-trained general encoders on many tasks.
- **English, US-centric.** US case law, US federal/state regulation, SEC filings.
- **Corpus bias.** Historical case law carries the biases of its era, and its OCR artifacts.
- **Cross-document attention** within packed blocks is not masked (standard for MLM, but worth knowing).
- **Coarse near-dedup**, not MinHash — company-name-substituted boilerplate variants survive.
- **LEDGAR false negatives** in contrastive training, until the label-aware sampler lands.

## License

**CC-BY-NC-SA-4.0 — non-commercial.** The corpus includes Pile of Law subsets (CC-BY-NC-SA), and that
term propagates. This is a research artifact.
