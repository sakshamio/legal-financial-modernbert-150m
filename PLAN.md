# Legal/Financial Matryoshka Embedding Model — Training Plan

**Goal:** train a text embedding model *from scratch* (random init — no pretrained checkpoint, no
distillation) on public legal + financial corpora, with Matryoshka Representation Learning so the
embeddings can be truncated to smaller dimensions without retraining. One DGX Spark (GB10).
Research/personal use.

**Status:** stage-1 pretraining RUNNING (271,000 steps, ~30 days). Corpus built, deduped, packed;
full-state resume verified.

---

## 0. Headline configuration

| | |
| --- | --- |
| Model | ModernBERT-base geometry, **149.7M params**, random init |
| Corpus | **124 GB deduped → 25.79B tokens** (80% legal/financial, 20% FineWeb-Edu) |
| Vocab | **50,368** = 50,196 BPE + 156 domain tokens + 16 reserved |
| Throughput | **~28,400 tok/s** measured in Trainer (bf16 + `sparse_prediction` + full `torch.compile`) |
| Budget | **30 days → 71B tokens → 2.76 epochs → 475 tokens/param** |
| Steps | **271,000** @ effective batch 262,144 tokens |
| Stage 2 | MRL dims `[768, 512, 256, 128, 64]` |

Throughput improved **+70%** during tuning (16,715 → ~28,400 tok/s) and the corpus tripled — together
lifting tokens/param from 203 → **475**.

---

## 1. Hardware: the GB10 is bandwidth-bound, and that governs everything

| Property | Value |
| --- | --- |
| Chip | NVIDIA GB10 (Grace Blackwell), **48 SMs**, sm_121 |
| Memory | 121 GB **unified** (CPU + GPU share one pool) |
| Memory bandwidth | **273 GB/s** LPDDR5X — the binding constraint |

**Measured MFU: ~9–19%.** At 25,400 tok/s the model does ~18.6 TFLOPS against a 100–213 TFLOPS bf16
peak. `nvidia-smi` reports 96–100% "utilization", but that only means *a kernel is resident* — the math
units are stalled ~85% of the time waiting on memory.

**Therefore the only optimizations that work are ones that MOVE FEWER BYTES.** Everything we measured
confirms this, in both directions:

| Change | Effect | Why |
| --- | --- | --- |
| `sparse_prediction=True` | **+23.6%**, −32% mem | Gathers masked positions *before* the LM head. Only ~30% of positions carry loss, so the dense path projected 32k positions × 50,368 vocab and threw ~70% away. Biggest single win. |
| Full-graph `torch.compile` | **+22.7%** | Beats ModernBERT's built-in `reference_compile` (which only compiles the MLP). More fusion → fewer round-trips. |
| Tied embeddings | already on | LM head shares the 38.7M embedding matrix instead of carrying a second copy. |
| Micro-batch 32 → 64 | **−0.3%**, 2× memory | Adds bytes, buys nothing. |
| Micro-batch 128 | **OOM** (~109 GB) | Killed the machine. |

Micro-batch is therefore a pure memory knob — with effective batch held fixed by gradient accumulation
it is *mathematically identical* (same gradients, no BatchNorm), so it affects neither speed nor
quality. We keep it at 32 for OOM headroom.

### FP8: it works, and it is still slower than bf16

Worth recording carefully, because we got this wrong twice.

- **First conclusion (WRONG):** "FP8 impossible — torchao has no aarch64 wheels, transformer_engine
  fails with `undefined symbol: cublasLtGroupedMatrixLayoutInit_internal`." That was a **venv artifact**:
  we tested under torch 2.9, where torchao silently disables itself (`requires torch >= 2.11`).
- **Truth:** under **torch 2.13**, `torch._scaled_mm` runs FP8 natively on sm_121 and torchao's float8
  training works fine. FP8 *is* available on this box.
- **But it loses anyway:**

| config | tok/s | peak mem |
| --- | --- | --- |
| bf16 + `reference_compile` | 17,715 | 29.9 GB |
| **bf16 + full `torch.compile`** | **21,740** | 36.4 GB |
| FP8 + full `torch.compile` | 20,254 | 32.2 GB |
| FP8 eager (no compile) | 8,311 | 29.7 GB |

At 150M params the per-linear scaling/cast overhead exceeds the bandwidth saved. **Verdict: bf16 + full
compile.** Not a toolchain excuse — a measured loss.

*(Benchmarked with a manual loop, hence absolute numbers differ slightly from the Trainer-measured
25,400; the relative ordering is what matters.)*

---

## 2. Corpus (124 GB deduped → 25.79B tokens)

| Source | Size | Note |
| --- | --- | --- |
| Case law — `common-pile/caselaw_access_project` | 36.5 GB | CC0 |
| SEC 10-K filings — `eloukas/edgar-corpus` | 30.1 GB | **maxed** |
| Contracts — `pile-of-law` Atticus + `albertvillanova/legal_contracts` | 15.1 GB | **maxed** |
| **General — `HuggingFaceFW/fineweb-edu`** | **29.4 GB** | **20% general mix** |
| SEC proceedings / tax rulings — `pile-of-law` | 8.4 GB | **maxed** |
| Regulations & statutes — `pile-of-law` | 4.5 GB | **maxed** |

**Why 20% general data.** Training from scratch means no pretrained "worldview" to inherit — and real
retrieval queries arrive in plain English, not contract prose. 20% FineWeb-Edu buys ~13B general tokens
(more than GPT-2 ever saw) for ~13B domain tokens. Precedent: **BloombergGPT deliberately mixed ~50/50**
financial/general. We stay domain-heavy (80/20) because this is a retrieval model evaluated purely on
legal/financial tasks. It also *lowers* epochs (more unique data), which is a bonus.

**Public financial and contract text is genuinely exhausted** — both eval-critical domains are pulled to
their public ceilings. Rejected after inspection: `JanosAudran/financial-reports-sec` (sentence-split
10-Ks we already have); `lexlms/lex_files` (contracts split duplicates `albertvillanova`, CourtListener
overlaps CAP, remainder is UK/EU/CA/India law while every eval is US); patents (wrong genre).

### Dedup: 12.88 GB (9.4%) of duplicate text removed

Normalized near-dedup (lowercase, digits→`#`, strip punctuation — catches boilerplate differing only by
date/amount/section number). 84M unique paragraphs. **No upsampling anywhere** (`UPSAMPLE = {}`) —
regulations went from 1.5% to a real share by pulling *unique* statutory text, not by repeating CFR.

| source | duplicate text |
| --- | --- |
| **Regulations** | **36.8%** |
| **Contracts** | **30.9%** |
| EDGAR 10-Ks | 6.5% |
| General web | 3.0% |
| Case law | 1.2% |

Contracts and regulations were *far* more boilerplate-heavy than a 20k-doc sample suggested (10.8%/-),
which is exactly the failure mode dedup exists to prevent: a nominal "2.56 epochs" would otherwise be
many more passes over templates and fewer over substance.

### Rewritten in Go: 24x faster, and it caught a silent corruption bug

Dedup ran at **25 MB/s on 1 of 20 cores** in Python (~87 min). It *looks* sequential —
first-occurrence-wins needs global order — but only the cheap set lookup needs ordering; JSON parse,
paragraph split, normalize and hash are embarrassingly parallel. Workers hash; one goroutine decides
keep/drop in strict document order. **Verified byte-identical decisions to the Python version.**

| | Python | Go |
| --- | --- | --- |
| Throughput | 25 MB/s | **~604 MB/s** |
| Cores | 1.0 / 20 | **11.6 / 20** |
| 130 GB | ~87 min | **~4 min** |

> **The bug it caught.** Go's `bufio.Scanner` has a max token size, and on a longer line it **returns
> false with no error** — it does not fail, it just STOPS. `legal_regulations` contains a **168 MB single
> line** (state codes are enormous documents), so the source was truncated to **36 of 88,804 documents**
> and reported success. We would have trained on a corpus missing 99.96% of the regulatory data we had
> gone out of our way to source. Fixed with an unbounded reader + a doc-count guard that now verifies
> every source against the raw corpus.

### Two other landmines

1. **A 16.2M-character document** blew single tokenizer workers to **20 GB RSS**, thrashing a
   unified-memory box (throughput collapsed ~3,000 → ~15 docs/sec). Fix: **split** (not truncate)
   documents >50k chars — lossless, since everything is re-chunked into fixed blocks anyway.
2. **`datasets` streaming segfaults the interpreter** (`PyGILState_Release`) when an iterator is
   abandoned mid-shard — which happens every time a source hits its byte budget. Harmless in a probe,
   fatal in a 10-hour unattended build. Fix: read parquet directly with pyarrow.

---

## 3. Tokenizer (50,368 = 50,196 BPE + 156 domain + 16 reserved)

Fresh byte-level BPE trained on the **mixed** corpus. Vocab is a multiple of 64 for tensor-core
alignment (the added tokens are subtracted from the learned BPE budget to keep it exactly 50,368).

**Fertility vs ModernBERT's general BPE: 6.8%** fewer tokens (contracts **14.7%**, case law 7.7%,
EDGAR 4.9%, general web 0.3% — i.e. parity, we didn't get *worse* at general text). It was 9.0% when
the corpus was domain-only; giving 20% of the vocab budget to general text is the correct trade now
that 20% of training tokens *are* general.

### Custom tokens: added only where BPE structurally CANNOT learn

BPE's pre-tokenizer splits on punctuation, so `U.S.C.` becomes `["U",".","S",".","C","."]` and **can
never be merged, however frequent it is** (and it occurs **4.9M times**). That is a *structural* failure,
not a frequency one — and it is the entire criterion.

| term | before | after |
| --- | --- | --- |
| `Fed. R. Civ. P.` | 8 tokens | **1** |
| `U.S.C.` / `C.F.R.` | 6 tokens | **1** |
| `K.S.A.`, `Sp. Sess. P.A.` | 6–8 tokens | **1** |
| `e.g.` / `i.e.` | 4 tokens | **1** |
| `P.A.` (665k occurrences) | 4 tokens | **1** |

**Result: 38.7% fewer tokens on citation-heavy legal text**, lossless round-trip, for **0.34% of vocab**.
The 78 terms were found by *scanning the corpus* for high-frequency punctuation-internal abbreviations,
not by brainstorming.

**Deliberately NOT added:**
- **Proper nouns (cities, states, companies).** BPE already learned every one that matters — `Delaware`,
  `California`, `Texas`, `Chicago` are all 1 token because they're frequent. What fragments is
  `Poughkeepsie`, and that is *correct*: its embedding row would get ~13k gradient updates while its
  subwords each get millions. **A rare token's embedding is worse than composing from well-trained
  subwords**, and vocab is zero-sum.
- **A legal word list.** `indemnification`, `notwithstanding`, `thereunder`, `EBITDA`, `GAAP` are already
  1 token.
- **Digit-splitting** (Llama/GPT-4 style). Digits are only ~2% of characters / ~5% of tokens here, so
  splitting them would inflate total tokens ~9% — cancelling the tokenizer's entire fertility win — to
  buy numeracy, which matters far more for generation than retrieval.

> **The bug this nearly shipped.** The natural implementation (`AddedToken(lstrip=True)`) gave the best
> compression but silently **ate the preceding space**: `"Roe v. Wade"` round-tripped as `"Roev. Wade"`.
> The model could not have distinguished `Roe v.` from `Roev.` on *every citation in the corpus*. Fixed by
> adding **both bare and space-prefixed variants** (the GPT-2 approach) — full compression AND lossless.
> Caught only because the test checked round-trip, not just token counts.

`[QUERY]`, `[PASSAGE]` and 14 spare tokens are **reserved now**, so stage 2 never has to resize the
embedding matrix (which would inject random rows into a trained matrix).

---

## 3b. Packing: a flat uint16 memmap, not Arrow

Vocab 50,368 < 65,536, so **every token id fits in a uint16**. The packed corpus is therefore just a flat
51.5 GB byte array, and "packing into 1024-token blocks" is pure arithmetic —
`block[i] = mmap[i*1024 : (i+1)*1024]`. No grouping pass, no `save_to_disk`, no Arrow. (The nanoGPT
layout.) Round-tripping 124 GB through Arrow three times projected to **~7 hours**; this took ~2.

It is also **faster to read during training**: mmap + OS page cache, zero-copy. On a bandwidth-bound box
the dataloader must never become the bottleneck.

**Go was *not* used here**, deliberately: tokenization is already Rust (HF `tokenizers`) running parallel
across cores, so Go adds nothing — and a Go tokenizer binding that diverged even slightly from our 156
AddedTokens would silently corrupt a 30-day run. Go went to dedup, which was genuinely single-threaded
Python. **Use Go where the work is single-threaded Python, not where it is already Rust.**

> **The vocab bug this caught.** `PreTrainedTokenizerFast.vocab_size` returns **only the learned BPE
> (50,196)** — it EXCLUDES added tokens. Real ids reach **50,367**. The model was being built with a
> 50,196-row embedding matrix, so every domain token and `[QUERY]`/`[PASSAGE]` would have **indexed past
> the end of it**. Use `len(tokenizer)`. Surfaced only because the dataset test asserted
> `max_id < vocab_size` instead of just checking that it ran.

---

## 4. Stage 1 — MLM pretraining

| Setting | Value | Why |
| --- | --- | --- |
| Objective | MLM, **30% masking** | Beats BERT's classic 15% (ModernBERT, MosaicBERT) |
| Precision | bf16 | FP8 measured slower (§1) |
| Compile | **full-graph `torch.compile`**, `reference_compile=False` | +22.7% |
| `sparse_prediction` | **True** | +23.6% |
| Optimizer | fused AdamW, wd 1e-5, grad-clip 1.0 | |
| LR schedule | **warmup-stable-decay** (5% / 10%) | Long stable phase ⇒ mid-run checkpoints are usable |
| Peak LR | **2e-4** | Derived: ModernBERT-base used ~8e-4 at a ~4.7M-token batch; ours is 18× smaller, so `8e-4/√18 ≈ 1.9e-4` |
| Micro-batch × accum | 32 × 8 = **256 seq = 262,144 tok/step** | ~251k optimizer steps — many updates, which is what a token-limited run wants |
| `max_steps` | **251,000** | |

Verified correct, not assumed: **dynamic masking** (fresh mask each pass, not static), **global shuffle**
(`RandomSampler` reshuffles each epoch → domains interleave), **RoPE thetas** (local 10k / global 160k),
**full-state resume** (`optimizer.pt` + `scheduler.pt` + `rng_state.pth` + dataloader position).

**Eval cost.** The full val split is ~40k blocks (~42M tokens); evaluating it every 1,000 steps would
cost ~20 min per eval — **~30 hours of the run burned on evals**. Now a fixed 2,000-block subset (~15s
per eval). Full val set retained on disk for one final evaluation.

---

## 5. Stage 1b — context extension (planned)

Enterprise documents are long, but training the *whole* run at long context is the expensive way there:
at 4096 tokens throughput drops ~2.1×, roughly doubling wall clock for the same tokens seen.

**Decision: bulk of training at 1024, then a short extension phase at 8192** — ModernBERT's own recipe.
`max_position_embeddings=16384` (RoPE has no learned position table, so the ceiling is free), letting
the model extrapolate past its trained context at inference.

---

## 6. Stage 2 — Matryoshka embeddings

`SentenceTransformer` (Transformer + **mean** pooling), `MultipleNegativesRankingLoss` wrapped in
`MatryoshkaLoss`, dims **`[768, 512, 256, 128, 64]`** — derived from model width, not hardcoded.

> **Bug caught:** dims were hardcoded `[768, 512, …]` from when the model was 1024-wide, which would
> have silently discarded the top 256 dims. And naive halving (768→384→192→96) **never reaches 64** —
> the dim people actually want for cheap first-stage retrieval, and the one the docs advertised.

**Pairs mined from existing labels — no LLM-generated data:**

| Source | Pair |
| --- | --- |
| LEDGAR (`coastalcph/lex_glue`) | Two provisions sharing a clause-type label (100 classes) |
| CUAD (`theatticusproject/cuad-qa`) | Clause-type question → annotated contract span |
| FinQA (`ChanceFocus/flare-finqa`) | Financial question → evidence context |
| EDGAR-CORPUS | Section query → that filing's section |

### Planned improvements (build during the 30-day run)
- **BM25 hard negatives** — MNRL with only random in-batch negatives is the weak version; hard negatives
  are usually the single biggest retrieval-quality lever, especially where surface overlap is high and
  distinctions are subtle.
- **Label-aware batch sampler** — LEDGAR same-label pairs create *false negatives*: two genuinely similar
  clauses in one batch train the model to push them apart. Exclude same-label items from acting as
  negatives for each other.
- **Asymmetric prefixes** (`query:` / `passage:`, E5/BGE-style) — for the short-query→long-passage pairs.
  Note LEDGAR clause↔clause pairs are **symmetric**, so they need passage/passage, not query/passage.
- **GradCache** if the contrastive batch turns out memory-constrained (MNRL wants many negatives).

**Eval:** `InformationRetrievalEvaluator` at every truncation dim, reporting **nDCG@10 *and* Recall@100**
— if this feeds a RAG first stage, what matters is whether the gold passage survives into the reranker's
candidate pool. Plus an external benchmark (BEIR legal / FiQA) for comparability: our own held-out set
proves *graceful degradation*; an external one proves the model is actually good.

---

## 7. Operations

- **Everything long-running lives in `tmux`.** Jobs launched from the agent session were killed twice by
  session teardown.
- **Checkpoint series.** Hub pushes *overwrite* `main`, and `save_total_limit` prunes local checkpoints —
  so by default **no intermediate survives**. `archive_checkpoints.py` (tmux `archiver`) preserves ~27
  log-spaced snapshots to `archive/` and to Hub branches `step-N`, loadable via
  `from_pretrained(repo, revision="step-50000")`. Deliberately **decoupled from the training loop** so a
  slow/failed 600MB upload can never stall a 30-day run. Tail cadence scales with `max_steps`.
- **Stats snapshots every 6h** (`training_stats.py`): step, %, epoch, tokens seen, loss/perplexity, LR,
  grad-norm, **plateau/divergence detection**, GPU/memory/disk, process liveness. Read-only.
- **`earlyoom` is armed** (SIGKILLs at mem ≤1%, prefers python). It has killed jobs twice — once as
  collateral damage when we ran a batch-128 benchmark (~109 GB) alongside a corpus job. **Do not run
  heavy jobs alongside training.**
- A stray **vllm Docker container** was found idling and stopped.

---

## 8. Why 150M and not 400M

The initial plan was 395M. That was wrong, and the reasoning is a trap worth naming.

Chinchilla says ~20 tokens/param is compute-optimal — and 395M × 20 ≈ our budget, so 395M *looks*
right-sized. **But Chinchilla is a result for causal LMs.** MLM is far less sample-efficient (only the
~30% masked positions produce loss), and real encoders train *orders of magnitude* past Chinchilla:

| Model | Params | Tokens | tok/param |
| --- | --- | --- | --- |
| BERT-base | 110M | ~128B | ~1,160 |
| ModernBERT-base | 150M | ~2T | ~13,300 |
| 395M @ our budget | 395M | 12.7B | **~32** ❌ |
| **150M @ our budget (final)** | **150M** | **65.8B** | **~440** ✅ |

The binding constraint is **tokens, not capacity** — and under a fixed *wall-clock* budget a smaller
model runs faster, sees more data, and lands closer to convergence. A 395M model here would just be a
bigger, more undertrained artifact. It also matches reality: essentially every strong production
embedding model is 22M–150M (MiniLM 22M, bge-small 33M, gte-base 110M, ModernBERT-base 150M).

At 440 tok/param we remain ~2.6× under BERT-base and well under ModernBERT — **still undertrained by
encoder standards**, and the model card says so. But it is a 2.2× improvement over where this started.

---

## 9. Known simplifications (accepted)

- **Cross-document attention within packed blocks.** A 1024-token block can straddle two documents and
  attention isn't masked across the boundary. Standard for MLM (ModernBERT does the same); matters more
  for causal LM. Not masking it.
- **Coarse near-dedup**, not MinHash/LSH. Catches the dominant mode (boilerplate differing by
  date/amount/section number); company-name-substituted variants survive.
- **Fixed 30% mask rate** — ModernBERT anneals to 15% late; we don't.
- **Flat data mix**, no temperature sampling.
