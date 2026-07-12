# legal-financial-modernbert-150m

Training a **from-scratch** ModernBERT encoder (random init — no pretrained checkpoint, no distillation)
on public **legal and financial** documents, then turning it into a **Matryoshka** embedding model whose
vectors truncate to smaller dimensions without retraining.

All on **one NVIDIA DGX Spark (GB10)**.

- 🤗 **Model + checkpoint series:** [sakshamio/legal-financial-modernbert-150m](https://huggingface.co/sakshamio/legal-financial-modernbert-150m)
- 📋 **Full technical plan & design decisions:** [`PLAN.md`](PLAN.md)

> **Status:** stage-1 pretraining is running (271,000 steps, ~30 days). The weights on the Hub are early
> checkpoints and are **not usable yet**. No quality claims are made.

---

## Configuration

| | |
| --- | --- |
| Model | 149.7M params, ModernBERT-base geometry, **random init** |
| Corpus | **124 GB deduplicated → 25.79B tokens** (80% legal/financial · 20% FineWeb-Edu) |
| Vocab | **50,368** = 50,196 BPE + 156 domain tokens + 16 reserved |
| Training | **271,000 steps → 71B tokens → 2.76 epochs → 475 tokens/param** |
| Throughput | **~28,400 tok/s** (bf16 + `sparse_prediction` + full `torch.compile`) |

---

## Pipeline

```bash
# 1. Corpus  (~115GB legal/financial + 29GB FineWeb-Edu)
python scripts/prepare_corpus.py                 # --only <source> to add one source

# 2. Tokenizer  (byte-level BPE + 78 domain tokens, vocab 50,368)
python scripts/train_tokenizer.py --vocab-size 50368 --sample-gb 4

# 3. Dedup  (Go: ~604 MB/s, 24x faster than the Python reference)
cd godedup && go build -o godedup . && ./godedup ..

# 4. Pack  (flat uint16 memmap -- no Arrow)
python scripts/pack_fast.py --num-proc 18 --block-size 1024

# 5. Stage 1: MLM pretraining
python scripts/pretrain_mlm.py --max-steps 271000 --push-to-hub

# 6. Stage 2: contrastive + Matryoshka  (planned)
python scripts/build_pairs.py
python scripts/train_matryoshka.py
python scripts/eval_retrieval.py
```

Long-running steps go in `tmux`. `scripts/run_stats_daemon.sh` snapshots training health every 6h;
`scripts/run_archiver_daemon.sh` preserves 28 checkpoints as `step-N` branches on the Hub.

---

## Four things worth knowing

### 1. `nvidia-smi` lies about utilization

It reported **96–100%** the entire run. Real **Model FLOP Utilization: 9–19%**. That number only means
*a kernel is resident* — the math units stall ~85% of cycles on a **273 GB/s** memory pipe.

**The GPU is "fully utilized" and simultaneously doing almost nothing.** So only optimizations that
**move fewer bytes** help; anything that adds work does not:

| Change | Result |
| --- | --- |
| `sparse_prediction=True` | **+23.6%**, −32% memory |
| Full-graph `torch.compile` | **+22.7%** |
| Micro-batch 32 → 64 (2× memory) | **−0.3%** |
| Micro-batch 128 | **OOM — killed the machine** |

Net **+70%** throughput, entirely from moving fewer bytes.

**Corollary: CPU work is not free either.** Running a CPU-only benchmark alongside training slowed it
from **9.17 → 9.35 s/it (2%)** — ~14 hours over a 30-day run — and it snapped back the instant the CPU
job was killed. Unified memory means the CPU competes with the GPU for the *same* 273 GB/s. On a
discrete-GPU box you could treat CPU as free; here you cannot. Heavy CPU jobs (benchmarks, dedup,
packing) get scheduled around training, not alongside it.


### 2. FP8 works on this chip — and is still slower than bf16

We concluded twice that FP8 was impossible (*"no aarch64 wheels; `transformer_engine` dies with
`undefined symbol: cublasLtGroupedMatrixLayoutInit_internal`"*). That was a **venv artifact** — torch 2.9
makes torchao silently disable itself. Under **torch 2.13**, `torch._scaled_mm` runs FP8 natively on
sm_121.

FP8 is available. It just **loses**: 20,254 tok/s vs **21,740** for bf16 + full compile. The per-linear
scaling/cast overhead exceeds the bandwidth it saves at 150M params.

### 3. The Chinchilla trap: why 150M beats 400M here

This started aimed at 395M. Chinchilla's ~20 tokens/param made that look right-sized — **but Chinchilla
is a causal-LM result.** MLM is far less sample-efficient (only ~30% of positions produce loss), and real
encoders train orders of magnitude past it:

| Model | Params | Tokens | tok/param |
| --- | --- | --- | --- |
| BERT-base | 110M | ~128B | ~1,160 |
| ModernBERT-base | 150M | ~2T | ~13,300 |
| 395M @ our budget | 395M | 12.7B | **~32** ❌ |
| **150M @ our budget** | **150M** | **71B** | **~475** ✅ |

The binding constraint is **tokens, not capacity**. Under a fixed *wall-clock* budget a smaller model
runs faster, sees more data, and lands closer to convergence. **Fewer parameters, better model.**

### 4. Use Go where the work is single-threaded Python — not where it's already Rust

**Dedup → Go.** It was pure single-threaded Python at 25 MB/s on 1 of 20 cores. It *looks* sequential
(first-occurrence-wins needs global order), but only the cheap set lookup needs ordering — parse, split,
normalize and hash are embarrassingly parallel. Workers hash; one goroutine decides in strict document
order. **~604 MB/s, 24× faster, verified byte-identical to the Python reference.**

**Packing → not Go.** Tokenization is already Rust (HF `tokenizers`) running parallel across cores, so Go
adds nothing — and a Go tokenizer binding that diverged even slightly from our 156 AddedTokens would
silently corrupt a 30-day run. The real win there was dropping **Arrow**: vocab < 65,536 means token ids
fit in **uint16**, so the corpus is a flat 51.5 GB array and "packing into blocks" is pure arithmetic.
~7h → ~2h, and faster to read during training (mmap, zero-copy).

---

## Three bugs that would have shipped silently

Each was caught by **asserting**, not eyeballing.

**1. A 168 MB line truncated the corpus — and reported success.**
Go's `bufio.Scanner` has a max token size, and on a longer line it *returns false with no error*. The
state codes contain a 168 MB document, so `legal_regulations` was silently cut to **36 of 88,804
documents**. We'd have trained on a corpus missing 99.96% of the regulatory data we went out of our way
to source. → unbounded reader + a doc-count guard against the raw corpus.

**2. `tokenizer.vocab_size` excludes added tokens.**
It returns only the learned BPE (**50,196**); real ids reach **50,367**. The model was being built with an
embedding matrix **172 rows too small** — every domain token and `[QUERY]`/`[PASSAGE]` would have indexed
past the end of it. → use `len(tokenizer)`. Caught by asserting `max_id < vocab_size`.

**3. `AddedToken(lstrip=True)` ate spaces.**
It gave the best compression but silently turned `"Roe v. Wade"` into `"Roev. Wade"` on round-trip. The
model could not have distinguished `Roe v.` from `Roev.` on *every citation in the corpus*. → add **both
bare and space-prefixed variants** (the GPT-2 approach): full compression *and* lossless. Caught by
asserting round-trip equality, not just token counts.

---

## Custom tokens: the criterion

Add a token **only if it is (a) high-frequency AND (b) structurally unlearnable by BPE.**

BPE's pre-tokenizer splits on punctuation, so `U.S.C.` becomes `["U",".","S",".","C","."]` and **can
never be merged, however frequent** — and it occurs **4.9 million times**. That's a *structural* failure.

| term | before | after |
| --- | --- | --- |
| `Fed. R. Civ. P.` | 8 tokens | **1** |
| `U.S.C.` / `C.F.R.` | 6 tokens | **1** |
| `e.g.` / `i.e.` | 4 tokens | **1** |

**38.7% fewer tokens on citation-heavy legal text**, for 0.34% of vocab. The 78 terms were found by
*scanning the corpus*, not brainstorming.

**Rejected:** proper nouns (BPE already learned `Delaware`, `California`, `Texas` — what fragments is
`Poughkeepsie`, and that's *correct*: a rare token's embedding gets ~13k updates while its subwords get
millions); a legal word list (`indemnification`, `EBITDA` are already 1 token); and **digit-splitting**
(digits are only ~2% of characters — splitting would inflate tokens ~9%, cancelling the entire fertility
win, to buy numeracy that matters more for generation than retrieval).

---

## Layout

```
scripts/
  prepare_corpus.py       corpus from HF (parquet direct -- datasets streaming segfaults on abandon)
  train_tokenizer.py      byte-level BPE + 78 domain tokens + reserved [QUERY]/[PASSAGE]
  pack_fast.py            flat uint16 memmap (nanoGPT layout), no Arrow
  pretrain_mlm.py         stage 1: MLM from random init; PackedDataset reads the memmap
  build_pairs.py          stage 2: contrastive pairs mined from labels (LEDGAR/CUAD/FinQA/EDGAR)
  train_matryoshka.py     stage 2: MNRL inside MatryoshkaLoss
  eval_retrieval.py       nDCG@10 + Recall@100 at every truncation dim
  training_stats.py       6h health snapshots (plateau/divergence detection)
  archive_checkpoints.py  preserve 28 checkpoints as HF step-N branches
  dedup_corpus.py         Python dedup -- the reference the Go version was verified against
  bench_*.py              throughput probes (sparse_prediction, batch, FP8)
godedup/main.go           parallel dedup, 24x faster
```

## License

Code: MIT. **Model: CC-BY-NC-SA-4.0** (non-commercial) — the corpus includes Pile of Law subsets and that
term propagates.
