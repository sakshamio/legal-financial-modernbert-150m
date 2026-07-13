# Diagnostics: making a training run legible

> Snapshot: **step 2,000 / 271,000 (0.7%)**. Numbers will move; the *method* is the point.

A loss of `4.70` is not information. It becomes information only once you know what it is being compared
to, and once you can see what the model actually does.

Two scripts:

| | |
| --- | --- |
| `scripts/analyze_loss.py` | Reference points that turn the loss into a claim, per-domain breakdown, qualitative mask-fills |
| `scripts/diagnostics.py` | Embedding geometry, token-frequency breakdown, domain-token probe, contextual neighbours, `--trend` |

---

## 1. What does the loss actually mean?

Compute the reference points from the corpus itself:

| | loss | perplexity | meaning |
| --- | --- | --- | --- |
| Uniform over vocab | **10.83** = ln(50,368) | 50,368 | what random init produces |
| **Unigram entropy** | **7.42** | 1,674 | **THE BAR** — predicting token *frequencies* alone |
| **Us @ step 2,000** | **4.70** | **110** | |
| Well-trained encoder | ~1.5–2.0 | ~5–7 | target |

**The unigram entropy is the number that matters.** It is the loss you get by knowing nothing except
which tokens are common — that "the" is frequent and "notwithstanding" is not. A model sitting at 7.42
has learned *statistics*. A model below it is **using context**.

At 4.70 we are decisively below it: the model reads context. It is now choosing between ~110 tokens per
masked position, down from 50,368.

*(Computed from the packed corpus, not looked up. `analyze_loss.py --skip-unigram` to skip.)*

## 2. Per-domain loss — a free sanity check

| domain | loss | top-1 |
| --- | --- | --- |
| **Contracts** | **3.83** | 44.4% |
| Regulations | 4.34 | 39.5% |
| EDGAR 10-Ks | 4.41 | 37.8% |
| Case law | 4.59 | 35.8% |
| SEC / tax rulings | 5.43 | 32.1% |
| **General web** | **5.64** | 28.7% |

The *ordering* is the check. Contracts are the most **formulaic** ("shall indemnify and hold harmless"),
so they should be the most predictable. General web is the most **diverse**, so it should be hardest. It
is. **If general web had come out easiest, something would be badly wrong** — and we would want to know
that before spending 30 days.

## 3. Embedding geometry — the diagnostic that decides whether this project works

MLM loss tells you the model is learning. It says **nothing** about whether the representation will be
usable for retrieval, which is what we are actually building.

```
mean pairwise cosine of random passages : +0.657   [anisotropic]
effective rank                          : 17.2 / 768
dims to explain 90% / 95% / 99% of var  : 35 / 62 / 105
```

Transformer representations notoriously collapse into a narrow cone — every embedding ends up similar to
every other. A model can have an excellent MLM loss and a **useless** embedding space.

Worse for us specifically: **if the representation only really uses ~17 of its 768 dimensions, Matryoshka
truncation is meaningless.** There is nothing to nest — the dimensions are not carrying independent
information.

At 0.7% trained, low rank and high anisotropy are **normal**: early representations are dominated by a
few frequency and position directions. But this is the metric to watch, and it is the **earliest
actionable warning we can get** — weeks before any retrieval eval could tell us.

**A flat effective rank by ~step 50,000 would mean the representation is collapsing and Matryoshka will
not work.** `diagnostics.py --trend` tracks it.

> The related number — *"7.9% of variance in the first 64 coordinates"* — is **not** a problem. That is
> the pre-MRL baseline. Stage 2's Matryoshka loss exists precisely to redistribute variance into the
> early coordinates. This is the "before" picture.

## 4. Loss by token frequency — learning language, or learning statistics?

| frequency bucket | loss | top-1 |
| --- | --- | --- |
| top 100 tokens | **2.10** | 58.8% |
| 100–1k | 5.25 | 19.3% |
| 1k–10k | 7.93 | 9.2% |
| **10k+ (rare)** | **10.61** | 4.6% |

A clean monotonic difficulty gradient — rare tokens *should* be much harder.

The rare bucket at **10.61 ≈ ln(50,368) = 10.83** is the honest headline: on rare tokens the model is
**still essentially guessing**. That is correct at 0.7% trained, and it is where the remaining learning
lives. Aggregate loss is flattered by frequent tokens; this bucket is not.

## 5. Are the 156 custom vocab tokens being learned?

We spent 156 vocab slots on citation machinery BPE structurally *cannot* learn (`U.S.C.`, `C.F.R.`,
`10-K`). Did we waste them?

Random block sampling **cannot** answer this — these tokens occur ~7 times per **million**, so a few
dozen blocks contain roughly zero of them (our first attempt returned `n=1`). So we **construct** the
examples: mask the domain token inside a real sentence and ask the model to recover it.

Baseline at step 2,000 — **not learned yet**, correct token ranked 4,000–48,000 of 50,368:

```
   10-Q     rank 18,026/50,368   top-3: ['filed', '10-K', 'basis']
   e.g.     rank  4,338/50,368   top-3: ['Inc.', 'assets', 'notes']
   U.S.C.   rank 40,444/50,368   top-3: ['.', '§', '']
```

But look at `10-Q`: **`10-K` is already in its top-3.** The model has learned that *"filed its quarterly
report on Form ___"* takes an SEC form — it just picks the wrong one. **The concept forms before the
token.** That is the trajectory we want, and it is a far more informative signal than the accuracy number.

## 6. Contextual neighbours — is meaning forming?

| term | nearest neighbours |
| --- | --- |
| **plaintiff** | **defendant (0.69)**, damages (0.67), negligence (0.60) |
| **defendant** | damages (0.71), **plaintiff (0.69)**, warranty (0.67) |
| **damages** | defendant (0.71), warranty (0.67), plaintiff (0.67) |
| revenue | liability (0.53), indemnification (0.52) |

**A litigation cluster has already emerged** — `plaintiff` and `defendant` are each other's top
neighbour, pulling in `damages` and `negligence`. At 0.7% of training.

But be honest about the margins: **mean pairwise cosine is 0.657**, so `plaintiff↔defendant` at 0.69 is
only *slightly* above the similarity of two random passages. The structure is real but **thin** — which
is exactly what anisotropy does: it compresses the dynamic range so everything resembles everything.
`revenue → liability, indemnification` confirms it: finance and law are **not yet separated**.

The geometry finding (§3) and this one are the same story from two angles.

---

## Two of these diagnostics were WRONG, and the data exposed them

Worth recording, because a diagnostic you cannot falsify is worse than none — you end up confidently
reading noise.

**1. "Loss by position" had a causal-LM premise.**
I expected loss to *fall* later in the sequence as left-context accumulates. That is how a **causal** LM
behaves. **MLM is bidirectional** — a token at position 10 already sees the entire block. The flat curve
(4.91 / 5.07 / 5.33 / 5.04) is not a failure; it is **confirmation that bidirectional attention is
working**. A *rise at the edges* would have signalled boundary artifacts. Wrong premise, right data.

**2. "Semantic neighbours" probed the wrong matrix.**
The first version looked at the **static input-embedding** table and returned pure noise
(`indemnification → amphetamine`). But meaning in a transformer lives in the **contextual**
representation, not the lookup table — and at 0.7% the input matrix is barely trained anyway. Switching
to contextual embeddings made real structure appear immediately (§6).

---

## What to watch

```bash
python scripts/diagnostics.py --trend
```

```
    step  eff.rank  mean cos  rare loss  dom acc  90% var
   2,000      17.1    +0.663      10.59      0%       32

  WANT:   eff.rank UP   mean-cos DOWN   rare-loss DOWN   dom-acc UP
```

| metric | want | why |
| --- | --- | --- |
| **effective rank** | **UP** | 17/768 at step 2k is fine. 17/768 at step 50k means Matryoshka is dead. |
| **mean cosine** | **DOWN** | anisotropy shrinking = embeddings spreading out = retrieval possible |
| **rare-token loss** | **DOWN** | where real language learning shows up; aggregate loss hides it |
| **domain-token acc** | **UP** | proof the 156 custom vocab slots were worth it |

Both scripts run on **CPU** by default. On this unified-memory box, CPU work costs a bandwidth-bound GPU
run ~2% — and a GPU eval would cost far more. A few hundred CPU forward passes is cheap.
