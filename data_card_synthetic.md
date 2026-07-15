---
license: apache-2.0
task_categories:
- sentence-similarity
- feature-extraction
language:
- en
tags:
- legal
- finance
- retrieval
- embeddings
- synthetic
- contrastive-learning
- matryoshka
pretty_name: Financial & Legal Synthetic Retrieval Pairs
size_categories:
- 100K<n<1M
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
---

# Financial & Legal Synthetic Retrieval Pairs

Synthetic **(query → passage)** training pairs spanning the financial/legal document world —
NDAs, investment management agreements, credit facilities, M&A agreements, fintech/SaaS contracts,
derivatives, securities, real estate finance, and regulatory filings. Built to train **asymmetric
retrieval** embedding models (a short natural-language query must retrieve the long passage that
answers it), which is where domain models trained on symmetric clause-matching data tend to fail.

Each example is a triple: a **query**, its **positive** passage, and a **hard negative** — a
similar-looking passage (a neighbouring clause or related document type) that does *not* answer the
query.

## Why this exists

Real legal/financial corpora (SEC filings, case law, statute dumps) are narrow and skewed: they
barely contain the *contract* types that matter for enterprise retrieval, and they contain almost no
natural-language **queries**. An encoder trained only on such data learns to match passages to
passages, not questions to answers. This dataset synthesises both sides across a broad taxonomy, the
approach behind E5-mistral, Gecko, and Qwen3-Embedding.

## How it was generated

- **Generator:** `Qwen/Qwen3.6-35B-A3B-FP8` (a 35B mixture-of-experts, ~3B active parameters),
  served locally via vLLM.
- **Taxonomy-driven prompting:** each generation is conditioned on an independently sampled cell of
  **10 categories × 75 document types × 106 clause topics**, further crossed with sampled
  *sector*, *governing law*, *principal party*, and *deal context* — **~20.7M distinct prompt cells**,
  so cells effectively never repeat. Temperature 0.9 adds within-cell diversity.
- **One call → one cluster:** each call returns a JSON `{passage, queries[3], hard_negative}`, giving
  three training triples per generated passage. The three queries vary in granularity (specific
  factual / topical keyword / practitioner-phrased) following the E5-mistral asymmetric task taxonomy.

### Taxonomy (categories)

`M&A` · `Investment Management` · `Lending & Credit` · `NDAs & Confidentiality` ·
`Fintech & Technology` · `Corporate & Securities` · `Derivatives & Structured` ·
`Real Estate & Project Finance` · `Regulatory & Compliance` · `Asset Management Ops`

## Format

JSON Lines. Each row:

| field | meaning |
| --- | --- |
| `anchor` | the query, prefixed `[QUERY] ` |
| `positive` | the passage that answers it, prefixed `[PASSAGE] ` |
| `negative_0` | an LLM-generated hard negative, prefixed `[PASSAGE] ` (may be absent) |
| `label` | a per-passage id; a passage's own queries are its only positives |
| `source` | `synth_<category>` |
| `meta` | `{category, subtype, clause}` |

The `[QUERY]` / `[PASSAGE]` prefixes make the asymmetry explicit for prefix-aware encoders; strip them
if your model does not use them.

## Intended use

Contrastive / Matryoshka training of sentence-embedding models for legal & financial retrieval.
Pairs are designed to drop into `sentence-transformers` (`MultipleNegativesRankingLoss`,
`CachedMultipleNegativesRankingLoss`, `MatryoshkaLoss`) with in-batch + hard negatives.

## Limitations & responsible use

- **Synthetic.** Passages are model-generated and *plausible*, not authentic legal instruments. They
  may contain factual, legal, or numerical inaccuracies and stylistic tells. **Not legal or financial
  advice; not a substitute for real documents.**
- Reflects the generator's knowledge and biases. Jurisdiction/sector labels are prompt conditioning,
  not guarantees of accuracy.
- Intended as **training signal for retrieval geometry**, where approximate realism and hard
  negatives matter more than authoritative correctness.

## Provenance

Generated with Qwen3.6-35B-A3B (Apache-2.0). Released under Apache-2.0. Part of an open
from-scratch legal/financial ModernBERT embedding project:
<https://github.com/sakshamio/legal-financial-modernbert-150m>.
