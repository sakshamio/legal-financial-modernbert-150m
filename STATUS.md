# Training status — 🧊 cooling down

`███░░░░░░░░░░░░░░░░░░░░░░░░░` **11.07%**

| | |
| --- | --- |
| **Step** | **30,001** / 271,000 |
| **Tokens** | 7.86B / 71B |
| **Speed** | 9.27 s/step → **28,279 tok/s** |
| **Elapsed** (this run) | 17:28:08 |
| **ETA** | **30d 20h** |
| ↳ compute | 25d 20h |
| ↳ cooldowns | 5d 0h (120× 1h rest) |

## Recipe

Since **step 14,000** this run uses a different recipe than steps 0–14,000
([why](https://github.com/sakshamio/legal-financial-modernbert-150m/blob/main/checkpoints/mlm_stage1/RECIPE_CHANGES.md)).

| | |
| --- | --- |
| Optimizer | **Muon 1e-4 + AdamW 2e-4** |
| Masking | **span** @ p=0.3 |
| Weight decay | 0.05 |
| Batch | 32 × 8 accum × 1024 tok |
| Seed | 1 |

## Loss

**2.479** · eval 1.757 · perplexity **12**

`██████████████████████░░`

`█▅▆█▅▄▅▄▇█▁██▄▄▅▇▁▅▄▂▁██▆▄▄▄▅▁▂▇` <sub>train loss, this run</sub>
↓ 0.0519 over 120 logged points this run

> ⚠️ **Span masking runs ~1.2 higher loss than random masking at equal model quality** — it must recover whole spans from context. Not comparable to the ≈1.62 the previous recipe ended at, and the reference band below is calibrated for random masking.

| reference | loss | |
| --- | --- | --- |
| random init `ln(50,368)` | 10.83 | start |
| **unigram** (frequencies only) | **7.42** | ✅ **beaten** — the model is using *context* |
| **current** | **2.48** | |
| well-trained encoder | 1.5–2.0 | target *(random-mask scale)* |

LR **2.00e-04** / 2.00e-04 peak
Grad norm **0.96** · `▃▂▁▂▂▁▂▁▂▃▄▃▂▃▃▂▂▃▃▂▄▃▂▂▂█▃▄▃▃▃▃` <sub>stability after the optimizer switch</sub>
Eval loss `█▂▁` (3 evals this run)

## Checkpoints

| `checkpoint-26000` | ✅ archived |
| `checkpoint-28000` | ✅ archived |
| `checkpoint-30000` | ⏳ pending |

<sub>3 local (`save_total_limit=3`) · all of step-2000..14000 preserved in `archive/` and on
[HF step-N branches](https://huggingface.co/sakshamio/legal-financial-modernbert-150m/branches)</sub>

## Machine

| | |
| --- | --- |
| GPU | 0 %, 13.57 W, 61 |
| Memory | 75GB free |
| Disk | 2.0T free |
| Cooldown | paused after checkpoint-30000 |
| Errors (this run) | 0 |

<sub>Updated 2026-07-23 01:04 UTC · [README](https://github.com/sakshamio/legal-financial-modernbert-150m) · [DIAGNOSTICS](https://github.com/sakshamio/legal-financial-modernbert-150m/blob/main/DIAGNOSTICS.md)</sub>
