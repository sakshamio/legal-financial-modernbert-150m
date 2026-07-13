"""Stage 2.5 -- distil an embedding teacher's GEOMETRY into the from-scratch encoder.

WHY GEOMETRY AND NOT TOKENS
---------------------------
The obvious idea -- distil a big causal LLM's next-token logits into a small model, then convert
that into an embedder -- is two lossy steps where one will do, and it is blocked outright by a
vocabulary mismatch: logit distillation requires teacher and student to share a tokenizer, and ours
deliberately does not (we spent 156 slots on `U.S.C.`, `C.F.R.`, `10-K`).

So we never compare tokens. For a batch of N texts we compare the teacher's N x N cosine-similarity
matrix with the student's. That is:

  * tokenizer-agnostic -- only pooled sentence vectors are ever compared, so the legal tokenizer
    survives untouched;
  * dimension-agnostic -- teacher is 1024-d, student is 768-d, and no projection is needed because
    both similarity matrices are N x N;
  * pair-free -- it needs raw text, not mined (query, positive, negative) triples. We have 118GB of
    exactly the right raw text.

WHY THIS FIXES OUR ACTUAL PROBLEM
---------------------------------
DIAGNOSTICS.md, step 2,000:  effective rank 17.2 / 768,  mean pairwise cosine +0.657.
The representation is collapsed into a narrow cone. If it stays there, THERE IS NOTHING TO NEST and
Matryoshka is meaningless. Contrastive training in stage 2 only *hopes* to discover a high-rank
isotropic geometry. Distillation instead regresses directly onto a geometry that already is one.

MATRYOSHKA: WHY THE TEACHER IS **NOT** TRUNCATED
------------------------------------------------
At each nested dim d we match student-at-d against the teacher's FULL 1024-d geometry, not against
teacher-at-d. Truncating the teacher would hand the student's first 64 dims a *degraded* target and
cap them at the teacher's own 64-d quality. Matching the full geometry sets an unreachable target
whose best compromise is exactly "the best d-dimensional approximation of the true geometry" -- which
is precisely what MRL wants, and what the original MRL paper does (each nested dim gets the full task
loss, never a weakened one). `--truncate-teacher` is left in as an ablation, off by default.

USAGE
-----
    # phase 1 (GPU): embed the sampled chunks once with the teacher, cache to disk. One-shot.
    python scripts/distill_geometry.py --phase teacher

    # phase 2 (GPU): train the student against the cached vectors. Reusable forever.
    python scripts/distill_geometry.py --phase student --init checkpoints/mlm_stage1/checkpoint-XXXX

The teacher pass is the expensive half and it is CACHEABLE: pay once, then student training is as
cheap as any 150M-model training. Both phases contend hard for the GPU -- they refuse to start while
stage-1 pretraining is alive unless you pass --force.
"""
import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parent.parent
DISTILL_DIR = Path(os.environ.get("DISTILL_DIR", PROJECT_DIR / "data" / "distill"))
TEACHER = "Qwen/Qwen3-Embedding-0.6B"  # Apache-2.0, 1024-d, MRL-trained, ~64 MTEB
TEACHER_DIM = 1024
MRL_DIMS = [768, 512, 256, 128, 64]


def guard_gpu(force):
    alive = subprocess.run("pgrep -f '[p]retrain_mlm.py'", shell=True, capture_output=True).stdout
    if alive and not force:
        raise SystemExit(
            "stage-1 pretraining is RUNNING. Teacher/student passes are real GPU work and will\n"
            "contend badly (unlike the CPU diagnostics, which cost ~2%). Wait for the GPU, or pass\n"
            "--force if you know what you are doing."
        )


def load_chunks(split, limit=None):
    p = DISTILL_DIR / f"{split}.jsonl"
    if not p.exists():
        raise SystemExit(f"{p} missing -- run scripts/sample_distill_corpus.py first")
    rows = []
    with open(p) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            rows.append(json.loads(line))
    return rows


# ======================================================================================
# PHASE 1 -- teacher. Expensive, one-shot, cached.
# ======================================================================================
def phase_teacher(args):
    from sentence_transformers import SentenceTransformer

    for split in ("train", "val"):
        rows = load_chunks(split, args.limit)
        texts = [r["text"] for r in rows]
        out = DISTILL_DIR / f"teacher_{split}.f16"
        prog = DISTILL_DIR / f"teacher_{split}.progress"

        done = int(prog.read_text()) if prog.exists() else 0
        if done >= len(texts):
            print(f"{split}: already cached ({done:,})")
            continue

        # fp16 memmap: 5M x 1024 x 2B = 10GB. Cheap to store, reusable forever.
        mm = np.memmap(out, dtype=np.float16, mode="r+" if out.exists() else "w+",
                       shape=(len(texts), TEACHER_DIM))

        print(f"loading teacher {TEACHER} ...")
        model = SentenceTransformer(TEACHER, model_kwargs={"dtype": torch.bfloat16},
                                    device=args.device)
        model.max_seq_length = args.max_seq
        # NOTE: no instruction prefix. Qwen3-Embedding takes an instruction for QUERIES only;
        # these are all documents/passages, which the teacher expects bare.

        t0 = time.time()
        for s in range(done, len(texts), args.batch_size):
            batch = texts[s : s + args.batch_size]
            v = model.encode(batch, batch_size=len(batch), normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False)
            mm[s : s + len(batch)] = v.astype(np.float16)
            n = s + len(batch)
            prog.write_text(str(n))
            if (s // args.batch_size) % 20 == 0:
                el = time.time() - t0
                rate = (n - done) / max(el, 1e-9)
                eta = (len(texts) - n) / max(rate, 1e-9) / 3600
                print(f"  {split} {n:,}/{len(texts):,}  {rate:.0f} chunk/s  ETA {eta:.1f}h", flush=True)
        mm.flush()
        print(f"{split}: cached {len(texts):,} teacher vectors -> {out}")


# ======================================================================================
# PHASE 2 -- student. Cheap, repeatable.
# ======================================================================================
def relational_loss(se, te, tau, w_kl, w_mse):
    """Match the teacher's similarity STRUCTURE over the batch, not its vectors.

    se: [N, d_s] student (any d)   te: [N, d_t] teacher (any d)  -- dims need not agree.
    The diagonal is masked out: self-similarity is always 1 and carries no signal, but at tau=0.05
    it would otherwise dominate every softmax row.
    """
    se = F.normalize(se.float(), dim=-1)
    te = F.normalize(te.float(), dim=-1)
    Ss, St = se @ se.T, te @ te.T

    n = Ss.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=Ss.device)
    # Scale by tau FIRST, then mask with a finite value. Masking with finfo.min and dividing
    # afterwards overflows (-3.4e38 / 0.05 = -inf), which makes the forward KL nan. Gradients happen
    # to survive that (d kl_div/d input = -target, and target is 0 on the diagonal), so the model
    # still trains -- but the loss is then unreadable, which is how it went unnoticed. -1e4 underflows
    # exp() to exactly 0 with no inf anywhere.
    kl = F.kl_div(
        F.log_softmax((Ss / tau).masked_fill(eye, -1e4), dim=-1),
        F.softmax((St / tau).masked_fill(eye, -1e4), dim=-1),
        reduction="batchmean",
    )
    # KL is scale-invariant and only fixes the RANKING. The MSE term pins the ABSOLUTE similarity
    # scale, which is what a retrieval threshold actually keys off -- and it is what pulls the mean
    # pairwise cosine down out of the cone.
    mse = F.mse_loss(Ss[~eye], St[~eye])
    return w_kl * kl + w_mse * mse, kl.detach(), mse.detach()


def geometry(emb):
    """Effective rank + anisotropy, IDENTICAL in definition to diagnostics.py.

    The spectrum is normalised over s**2 -- the eigenvalues of the covariance -- not over s. This is
    not cosmetic: squaring sharply peaks the spectrum, and the same embeddings score 17 by one
    definition and 272 by the other. DIAGNOSTICS.md publishes 17.2/768 and --trend tracks it, so we
    must use its definition or nothing here is comparable to anything there.
    """
    x = emb.float()
    x = x - x.mean(0, keepdim=True)
    var = torch.linalg.svdvals(x) ** 2
    p = var / var.sum().clamp_min(1e-12)
    p = p[p > 0]
    eff_rank = float(torch.exp(-(p * p.log()).sum()))
    xn = F.normalize(emb.float(), dim=-1)
    S = xn @ xn.T
    n = S.size(0)
    mean_cos = float(S[~torch.eye(n, dtype=torch.bool, device=S.device)].mean())
    return eff_rank, mean_cos


def spearman(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra @ rb) / (ra.norm() * rb.norm()).clamp_min(1e-12))


class ChunkDS(torch.utils.data.Dataset):
    def __init__(self, texts, vec_path, n):
        self.texts, self.vec_path, self.n, self._v = texts, str(vec_path), n, None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self._v is None:  # lazy per worker
            self._v = np.memmap(self.vec_path, dtype=np.float16, mode="r").reshape(-1, TEACHER_DIM)
        return self.texts[i], torch.from_numpy(np.asarray(self._v[i]).astype(np.float32))


@torch.no_grad()
def evaluate(student, val_texts, val_vecs, dims, device, bs):
    student.eval()
    se = torch.cat([student.encode(val_texts[i : i + bs], convert_to_tensor=True,
                                   normalize_embeddings=False, show_progress_bar=False).cpu()
                    for i in range(0, len(val_texts), bs)])
    te = torch.from_numpy(np.asarray(val_vecs)).float()
    tn = F.normalize(te, dim=-1)
    St = (tn @ tn.T)
    n = St.size(0)
    off = ~torch.eye(n, dtype=torch.bool)

    rows = {}
    for d in dims:
        sd = F.normalize(se[:, :d], dim=-1)
        Ss = sd @ sd.T
        er, mc = geometry(se[:, :d])
        rows[d] = {
            "spearman_vs_teacher": spearman(Ss[off], St[off]),
            "eff_rank": er,
            "mean_cos": mc,
        }
    et, tc = geometry(te)
    student.train()
    return rows, {"eff_rank": et, "mean_cos": tc}


def phase_student(args):
    from sentence_transformers import SentenceTransformer, models

    train_rows = load_chunks("train", args.limit)
    val_rows = load_chunks("val", args.val_limit)
    tv = np.memmap(DISTILL_DIR / "teacher_train.f16", dtype=np.float16, mode="r").reshape(-1, TEACHER_DIM)
    vv = np.memmap(DISTILL_DIR / "teacher_val.f16", dtype=np.float16, mode="r").reshape(-1, TEACHER_DIM)

    # The memmap is preallocated to full size, but the teacher fills it INCREMENTALLY. Rows past the
    # progress pointer are still zeros -- training on them would silently distil garbage and nothing
    # downstream would flag it. Hard-cap every read at what the teacher has actually written.
    #
    # This cap is also the feature that makes the scaling ladder free: the cache is a PREFIX, so the
    # 5M run is literally the first 5M rows of the same file the 10M and 40M runs use. One teacher
    # pass, N rungs, zero recomputation.
    done = int((DISTILL_DIR / "teacher_train.progress").read_text())
    n_train = min(len(train_rows), tv.shape[0], done)
    if args.limit:
        n_train = min(n_train, args.limit)
    if args.limit and args.limit > done:
        raise SystemExit(f"--limit {args.limit:,} but teacher has only embedded {done:,} rows")
    if n_train == 0:
        raise SystemExit("teacher cache is empty -- run --phase teacher first")

    n_val = min(len(val_rows), int((DISTILL_DIR / "teacher_val.progress").read_text()))
    val_texts = [r["text"] for r in val_rows][:n_val]
    val_vecs = vv[:n_val]

    word = models.Transformer(args.init, max_seq_length=args.max_seq)
    # Our tokenizer_config advertises token_type_ids, but ModernBertModel.forward() does not accept
    # it. Fix it at the tokenizer so every downstream path (encode(), tokenize()) is covered.
    word.tokenizer.model_input_names = ["input_ids", "attention_mask"]
    pool = models.Pooling(word.get_word_embedding_dimension(), "mean")
    student = SentenceTransformer(modules=[word, pool], device=args.device)
    student.train()

    ds = ChunkDS([r["text"] for r in train_rows], DISTILL_DIR / "teacher_train.f16", n_train)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                     num_workers=args.workers, pin_memory=True,
                                     collate_fn=lambda b: ([x[0] for x in b], torch.stack([x[1] for x in b])))

    steps = args.max_steps if args.max_steps > 0 else len(dl) * args.epochs
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    warm = int(steps * 0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    dims = MRL_DIMS
    weights = [1.0] * len(dims)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"student {sum(p.numel() for p in student.parameters())/1e6:.0f}M  "
          f"train {n_train:,}  steps {steps:,}  batch {args.batch_size}  dims {dims}")

    # Baseline BEFORE any distillation -- this is the number DIAGNOSTICS.md reports (17.2 / +0.657).
    rows, tinfo = evaluate(student, val_texts, val_vecs, dims, args.device, args.eval_bs)
    print(f"\n  teacher geometry: eff.rank {tinfo['eff_rank']:.1f}/{TEACHER_DIM}  "
          f"mean-cos {tinfo['mean_cos']:+.3f}")
    print(f"  {'step':>7} {'dim':>5} {'spearman':>9} {'eff.rank':>9} {'mean-cos':>9}")
    for d in dims:
        r = rows[d]
        print(f"  {0:>7} {d:>5} {r['spearman_vs_teacher']:>9.3f} {r['eff_rank']:>9.1f} {r['mean_cos']:>+9.3f}")

    step, t0, hist = 0, time.time(), []
    done = False
    while not done:
        for texts, tvec in dl:
            tvec = tvec.to(args.device, non_blocking=True)
            feats = student.tokenize(texts)
            feats = {k: (v.to(args.device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in feats.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                emb = student(feats)["sentence_embedding"]

            loss, kl_sum, mse_sum = 0.0, 0.0, 0.0
            for d, w in zip(dims, weights):
                l, kl, mse = relational_loss(emb[:, :d], tvec[:, :d] if args.truncate_teacher else tvec,
                                             args.tau, args.w_kl, args.w_mse)
                loss = loss + w * l
                kl_sum += float(kl) * w
                mse_sum += float(mse) * w
            loss = loss / sum(weights)

            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at step {step} (kl={kl_sum} mse={mse_sum}) -- refusing "
                                 f"to train on it")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_steps == 0:
                el = time.time() - t0
                w_tot = sum(weights)
                print(f"  step {step:,}/{steps:,}  loss {loss.item():.4f}  "
                      f"(kl {kl_sum/w_tot:.4f} · mse {mse_sum/w_tot:.5f})  "
                      f"lr {sched.get_last_lr()[0]:.2e}  {step/el:.2f} it/s", flush=True)
            if step % args.eval_steps == 0 or step == steps:
                rows, _ = evaluate(student, val_texts, val_vecs, dims, args.device, args.eval_bs)
                for d in dims:
                    r = rows[d]
                    print(f"  {step:>7} {d:>5} {r['spearman_vs_teacher']:>9.3f} "
                          f"{r['eff_rank']:>9.1f} {r['mean_cos']:>+9.3f}", flush=True)
                hist.append({"step": step, **{str(d): rows[d] for d in dims}})
                (out / "geometry_trend.json").write_text(json.dumps(hist, indent=2))
                student.save(str(out / f"checkpoint-{step}"))
            if step >= steps:
                done = True
                break

    student.save(str(out / "final"))
    print(f"\nsaved -> {out/'final'}  (drops straight into train_matryoshka.py / eval_benchmarks.py)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["teacher", "student"], required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true", help="run even while stage-1 is training")
    ap.add_argument("--limit", type=int, default=None, help="cap train chunks (smoke tests)")
    ap.add_argument("--val-limit", type=int, default=512)
    ap.add_argument("--max-seq", type=int, default=512)
    # teacher
    ap.add_argument("--batch-size", type=int, default=64)
    # student
    ap.add_argument("--init", default="checkpoints/mlm_stage1/checkpoint-2000")
    ap.add_argument("--output-dir", default="checkpoints/distilled")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--w-kl", type=float, default=1.0, help="ranking: which texts are near which")
    ap.add_argument("--w-mse", type=float, default=10.0, help="absolute scale: pulls mean-cos down")
    ap.add_argument("--truncate-teacher", action="store_true", help="ablation; see module docstring")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-steps", type=int, default=20)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--eval-bs", type=int, default=64)
    args = ap.parse_args()

    os.chdir(PROJECT_DIR)
    guard_gpu(args.force)
    (phase_teacher if args.phase == "teacher" else phase_student)(args)


if __name__ == "__main__":
    main()
