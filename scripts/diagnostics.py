"""Deeper diagnostics: does this model look like it will become a good EMBEDDING model?

MLM loss tells you the model is learning. It does NOT tell you whether the representation will be any
good for retrieval -- and that is what we are actually building. These probes answer questions the loss
curve cannot:

1. EMBEDDING GEOMETRY (the most important one). Transformer representations notoriously collapse into a
   narrow cone -- every embedding ends up similar to every other ("anisotropy"). A model can have a
   great MLM loss and a useless embedding space. Worse, if the representation only really uses a few of
   its 768 dimensions, Matryoshka truncation is meaningless: the dims are not carrying independent
   information to nest.
      - mean pairwise cosine of random passages: near 0 is healthy, near 1 is collapse.
      - PCA spectrum: how many dims to explain 90/95/99% of variance. This DIRECTLY predicts how well
        MRL truncation to 512/256/128/64 will work -- if 64 PCA dims already cover most variance, a
        64-dim truncation has a real shot.
      - effective rank (entropy of the normalized eigenspectrum).

2. LEARNING-CURVE EXTRAPOLATION. Fit L(t) = a*t^-b + c across checkpoints to predict where loss lands at
   271k steps. Tells us whether the 30-day budget is enough BEFORE spending it.

3. DOMAIN-TOKEN LEARNING. We spent 156 vocab slots on citation machinery (U.S.C., C.F.R., 10-K).
   Are they actually being learned, or did we waste the slots? Prediction accuracy on exactly those ids.

4. LOSS BY TOKEN FREQUENCY. Common tokens are easy. If loss is only dropping on frequent tokens, the
   model is learning statistics, not language. Bucketed by corpus frequency.

5. LOSS BY POSITION. NOTE: in a CAUSAL LM you expect loss to fall later in the sequence as left-context
   accumulates. MLM is BIDIRECTIONAL -- a token at position 10 already sees the whole block -- so a FLAT
   curve is the correct, healthy result and confirms bidirectionality. A rise at the edges would signal
   boundary artifacts.

6. SEMANTIC NEIGHBOURHOODS. Nearest neighbours of domain terms in embedding space -- the earliest
   readable sign that meaning is forming.

CPU by default: on unified memory, CPU work costs a bandwidth-bound GPU run ~2%, and a GPU eval costs
far more.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import DataCollatorForLanguageModeling, ModernBertForMaskedLM, ModernBertModel, PreTrainedTokenizerFast

PROJECT_DIR = Path(__file__).resolve().parent.parent
TOKENIZER_DIR = PROJECT_DIR / "tokenizer"
PACKED_DIR = PROJECT_DIR / "data" / "packed"
CKPT_DIR = PROJECT_DIR / "checkpoints" / "mlm_stage1"

NEIGHBOUR_PROBES = [
    "indemnification", "plaintiff", "defendant", "revenue", "liability",
    "10-K", "U.S.C.", "warranty", "dividend", "negligence",
]


def checkpoints():
    return sorted(CKPT_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))


# ---------------------------------------------------------------------------------------------------
# 1. embedding geometry -- the diagnostic that decides whether Matryoshka can work at all
# ---------------------------------------------------------------------------------------------------
@torch.no_grad()
def embedding_geometry(ckpt, tok, m, n_passages=256, block=256, device="cpu"):
    enc = ModernBertModel.from_pretrained(str(ckpt)).to(device).eval()
    rng = np.random.default_rng(0)
    starts = rng.integers(0, len(m) - block, size=n_passages)

    embs = []
    for s in starts:
        ids = torch.from_numpy(np.asarray(m[s : s + block]).astype(np.int64))[None].to(device)
        h = enc(input_ids=ids).last_hidden_state[0]     # [T, D]
        embs.append(h.mean(0))                          # mean pooling -- what stage 2 will use
    E = torch.stack(embs).float()                       # [N, D]

    # --- anisotropy: are all passages collapsing onto each other?
    En = torch.nn.functional.normalize(E, dim=-1)
    sim = (En @ En.T)
    off = ~torch.eye(len(En), dtype=torch.bool)
    mean_cos = sim[off].mean().item()

    # --- PCA spectrum: how many dims carry the information?
    Ec = E - E.mean(0, keepdim=True)
    # svd on the centered matrix; eigenvalues of covariance ~ s^2
    s = torch.linalg.svdvals(Ec)
    var = (s ** 2)
    var = var / var.sum()
    cum = torch.cumsum(var, 0)
    dims_for = {q: int((cum < q).sum().item()) + 1 for q in (0.90, 0.95, 0.99)}

    # --- effective rank = exp(entropy of the normalized spectrum)
    p = var[var > 0]
    eff_rank = float(torch.exp(-(p * p.log()).sum()).item())

    # --- variance captured by the MATRYOSHKA prefixes (this is the actual MRL question:
    #     the first k *coordinates*, not the top-k PCA components)
    tot = (Ec ** 2).sum()
    prefix = {k: float(((Ec[:, :k] ** 2).sum() / tot).item()) for k in (64, 128, 256, 512, 768) if k <= E.shape[1]}

    return {
        "mean_pairwise_cosine": mean_cos,
        "dims_for_90pct_var": dims_for[0.90],
        "dims_for_95pct_var": dims_for[0.95],
        "dims_for_99pct_var": dims_for[0.99],
        "effective_rank": eff_rank,
        "total_dims": int(E.shape[1]),
        "variance_in_first_k_coords": prefix,
    }, enc


# ---------------------------------------------------------------------------------------------------
# 2. learning curve extrapolation
# ---------------------------------------------------------------------------------------------------
def extrapolate(max_steps=271000):
    """Fit L(t) = a * t^-b + c over the logged losses; predict the final loss."""
    log = (PROJECT_DIR / "train_stage1.log").read_text(errors="ignore")
    import re

    pairs = []
    for m_ in re.finditer(r"'loss': ([\d.]+).*?'epoch'", log):
        pairs.append(float(m_.group(1)))
    if len(pairs) < 20:
        return None
    # losses are logged every N steps in order; reconstruct step axis from the log line count
    steps = np.linspace(1, len(pairs), len(pairs)) * 50  # logging_steps=50
    y = np.array(pairs)
    keep = steps > steps.max() * 0.15  # drop the very noisy warmup head
    steps, y = steps[keep], y[keep]
    if len(y) < 15:
        return None

    from scipy.optimize import curve_fit

    def f(t, a, b, c):
        return a * np.power(t, -b) + c

    try:
        p, _ = curve_fit(f, steps, y, p0=[20.0, 0.3, 1.5], maxfev=20000,
                         bounds=([0, 0.01, 0.0], [1e4, 2.0, 8.0]))
    except Exception:
        return None
    a, b, c = p
    return {
        "fit": {"a": float(a), "b": float(b), "c": float(c)},
        "irreducible_loss_c": float(c),
        "predicted_at_max_steps": float(f(max_steps, *p)),
        "fitted_on_steps": [int(steps.min()), int(steps.max())],
    }


# ---------------------------------------------------------------------------------------------------
# 3/4/5. token-level breakdowns
# ---------------------------------------------------------------------------------------------------
@torch.no_grad()
def token_breakdowns(ckpt, tok, m, n_blocks=48, block=1024, device="cpu"):
    model = ModernBertForMaskedLM.from_pretrained(str(ckpt)).to(device).eval()
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=True, mlm_probability=0.3)
    V = len(tok)

    # corpus token frequencies -> frequency buckets
    sample = np.asarray(m[:: max(1, len(m) // 50_000_000)][:50_000_000])
    freq = np.bincount(sample, minlength=V).astype(np.float64)
    order = np.argsort(-freq)
    rank = np.empty(V, dtype=np.int64)
    rank[order] = np.arange(V)

    BUCKETS = [(0, 100, "top 100"), (100, 1000, "100-1k"), (1000, 10000, "1k-10k"), (10000, V, "10k+")]
    bstats = {name: [0.0, 0, 0] for _, _, name in BUCKETS}   # loss_sum, tokens, correct

    dstat = [0.0, 0, 0]  # domain tokens are handled separately -- see domain_token_probe()

    pos_bins = np.zeros(4)
    pos_cnt = np.zeros(4)

    rng = np.random.default_rng(1)
    starts = rng.integers(0, len(m) - block, size=n_blocks)
    lossf = torch.nn.CrossEntropyLoss(reduction="none")

    for s in starts:
        ids = torch.from_numpy(np.asarray(m[s : s + block]).astype(np.int64))
        batch = collator([{"input_ids": ids}])
        inp, lab = batch["input_ids"].to(device), batch["labels"].to(device)
        out = model(input_ids=inp)                       # no labels -> dense logits [1,T,V]
        logits = out.logits[0]
        mask = (lab[0] != -100)
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        gold = lab[0][idx]
        lg = logits[idx]
        l = lossf(lg, gold)
        pred = lg.argmax(-1)

        for j in range(idx.numel()):
            g = int(gold[j]); li = float(l[j]); ok = int(pred[j] == gold[j])
            r = rank[g]
            for lo, hi, name in BUCKETS:
                if lo <= r < hi:
                    bstats[name][0] += li; bstats[name][1] += 1; bstats[name][2] += ok
                    break
            b = min(3, int(idx[j].item() / block * 4))
            pos_bins[b] += li; pos_cnt[b] += 1

    freq_rows = {
        name: {"loss": v[0] / v[1], "acc": v[2] / v[1], "n": v[1]}
        for name, v in bstats.items() if v[1] > 0
    }
    dom = {"loss": dstat[0] / dstat[1], "acc": dstat[2] / dstat[1], "n": dstat[1]} if dstat[1] else None
    pos = [float(pos_bins[i] / pos_cnt[i]) if pos_cnt[i] else None for i in range(4)]
    return freq_rows, dom, pos, model


# ---------------------------------------------------------------------------------------------------
@torch.no_grad()
def domain_token_probe(model, tok, device="cpu"):
    """Are the 156 custom tokens actually being learned, or did we waste the vocab slots?

    Random block sampling CANNOT answer this: these tokens occur ~7 times per MILLION, so a few dozen
    blocks contain approximately zero of them (our first attempt got n=1). Instead we CONSTRUCT the
    examples -- mask the domain token inside a real sentence and ask the model to recover it.
    """
    cases = [
        ("U.S.C.",  "Pursuant to 15 [MASK] § 78j(b), the defendant is liable for securities fraud."),
        ("C.F.R.",  "The rule is codified at 17 [MASK] § 240.10b-5 and applies to all issuers."),
        ("10-K",    "The Company filed its annual report on Form [MASK] with the Commission."),
        ("10-Q",    "The Company filed its quarterly report on Form [MASK] for the period ended June 30."),
        ("8-K",     "A current report on Form [MASK] was furnished to disclose the material event."),
        ("P.A.",    "This section was amended by [MASK] 97-123 during the regular session."),
        ("e.g.",    "Certain instruments, [MASK], notes and debentures, are excluded from the definition."),
        ("§§",      "The provisions of [MASK] 12-101 through 12-115 shall govern this proceeding."),
        ("Inc.",    "The agreement was executed by Acme Corp. and Beta Holdings, [MASK]"),
        ("Q3",      "Revenue for [MASK] of fiscal 2023 increased over the prior-year quarter."),
    ]
    hits = 0
    rows = []
    for term, text in cases:
        # Each domain term exists as TWO ids: bare ("10-K") and space-prefixed (" 10-K"). Running
        # text almost always contains the space-prefixed one, so checking only the bare id meant this
        # probe could essentially never fire -- it reported 0% at step 6,000 while the mined-context
        # probe in research_probes.py, which accepts either, reported 31.1% on the same checkpoint.
        tids = [i for i in (tok.convert_tokens_to_ids(term), tok.convert_tokens_to_ids(" " + term))
                if i is not None and i != tok.unk_token_id]
        if not tids:
            continue
        # (both variants scored below)
        enc_in = tok(text, return_tensors="pt")
        enc_in = {k: v.to(device) for k, v in enc_in.items() if k in ("input_ids", "attention_mask")}
        pos = (enc_in["input_ids"][0] == tok.mask_token_id).nonzero()
        if len(pos) == 0:
            continue
        logits = model(**enc_in).logits
        lg = logits[0][pos[0, 0]] if logits.dim() == 3 else logits[pos[0, 0]]
        top5 = lg.topk(5).indices.tolist()
        best = max(float(lg[i]) for i in tids)       # score the better of the two variants
        rank = int((lg > best).sum()) + 1            # rank of the CORRECT token out of 50,368
        ok = top5[0] in tids
        hits += ok
        rows.append({
            "term": term, "rank_of_correct": rank, "top1_correct": bool(ok),
            "top3": [tok.decode([t]).strip() for t in top5[:3]],
        })
    return {"top1_accuracy": hits / max(len(rows), 1), "n": len(rows), "rows": rows}


@torch.no_grad()
def neighbours(enc, tok, device="cpu", k=4):
    """Nearest neighbours using CONTEXTUAL embeddings.

    An earlier version probed the static input-embedding matrix and got noise. That was the wrong place
    to look: the input matrix is barely trained and, more importantly, meaning in a transformer lives in
    the CONTEXTUAL representation, not the lookup table. Here each term is embedded inside a short
    domain sentence and we compare those contextual vectors.
    """
    frame = {
        "indemnification": "The party shall provide indemnification for all losses.",
        "plaintiff": "The plaintiff filed a complaint in district court.",
        "defendant": "The defendant moved to dismiss the complaint.",
        "revenue": "Total revenue for the fiscal year increased.",
        "liability": "The company recorded a liability on its balance sheet.",
        "warranty": "The seller makes no warranty as to the goods.",
        "dividend": "The board declared a dividend payable to shareholders.",
        "negligence": "The claim alleges negligence by the operator.",
        "damages": "The court awarded damages to the injured party.",
        "collateral": "The loan is secured by collateral pledged by the borrower.",
    }
    terms = [t for t in frame if len(tok(t, add_special_tokens=False)["input_ids"]) >= 1]
    vecs = []
    for t in terms:
        enc_in = tok(frame[t], return_tensors="pt")
        # ModernBERT takes no token_type_ids
        enc_in = {k: v.to(device) for k, v in enc_in.items() if k in ("input_ids", "attention_mask")}
        h = enc(**enc_in).last_hidden_state[0].mean(0)
        vecs.append(h.float())
    E = torch.nn.functional.normalize(torch.stack(vecs), dim=-1)
    sim = E @ E.T
    sim.fill_diagonal_(-1)
    out = {}
    for i, t in enumerate(terms):
        top = sim[i].topk(min(k, len(terms) - 1)).indices.tolist()
        out[t] = [f"{terms[j]} ({sim[i][j]:.2f})" for j in top]
    return out


def show_trend():
    """The single-point numbers are hard to judge. The TRENDS are what matter:

      effective rank   must CLIMB   (17/768 at step 2k is fine; 17/768 at step 50k is a dead model)
      mean cosine      must FALL    (anisotropy shrinking = embeddings spreading out)
      rare-token loss  must FALL    (this is where real language learning shows up)
      domain-token acc must CLIMB   (proof the 156 custom vocab slots were worth it)
    """
    files = sorted(PROJECT_DIR.glob("diagnostics_step*.json"),
                   key=lambda p: int(p.stem.split("step")[1]))
    if len(files) < 1:
        print("no diagnostics files yet")
        return
    print(f"{'step':>8} {'eff.rank':>9} {'mean cos':>9} {'rare loss':>10} {'dom acc':>8} {'90% var':>8}")
    for f in files:
        d = json.loads(f.read_text())
        g = d.get("geometry", {})
        rare = (d.get("freq_buckets", {}) or {}).get("10k+", {}).get("loss")
        dom = (d.get("domain_tokens") or {}).get("top1_accuracy")
        print(f"{d['step']:>8,} {g.get('effective_rank', 0):>9.1f} {g.get('mean_pairwise_cosine', 0):>+9.3f} "
              f"{rare if rare else 0:>10.2f} {dom if dom is not None else 0:>7.0%} "
              f"{g.get('dims_for_90pct_var', 0):>8}")
    print()
    print("  WANT:   eff.rank UP   mean-cos DOWN   rare-loss DOWN   dom-acc UP")
    print("  A flat effective rank by ~step 50k would mean the representation is collapsing")
    print("  and Matryoshka truncation will not work -- the earliest actionable warning we have.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trend", action="store_true", help="show metric trends across all saved checkpoints")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--passages", type=int, default=192)
    ap.add_argument("--blocks", type=int, default=32)
    args = ap.parse_args()

    if args.trend:
        show_trend()
        return

    tok = PreTrainedTokenizerFast.from_pretrained(str(TOKENIZER_DIR))
    ckpt = Path(args.checkpoint) if args.checkpoint else checkpoints()[-1]
    step = int(ckpt.name.split("-")[1])
    m = np.memmap(PACKED_DIR / "train.bin", dtype=np.uint16, mode="r")
    print(f"checkpoint: {ckpt.name} (step {step:,})\n")

    # ---- 1. embedding geometry
    print("=" * 78)
    print("1. EMBEDDING GEOMETRY  -- will this ever be a good retrieval model?")
    print("=" * 78)
    geo, enc = embedding_geometry(ckpt, tok, m, args.passages, device=args.device)
    c = geo["mean_pairwise_cosine"]
    verdict = "HEALTHY" if c < 0.5 else ("WATCH" if c < 0.8 else "COLLAPSED")
    print(f"  mean pairwise cosine of random passages : {c:+.3f}   [{verdict}]")
    print("      0 = passages point in independent directions (good).")
    print("      1 = every passage embeds to nearly the same vector -> useless for retrieval.")
    print()
    print(f"  effective rank                          : {geo['effective_rank']:.1f} / {geo['total_dims']}")
    print(f"  dims to explain 90% / 95% / 99% of var  : {geo['dims_for_90pct_var']} / {geo['dims_for_95pct_var']} / {geo['dims_for_99pct_var']}")
    print()
    print("  variance captured by the FIRST k coordinates (this IS the Matryoshka question):")
    for k, v in geo["variance_in_first_k_coords"].items():
        bar = "#" * int(v * 40)
        print(f"      first {k:>3} dims : {v:6.1%}  {bar}")
    print("      MRL training will REDISTRIBUTE variance toward the early coords; this is the 'before'.")

    # ---- 2. learning curve
    print("\n" + "=" * 78)
    print("2. LEARNING-CURVE EXTRAPOLATION  -- is 30 days enough?")
    print("=" * 78)
    ex = extrapolate()
    if ex:
        print(f"  fit L(t) = a*t^-b + c  over steps {ex['fitted_on_steps'][0]:,}-{ex['fitted_on_steps'][1]:,}")
        print(f"  predicted loss at 271,000 steps : {ex['predicted_at_max_steps']:.2f}")
        print(f"  irreducible term c              : {ex['irreducible_loss_c']:.2f}")
        print("  (early-run fits are OPTIMISTIC and unstable -- treat as a rough bound, not a promise)")
    else:
        print("  not enough logged points yet")

    # ---- 3/4/5
    print("\n" + "=" * 78)
    print("3. LOSS BY TOKEN FREQUENCY  -- learning language, or just statistics?")
    print("=" * 78)
    freq, dom, pos, _ = token_breakdowns(ckpt, tok, m, args.blocks, device=args.device)
    print(f"  {'frequency bucket':<18} {'loss':>7} {'top-1':>8} {'n':>8}")
    for name in ["top 100", "100-1k", "1k-10k", "10k+"]:
        if name in freq:
            r = freq[name]
            print(f"  {name:<18} {r['loss']:>7.3f} {r['acc']:>7.1%} {r['n']:>8,}")
    print("  Rare tokens SHOULD be much harder. If they aren't, the model is not really discriminating.")

    print("\n" + "=" * 78)
    print("4. CUSTOM DOMAIN TOKENS  -- were the 156 vocab slots worth it?")
    print("=" * 78)
    mlm = ModernBertForMaskedLM.from_pretrained(str(ckpt)).to(args.device).eval()
    dprobe = domain_token_probe(mlm, tok, args.device)
    print(f"  top-1 accuracy on constructed probes: {dprobe['top1_accuracy']:.0%}  (n={dprobe['n']})")
    print(f"  {'token':<9} {'rank of correct':>16}  top-3 predictions")
    for r in dprobe["rows"]:
        mark = "OK " if r["top1_correct"] else "   "
        print(f"  {mark}{r['term']:<7} {r['rank_of_correct']:>13,}/{len(tok):,}  {r['top3']}")
    print("  Rank 1 = learned. Rank in the thousands = not yet. (These ids BPE could NEVER have learned.)")

    print("\n" + "=" * 78)
    print("5. LOSS BY POSITION  -- FLAT is correct here (MLM is bidirectional, unlike a causal LM)")
    print("=" * 78)
    for i, v in enumerate(pos):
        if v:
            print(f"  tokens {i*25:>3}-{(i+1)*25:>3}% of block : loss {v:.3f}")
    print("  Flat => bidirectional attention is working. A rise at the edges would mean boundary artifacts.")

    print("\n" + "=" * 78)
    print("6. SEMANTIC NEIGHBOURHOODS  -- is meaning forming?")
    print("=" * 78)
    for term, ns in neighbours(enc, tok, args.device).items():
        print(f"  {term:<18} -> {ns}")

    out = {"step": step, "geometry": geo, "extrapolation": ex, "freq_buckets": freq,
           "domain_tokens": dprobe, "loss_by_position": pos}
    (PROJECT_DIR / f"diagnostics_step{step}.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nsaved -> diagnostics_step{step}.json")


if __name__ == "__main__":
    main()
