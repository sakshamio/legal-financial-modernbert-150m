"""Longitudinal probes. These exist to make three specific research questions answerable.

Every measurement here can ONLY be taken while the run is alive. There is no way to recover what the
model looked like at step 40,000 once the run has moved past it, so these are captured at every
archived checkpoint and accumulated into research_step*.json. Run by scripts/run_research_daemon.sh.

The three probes, and the question each one exists to answer:

1. NESTING CONCENTRATION -- "does Matryoshka structure emerge, or must it be imposed?"
   Track how variance distributes across the FIRST k coordinates over training. The key statistic is
   not the raw fraction but the CONCENTRATION RATIO:

       ratio(k) = [var in first k coords / total var] / (k / D)

   Under exchangeable coordinates -- no nesting -- variance is spread evenly and ratio(k) == 1.0 for
   every k. ratio(k) > 1 means variance is spontaneously concentrating in the early coordinates, i.e.
   the model is discovering nested structure with no MRL loss anywhere. ratio(k) ~ 1 flat across all
   of pretraining means MRL IMPOSES nesting rather than discovering it. Either answer is a result;
   nobody appears to have measured it from random init.

2. DOMAIN DIFFERENTIATION -- "when does finance stop looking like law?"
   At step 2,000 the neighbours of `revenue` were `liability` and `indemnification`: finance and law
   were NOT separated. This tracks when they separate, via a linear probe over domain labels and the
   between/within-centroid spread. The trajectory of that separation is the developmental result.

3. CONCEPT BEFORE TOKEN -- "is the semantic slot learned before the surface form?"
   At step 2,000, masking `10-Q` produced `10-K` in the top-3: the model knew the blank took an SEC
   FORM before it knew WHICH form. We measure that directly by scoring two things at every step --
   whether the exact token is recovered, and whether ANY token of the same class is. The GAP between
   those two curves is the phenomenon. Contexts are MINED from the real corpus, not hand-written:
   these tokens occur ~7 times per million, so hand-written probes would be a tiny biased sample.

CPU by default; on this unified-memory box CPU work costs the bandwidth-bound GPU run ~1-2%, and the
probes run for a few minutes once every ~5 hours (one checkpoint), so the duty cycle is negligible.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import ModernBertForMaskedLM, ModernBertModel, PreTrainedTokenizerFast

PROJECT_DIR = Path(__file__).resolve().parent.parent
TOKENIZER_DIR = PROJECT_DIR / "tokenizer"
PACKED_DIR = PROJECT_DIR / "data" / "packed"
CKPT_DIR = PROJECT_DIR / "checkpoints" / "mlm_stage1"

# The packed corpus is written in alphabetical file order, so each domain is a contiguous region of
# the token stream. Fractions of train.bin, from the deduped source sizes. (Same as analyze_loss.py.)
DOMAIN_REGIONS = {
    "financial_edgar": (0.00, 0.24),
    "financial_pol": (0.24, 0.31),
    "general_web": (0.31, 0.55),
    "legal_caselaw": (0.55, 0.84),
    "legal_contracts": (0.84, 0.96),
    "legal_regulations": (0.96, 1.00),
}

# Classes for probe 3. The concept-before-token claim is that the model learns the CLASS of the blank
# ("this slot takes an SEC form") before it learns WHICH member of the class goes there.
TOKEN_CLASSES = {
    "sec_form": ["10-K", "10-Q", "8-K", "20-F", "6-K", "S-1", "DEF 14A"],
    "reporter": ["N.W.", "N.E.", "S.W.", "S.E.", "N.W.2d", "N.E.2d", "S.W.2d", "So.2d", "A.2d",
                 "P.2d", "F.2d", "F.3d", "F. Supp.", "S. Ct.", "L. Ed."],
    "statute": ["U.S.C.", "C.F.R.", "P.A.", "P.L.", "R.S.", "G.S.", "K.S.A.", "H.B.", "S.B.",
                "W.S.", "G.L.", "A.L.", "C.S.", "R.C.M."],
    "entity_suffix": ["Inc.", "Corp.", "LLP", "L.P.", "Ltd.", "Co.", "N.A.", "plc"],
    "fin_shorthand": ["IFRS", "ROE", "ROI", "P/E", "YoY", "QoQ", "Q1", "Q2", "Q3", "Q4"],
    "quarter_marker": ["Q1", "Q2", "Q3", "Q4"],
    "signal": ["et seq.", "id.", "infra", "e.g.", "i.e.", "cf.", "v."],
    "subdivision": ["Pt.", "Subsec.", "Subsecs.", "Subd.", "Subdiv.", "Secs.", "Amend."],
    "session": ["Sp. Sess.", "Reg. Sess.", "Ex. Sess."],
}


def checkpoints():
    return sorted(CKPT_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))


@torch.no_grad()
def embed_passages(enc, m, lo, hi, n, block, device, seed=0):
    rng = np.random.default_rng(seed)
    span = (hi - lo) // block
    if span < 1:
        return None
    idx = rng.choice(span, size=min(n, span), replace=False)
    out = []
    for i in idx:
        s = lo + int(i) * block
        ids = torch.from_numpy(np.asarray(m[s : s + block]).astype(np.int64))[None].to(device)
        h = enc(input_ids=ids).last_hidden_state[0]
        out.append(h.mean(0).float())  # mean pooling -- what stage 2 will use
    return torch.stack(out)


# ---------------------------------------------------------------------------------------------------
# PROBE 1 -- does nesting emerge on its own?
# ---------------------------------------------------------------------------------------------------
def nesting_concentration(E):
    """Variance in the first k COORDINATES (not the top-k PCA components -- MRL truncates coordinates).

    Reported as a ratio against the exchangeable-coordinate null (k/D). 1.0 means no nesting.
    """
    Ec = E - E.mean(0, keepdim=True)
    per_coord = (Ec ** 2).sum(0)              # variance carried by each coordinate
    tot = per_coord.sum()
    D = E.shape[1]
    cum = torch.cumsum(per_coord, 0) / tot

    rows = {}
    for k in (8, 16, 32, 64, 128, 256, 384, 512, 640, 768):
        if k > D:
            continue
        frac = float(cum[k - 1])
        rows[k] = {"var_fraction": frac, "null": k / D, "concentration_ratio": frac / (k / D)}

    # Gini over the per-coordinate variances: a single scalar for "how unequal are the coordinates".
    # 0 = perfectly even (no nesting possible); ->1 = a few coordinates carry everything.
    v = torch.sort(per_coord).values.double()
    n = v.numel()
    gini = float((2 * torch.arange(1, n + 1).double() @ v) / (n * v.sum()) - (n + 1) / n)
    return {"by_k": rows, "coord_variance_gini": gini, "total_dims": int(D)}


# ---------------------------------------------------------------------------------------------------
# PROBE 2 -- when does finance stop looking like law?
# ---------------------------------------------------------------------------------------------------
def domain_differentiation(enc, m, device, n_per_domain, block):
    total = len(m)
    embs, labels, names = [], [], []
    for di, (name, (a, b)) in enumerate(DOMAIN_REGIONS.items()):
        E = embed_passages(enc, m, int(a * total), int(b * total), n_per_domain, block, device, seed=di)
        if E is None:
            continue
        embs.append(E)
        labels += [di] * len(E)
        names.append(name)
    if len(embs) < 2:
        return None
    X = torch.cat(embs)
    y = np.array(labels)

    # centroid cosine matrix -- the readable version: how similar is each domain's mean vector?
    cents = F.normalize(torch.stack([e.mean(0) for e in embs]), dim=-1)
    cos = (cents @ cents.T).numpy()
    pairs = {f"{names[i]}|{names[j]}": float(cos[i, j])
             for i in range(len(names)) for j in range(i + 1, len(names))}

    # separability: between-centroid spread vs within-domain spread. Rises as domains pull apart.
    within = float(torch.stack([ (e - e.mean(0)).norm(dim=-1).mean() for e in embs ]).mean())
    C = torch.stack([e.mean(0) for e in embs])
    between = float(torch.pdist(C).mean())
    fisher = between / max(within, 1e-9)

    # linear probe: can a linear map read the domain off the embedding? This is the headline number.
    probe_acc = None
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        Xn = F.normalize(X, dim=-1).numpy()
        clf = LogisticRegression(max_iter=2000, multi_class="multinomial")
        probe_acc = float(cross_val_score(clf, Xn, y, cv=3, scoring="accuracy").mean())
    except Exception:
        pass

    return {
        "domains": names,
        "linear_probe_acc": probe_acc,
        "chance": 1.0 / len(names),
        "fisher_ratio": fisher,
        "centroid_cosine": pairs,
        "law_vs_finance": float(np.mean([
            cos[names.index(l), names.index(f)]
            for l in ("legal_caselaw", "legal_contracts") if l in names
            for f in ("financial_edgar", "financial_pol") if f in names
        ])) if any(n.startswith("legal") for n in names) else None,
    }


# ---------------------------------------------------------------------------------------------------
# PROBE 3 -- is the concept learned before the token?
# ---------------------------------------------------------------------------------------------------
def mine_contexts(m, tok, per_token, window, scan_tokens):
    """Find REAL occurrences of each domain token in the corpus.

    Hand-written probe sentences would be a tiny, biased sample and could not cover 156 tokens. These
    tokens occur ~7 times per million, so we scan a large slice of the stream and index the hits.
    """
    want = {}
    for cls, terms in TOKEN_CLASSES.items():
        for t in terms:
            for variant in (t, " " + t):
                tid = tok.convert_tokens_to_ids(variant)
                if tid is not None and tid != tok.unk_token_id:
                    want[tid] = (t, cls)
    if not want:
        return {}

    lut = np.zeros(len(tok), dtype=bool)
    lut[np.fromiter(want.keys(), dtype=np.int64)] = True

    found = defaultdict(list)
    step = 50_000_000
    scanned = 0
    for s in range(0, min(scan_tokens, len(m) - window), step):
        chunk = np.asarray(m[s : s + step]).astype(np.int64)
        hits = np.flatnonzero(lut[chunk])
        for h in hits:
            g = s + int(h)
            if g < window // 2 or g > len(m) - window:
                continue
            term, cls = want[int(chunk[h])]
            if len(found[term]) < per_token:
                found[term].append((g, cls))
        # .get, NOT found[t] -- indexing a defaultdict here would CREATE empty entries for every
        # token the corpus never contains, and those empty lists then blow up downstream.
        if all(len(found.get(t, ())) >= per_token for t, _ in want.values()):
            break
    return {t: v for t, v in found.items() if v}


@torch.no_grad()
def concept_before_token(model, tok, m, device, per_token=6, window=128, scan_tokens=400_000_000):
    found = mine_contexts(m, tok, per_token, window, scan_tokens)
    if not found:
        return None

    # id sets: the "exact" answer is either variant (bare or space-prefixed); the "class" answer is
    # any variant of any member of the same class.
    ids_of = {}
    for cls, terms in TOKEN_CLASSES.items():
        for t in terms:
            v = [tok.convert_tokens_to_ids(x) for x in (t, " " + t)]
            ids_of[t] = [i for i in v if i is not None and i != tok.unk_token_id]
    class_ids = {cls: sorted({i for t in terms for i in ids_of.get(t, [])})
                 for cls, terms in TOKEN_CLASSES.items()}

    rows, agg = [], defaultdict(lambda: [0, 0, 0, 0])  # cls -> [n, exact_top1, class_top1, class_top5]
    for term, occs in found.items():
        cls = occs[0][1]
        exact = ids_of.get(term, [])
        if not exact:
            continue
        cids = class_ids[cls]
        ranks, e1, c1, c5 = [], 0, 0, 0
        for pos, _ in occs:
            s = pos - window // 2
            ids = torch.from_numpy(np.asarray(m[s : s + window]).astype(np.int64))[None].clone()
            at = window // 2
            ids[0, at] = tok.mask_token_id
            lg = model(input_ids=ids.to(device)).logits
            lg = lg[0][at] if lg.dim() == 3 else lg[at]

            best_exact = max(float(lg[i]) for i in exact)
            ranks.append(int((lg > best_exact).sum()) + 1)
            top5 = lg.topk(5).indices.tolist()
            e1 += int(top5[0] in exact)
            c1 += int(top5[0] in cids)
            c5 += int(any(t in cids for t in top5))
        n = len(occs)
        rows.append({
            "term": term, "class": cls, "n": n,
            "median_rank": float(np.median(ranks)),
            "exact_top1": e1 / n, "class_top1": c1 / n, "class_top5": c5 / n,
        })
        a = agg[cls]
        a[0] += n; a[1] += e1; a[2] += c1; a[3] += c5

    by_class = {c: {"n": v[0], "exact_top1": v[1] / v[0], "class_top1": v[2] / v[0],
                    "class_top5": v[3] / v[0]} for c, v in agg.items() if v[0]}
    N = sum(v[0] for v in agg.values())
    overall = {
        "n": N,
        "exact_top1": sum(v[1] for v in agg.values()) / N,
        "class_top1": sum(v[2] for v in agg.values()) / N,
        "class_top5": sum(v[3] for v in agg.values()) / N,
        "median_rank": float(np.median([r["median_rank"] for r in rows])),
    }
    # THE measurement: how far ahead of the exact token is the concept?
    overall["concept_lead"] = overall["class_top1"] - overall["exact_top1"]
    return {"overall": overall, "by_class": by_class, "by_token": sorted(rows, key=lambda r: -r["exact_top1"])}


# ---------------------------------------------------------------------------------------------------
def show_trend():
    files = sorted(PROJECT_DIR.glob("research_step*.json"),
                   key=lambda p: int(p.stem.split("step")[1]))
    if not files:
        print("no research_step*.json yet")
        return

    print("\n1. NESTING -- does Matryoshka structure emerge on its own?")
    print("   concentration ratio = (var in first k coords) / (k/D).   1.00 = NO nesting.\n")
    print(f"   {'step':>8} {'k=64':>8} {'k=128':>8} {'k=256':>8} {'gini':>8}")
    for f in files:
        d = json.loads(f.read_text())
        n = d.get("nesting", {}).get("by_k", {})
        g = d.get("nesting", {}).get("coord_variance_gini", 0)
        r = lambda k: n.get(str(k), {}).get("concentration_ratio", 0)
        print(f"   {d['step']:>8,} {r(64):>8.2f} {r(128):>8.2f} {r(256):>8.2f} {g:>8.3f}")
    print("\n   Flat at ~1.00 => MRL IMPOSES nesting. Climbing => the model DISCOVERS it unprompted.")

    print("\n2. DOMAIN DIFFERENTIATION -- when does finance stop looking like law?\n")
    print(f"   {'step':>8} {'probe acc':>10} {'chance':>8} {'fisher':>8} {'law~fin cos':>12}")
    for f in files:
        d = json.loads(f.read_text())
        x = d.get("domains") or {}
        pa = x.get("linear_probe_acc")
        print(f"   {d['step']:>8,} {pa if pa else 0:>10.1%} {x.get('chance',0):>8.1%} "
              f"{x.get('fisher_ratio',0):>8.2f} {x.get('law_vs_finance') or 0:>+12.3f}")
    print("\n   WANT: probe acc UP, fisher UP, law~fin cosine DOWN (the domains pulling apart).")

    print("\n3. CONCEPT BEFORE TOKEN -- is the slot learned before the surface form?\n")
    print(f"   {'step':>8} {'exact top1':>11} {'class top1':>11} {'LEAD':>7} {'med rank':>9}")
    for f in files:
        d = json.loads(f.read_text())
        o = (d.get("concept") or {}).get("overall") or {}
        print(f"   {d['step']:>8,} {o.get('exact_top1',0):>11.1%} {o.get('class_top1',0):>11.1%} "
              f"{o.get('concept_lead',0):>+7.1%} {o.get('median_rank',0):>9,.0f}")
    print("\n   A POSITIVE and early-peaking lead is the finding: the model knows the blank takes an")
    print("   SEC form / a reporter / a statute long before it knows WHICH one.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trend", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--passages", type=int, default=256)
    ap.add_argument("--per-domain", type=int, default=96)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--contexts-per-token", type=int, default=6)
    ap.add_argument("--force", action="store_true", help="recompute even if the json exists")
    args = ap.parse_args()

    if args.trend:
        show_trend()
        return

    tok = PreTrainedTokenizerFast.from_pretrained(str(TOKENIZER_DIR))
    ckpt = Path(args.checkpoint) if args.checkpoint else checkpoints()[-1]
    step = int(ckpt.name.split("-")[1])
    out_path = PROJECT_DIR / f"research_step{step}.json"
    if out_path.exists() and not args.force:
        print(f"{out_path.name} exists; skipping (use --force)")
        return

    m = np.memmap(PACKED_DIR / "train.bin", dtype=np.uint16, mode="r")
    print(f"checkpoint {ckpt.name} (step {step:,})")

    enc = ModernBertModel.from_pretrained(str(ckpt)).to(args.device).eval()

    print("  [1/3] nesting concentration ...")
    E = embed_passages(enc, m, 0, len(m), args.passages, args.block, args.device)
    nest = nesting_concentration(E)

    print("  [2/3] domain differentiation ...")
    dom = domain_differentiation(enc, m, args.device, args.per_domain, args.block)

    print("  [3/3] concept-before-token (mining real contexts) ...")
    mlm = ModernBertForMaskedLM.from_pretrained(str(ckpt)).to(args.device).eval()
    con = concept_before_token(mlm, tok, m, args.device, args.contexts_per_token)

    out = {"step": step, "nesting": nest, "domains": dom, "concept": con}
    out_path.write_text(json.dumps(out, indent=2, default=float))

    r64 = nest["by_k"][64]["concentration_ratio"]
    print(f"\n  nesting ratio @k=64 : {r64:.2f}   (1.00 = no nesting)")
    if dom:
        print(f"  domain probe acc    : {dom['linear_probe_acc']:.1%}  (chance {dom['chance']:.1%})"
              if dom.get("linear_probe_acc") else "")
        print(f"  law~finance cosine  : {dom['law_vs_finance']:+.3f}")
    if con:
        o = con["overall"]
        print(f"  exact-token top1    : {o['exact_top1']:.1%}")
        print(f"  same-CLASS top1     : {o['class_top1']:.1%}   -> concept lead {o['concept_lead']:+.1%}")
    print(f"\nsaved -> {out_path.name}")


if __name__ == "__main__":
    main()
