"""Near-deduplicate the corpus at paragraph level, and rebalance the domain mix.

WHY DEDUP: EDGAR 10-Ks and contracts are boilerplate-heavy -- risk-factor language,
forward-looking-statement disclaimers, standard clause scaffolding recur near-verbatim across thousands
of filings. Measured on a 20k-doc sample: 12.3% duplicate paragraphs in EDGAR, 10.8% in contracts (vs
0.6% in case law). Undeduped, a nominal "2.76 epochs" is really many more passes over the boilerplate
and fewer over the substance -- precisely how a token-limited encoder overfits to templated text.

Normalization (lowercase, digits->#, strip punctuation) before hashing catches boilerplate that differs
only by date, dollar amount, or section number -- which is most of it. This is a coarse near-dedup, not
MinHash/LSH; it is cheap, single-pass, and captures the dominant mode. Company-name-substituted variants
will survive, which we accept.

WHY REBALANCE: stage-2 evaluation leans on contracts (LEDGAR, CUAD) and regulations, but CFR was only
1.5% of pretraining. Upsampling the under-represented, eval-relevant domains narrows the
pretrain/eval distribution mismatch.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUT_DIR = PROJECT_DIR / "data" / "raw_dedup"

MIN_PARA_CHARS = 120  # shorter fragments (headings, "Not Applicable") repeat legitimately; don't dedup them

# No upsampling. The first corpus had regulations at only 1.5%, and the fix could have been to repeat
# CFR 3-6x -- but repeating a tiny set just memorizes it. Instead prepare_corpus.py now pulls *unique*
# statutory text (state codes, US bills, US Code, Federal Register) and broader financial text (SEC
# proceedings, tax rulings), so the mix is balanced with real tokens. Every token here is unique.
UPSAMPLE = {}


def normalize(p):
    p = " ".join(p.split()).lower()
    p = re.sub(r"\d+", "#", p)
    p = re.sub(r"[^\w\s#]", "", p)
    return p


def dedup_source(src, seen, stats):
    """Drop paragraphs whose normalized form was already emitted (globally, across all sources)."""
    in_path = RAW_DIR / f"{src}.jsonl"
    out_path = OUT_DIR / f"{src}.jsonl"
    kept_chars = dropped_chars = 0
    docs_in = docs_out = 0

    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            docs_in += 1
            out_paras = []
            for para in re.split(r"\n\s*\n", rec["text"]):
                p = " ".join(para.split())
                if not p:
                    continue
                if len(p) < MIN_PARA_CHARS:
                    out_paras.append(para)
                    kept_chars += len(p)
                    continue
                h = hashlib.blake2b(normalize(p).encode(), digest_size=16).digest()
                if h in seen:
                    dropped_chars += len(p)
                    continue
                seen.add(h)
                out_paras.append(para)
                kept_chars += len(p)

            text = "\n\n".join(out_paras).strip()
            if len(text) < 200:
                continue
            docs_out += 1
            for _ in range(UPSAMPLE.get(src, 1)):
                fout.write(json.dumps({"text": text, "source": src, "id": rec.get("id", "")}, ensure_ascii=False) + "\n")

    total = kept_chars + dropped_chars
    stats[src] = dict(
        docs_in=docs_in,
        docs_out=docs_out,
        kept_gb=kept_chars / 1024**3,
        dropped_gb=dropped_chars / 1024**3,
        dropped_pct=100 * dropped_chars / max(total, 1),
        upsample=UPSAMPLE.get(src, 1),
    )
    s = stats[src]
    print(
        f"{src:<18} docs {docs_in:>8} -> {docs_out:>8} | dropped {s['dropped_gb']:>5.2f}GB "
        f"({s['dropped_pct']:>5.1f}%) | kept {s['kept_gb']:>5.2f}GB | x{s['upsample']}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seen = set()
    stats = {}

    # Order matters: dedup is "keep first occurrence". Process the highest-signal, least-boilerplate
    # source first, so when text IS shared across sources the copy we keep sits in the corpus we trust
    # most. general_web goes LAST -- if a legal passage also appears on the web, keep the legal copy.
    for src in [
        "legal_caselaw",
        "legal_regulations",
        "legal_contracts",
        "financial_edgar",
        "financial_pol",
        "general_web",
    ]:
        if (RAW_DIR / f"{src}.jsonl").exists():
            dedup_source(src, seen, stats)
        else:
            print(f"[{src}] MISSING -- skipped", flush=True)

    print("\n=== summary ===")
    dropped = sum(s["dropped_gb"] for s in stats.values())
    kept = sum(s["kept_gb"] for s in stats.values())
    print(f"unique paragraphs hashed: {len(seen):,}")
    print(f"dropped {dropped:.2f}GB of duplicated text ({100*dropped/max(kept+dropped,1):.1f}% of corpus)")
    final = sum(s["kept_gb"] * s["upsample"] for s in stats.values())
    print(f"corpus after dedup+upsample: {final:.2f}GB (was {kept+dropped:.2f}GB)")
    print("\nnew domain mix:")
    for src, s in stats.items():
        print(f"  {src:<18} {s['kept_gb']*s['upsample']:>6.2f}GB  ({100*s['kept_gb']*s['upsample']/final:>4.1f}%)")


if __name__ == "__main__":
    main()
