"""Sample the legal + financial pretraining corpus from public HF datasets.

TARGET: ~70GB text / ~15B tokens. Sized so that a 30-day run at the measured 20,665 tok/s (150M,
bf16, sparse_prediction) sees ~53.6B tokens = ~3.7 epochs -- under the ~4-epoch ceiling past which
repeated data yields materially less than fresh tokens.

Composition is deliberate:
  - REGULATIONS were 1.5% of the first corpus but stage-2 eval leans on regulatory/contract language.
    Rather than upsample a tiny CFR set 6x (which just memorizes it), we pull *unique* statutory text:
    state codes, US bills, US Code, Federal Register.
  - FINANCIAL is broadened beyond 10-K filings with SEC proceedings and tax rulings.
  - No upsampling anywhere: every token is unique text.

Writes one JSONL per source to data/raw/, records as {"text", "source", "id"}.
"""
import gzip
import json
import lzma
import random
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

MIN_TEXT_CHARS = 200
GB = 1024**3

POL = "pile-of-law/pile-of-law"

# Sized so a 30-day run at ~25,400 tok/s (150M, bf16, sparse_prediction, full torch.compile) sees
# ~66B tokens = ~2.1 epochs -- under the ~4-epoch ceiling past which repeats underperform fresh tokens.
#
# The two domains stage-2 is actually evaluated on (contracts, financial filings) are MAXED OUT at
# their public ceilings: essentially all the public financial document text that exists at scale is
# EDGAR + SEC/tax rulings, and contracts run out at Atticus + albertvillanova. Everything else on the
# Hub labelled "financial" turned out to be a re-cut of the same 10-Ks.
#
# lexlms/lex_files was evaluated and REJECTED: its us_contracts split (9.2GB) is the same corpus as
# albertvillanova/legal_contracts, its courtlistener split overlaps the Caselaw Access Project, and the
# rest is UK/EU/CA/India law -- non-US, while every stage-2 eval (LEDGAR, CUAD, FinQA, EDGAR) is US.
#
# GENERAL DATA (20%): the model is trained from scratch, so unlike a domain fine-tune it has no
# pretrained "worldview" to inherit -- and real retrieval queries arrive in plain English, not
# contract prose. 20% FineWeb-Edu (~13B general tokens seen, more than GPT-2 ever saw) buys general
# linguistic competence at the cost of ~13B domain tokens. Precedent: BloombergGPT deliberately mixed
# ~50/50 financial/general; we stay domain-heavy because this is a retrieval model evaluated purely on
# legal/financial tasks.
TARGETS = {
    "legal_caselaw": 35 * GB,        # common-pile/caselaw_access_project (CC0)
    "financial_edgar": 32 * GB,      # eloukas/edgar-corpus 10-Ks -- MAX available
    "financial_pol": 8 * GB,         # pile-of-law: edgar + sec + taxrulings -- MAX available
    "legal_contracts": 32 * GB,      # pile-of-law atticus + albertvillanova/legal_contracts -- MAX
    "legal_regulations": 8 * GB,     # pile-of-law: state_code, us_bills, cfr, uscode, federal_register
    "general_web": 29 * GB,          # HuggingFaceFW/fineweb-edu (sample/10BT) -- 20% of final corpus
}

POL_SUBSETS = {
    "financial_pol": ["edgar", "sec", "taxrulings"],
    "legal_regulations": ["state_code", "us_bills", "cfr", "uscode", "federal_register"],
}


def write_source(name, record_iter, target_bytes):
    out_path = RAW_DIR / f"{name}.jsonl"
    written = n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for rec_id, text in record_iter:
            text = text.strip()
            if len(text) < MIN_TEXT_CHARS:
                continue
            line = json.dumps({"text": text, "source": name, "id": rec_id}, ensure_ascii=False)
            out.write(line + "\n")
            written += len(line.encode("utf-8"))
            n += 1
            if n % 20000 == 0:
                print(f"  [{name}] {written/GB:.2f}GB, {n} records", flush=True)
            if target_bytes and written >= target_bytes:
                break
    print(f"[{name}] done: {written/GB:.2f}GB, {n} records", flush=True)


def iter_caselaw(api, target_bytes):
    info = api.dataset_info("common-pile/caselaw_access_project", files_metadata=True)
    shards = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(".jsonl.gz"))
    est = 0
    for shard in shards:
        if est >= target_bytes:
            return
        path = hf_hub_download("common-pile/caselaw_access_project", shard, repo_type="dataset")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                text = rec.get("text", "")
                est += len(text.encode("utf-8"))
                yield rec.get("id", shard), text
                if est >= target_bytes:
                    return


def iter_pol(api, subsets, target_bytes):
    """Stream one or more pile-of-law subsets, round-robin across subsets for an even mix."""
    info = api.dataset_info(POL, files_metadata=True)
    shards_by_subset = {}
    for sub in subsets:
        shards = sorted(
            s.rfilename
            for s in info.siblings
            if s.rfilename.startswith(f"data/train.{sub}.") and s.rfilename.endswith(".jsonl.xz")
        )
        if shards:
            shards_by_subset[sub] = shards
    est = 0
    for sub, shards in shards_by_subset.items():
        for shard in shards:
            if est >= target_bytes:
                return
            try:
                path = hf_hub_download(POL, shard, repo_type="dataset")
            except Exception as e:
                print(f"  [skip {shard}: {e}]", flush=True)
                continue
            with lzma.open(path, "rt", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    try:
                        text = json.loads(line).get("text", "")
                    except json.JSONDecodeError:
                        continue
                    est += len(text.encode("utf-8"))
                    yield f"{sub}:{i}", text
                    if est >= target_bytes:
                        return


SECTION_KEYS = [
    "section_1", "section_1A", "section_1B", "section_2", "section_3", "section_4", "section_5",
    "section_6", "section_7", "section_7A", "section_8", "section_9", "section_9A", "section_9B",
    "section_10", "section_11", "section_12", "section_13", "section_14", "section_15",
]


def iter_edgar(api, target_bytes, seed=0):
    info = api.dataset_info("eloukas/edgar-corpus", files_metadata=True)
    files = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith("train.jsonl"))
    random.Random(seed).shuffle(files)  # spread across years; download lazily
    est = 0
    for fname in files:
        if est >= target_bytes:
            return
        path = hf_hub_download("eloukas/edgar-corpus", fname, repo_type="dataset")
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                parts = [rec.get(k) or "" for k in SECTION_KEYS]
                text = "\n\n".join(p for p in parts if p.strip())
                est += len(text.encode("utf-8"))
                yield rec.get("filename", ""), text
                if est >= target_bytes:
                    return


def iter_hf_text(name, target_bytes, config=None, revision=None, field="text"):
    """Read any HF dataset's parquet shards directly.

    Deliberately NOT using `datasets` streaming: abandoning a streaming iterator mid-shard (which we do
    every time a source hits its byte budget) leaves its background threads in a bad state and the
    interpreter dies at exit with `PyGILState_Release: thread state must be current`. Fine in a probe,
    fatal in a 10-hour unattended corpus build. Reading parquet with pyarrow has no such threads.
    """
    import pyarrow.parquet as pq

    api = HfApi()
    files = [f for f in api.list_repo_files(name, repo_type="dataset", revision=revision) if f.endswith(".parquet")]
    if config:
        scoped = [f for f in files if f.startswith(f"{config}/") or f"/{config}/" in f]
        files = scoped or files
    files = sorted(f for f in files if "train" in f) or sorted(files)
    if not files:
        raise SystemExit(f"no parquet shards found for {name}")

    est = 0
    for shard in files:
        if est >= target_bytes:
            return
        path = hf_hub_download(name, shard, repo_type="dataset", revision=revision)
        table = pq.read_table(path, columns=[field])
        for i, val in enumerate(table.column(field)):
            text = val.as_py() or ""
            est += len(text.encode("utf-8"))
            yield f"{name}:{shard}:{i}", text
            if est >= target_bytes:
                return


def iter_contracts(api, target_bytes):
    """Contracts from BOTH public sources -- this domain is eval-critical (LEDGAR/CUAD) and scarce,
    so we exhaust it: pile-of-law's Atticus set plus albertvillanova/legal_contracts."""
    half = target_bytes // 2
    yield from iter_pol(api, ["atticus_contracts"], target_bytes - half)
    yield from iter_hf_text("albertvillanova/legal_contracts", half, revision="refs/convert/parquet")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="build only these sources (lets us add a source without rewriting the rest)")
    args = ap.parse_args()
    want = set(args.only) if args.only else set(TARGETS)
    api = HfApi()

    def wanted(name):
        return name in want

    if wanted("legal_caselaw"):
        print("=== legal_caselaw (Caselaw Access Project, CC0) ===")
        write_source("legal_caselaw", iter_caselaw(api, TARGETS["legal_caselaw"]), TARGETS["legal_caselaw"])

    if wanted("financial_edgar"):
        print("=== financial_edgar (EDGAR-CORPUS 10-Ks) ===")
        write_source("financial_edgar", iter_edgar(api, TARGETS["financial_edgar"]), TARGETS["financial_edgar"])

    if wanted("legal_contracts"):
        print("=== legal_contracts (pile-of-law atticus + albertvillanova/legal_contracts) ===")
        write_source("legal_contracts", iter_contracts(api, TARGETS["legal_contracts"]), TARGETS["legal_contracts"])

    if wanted("financial_pol"):
        print("=== financial_pol (pile-of-law: edgar, sec, taxrulings) ===")
        write_source(
            "financial_pol",
            iter_pol(api, POL_SUBSETS["financial_pol"], TARGETS["financial_pol"]),
            TARGETS["financial_pol"],
        )

    if wanted("legal_regulations"):
        print("=== legal_regulations (pile-of-law: state_code, us_bills, cfr, uscode, federal_register) ===")
        write_source(
            "legal_regulations",
            iter_pol(api, POL_SUBSETS["legal_regulations"], TARGETS["legal_regulations"]),
            TARGETS["legal_regulations"],
        )

    if wanted("general_web"):
        print("=== general_web (FineWeb-Edu sample/10BT) ===")
        write_source(
            "general_web",
            iter_hf_text("HuggingFaceFW/fineweb-edu", TARGETS["general_web"], config="sample/10BT"),
            TARGETS["general_web"],
        )

    print("\n=== corpus ===")
    total = 0
    for n in TARGETS:
        p = RAW_DIR / f"{n}.jsonl"
        if p.exists():
            gb = p.stat().st_size / GB
            total += gb
            print(f"  {n:<20} {gb:>6.2f} GB")
    print(f"  {'TOTAL':<20} {total:>6.2f} GB  (~{total*0.217:.1f}B tokens)")


if __name__ == "__main__":
    main()
