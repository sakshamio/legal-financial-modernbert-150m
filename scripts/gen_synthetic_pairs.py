"""Generate synthetic (query -> passage) training pairs with a local LLM.

WHY. Our Stage-2 embedder scored 0.076 on FiQA2018 (asymmetric question -> answer retrieval) because
its training pairs are mostly SYMMETRIC clause-to-clause matches (LEDGAR) and short spans. It never
learned to place a short natural-language QUESTION near the long passage that answers it. That is the
single largest gap between us (0.285 avg) and MiniLM (0.575), and the literature is unanimous on the
fix: synthesise asymmetric queries with an LLM.

  E5-mistral (Wang 2024)   -- a top embedder trained almost entirely on GPT-4-generated triples.
  Gecko (Google 2024)      -- generate a query from a passage, then RE-LABEL positives/negatives.
  Qwen3-Embedding (2025)   -- our teacher; built on large-scale LLM-synthetic data + SFT + merging.

Design choices taken from that literature:
  * ASYMMETRIC TASK TAXONOMY (E5-mistral): we generate several query *types* per passage
    -- a specific factual question, a broader topical query, and a keyword-style query -- so the model
    learns short->long matching at multiple granularities, not one narrow style.
  * DOMAIN-CONDITIONED PROMPTS: a financial-analyst persona for filings, a lawyer persona for
    contracts/caselaw/regulations. The query vocabulary has to match how real users search.
  * OUTPUT is the SAME pair format as build_pairs.py, so hard negatives are then mined by gopairs
    (BM25, already 2.4s for the whole set) exactly as for the human-labelled pairs. LLM-generated
    hard negatives (Shao 2025) are a later upgrade; BM25 within a large synthetic set is a strong
    start and far cheaper.

COMPUTE. Generation is the one expensive step and it is one-shot + cacheable. It runs against an
OpenAI-compatible server that sparkrun stands up (vLLM / sglang), NOT an in-process model:

  * a much stronger generator than we could load by hand -- qwen3.6-35b-a3b-fp8 is a 35B MoE with
    only ~3B ACTIVE params per token. On this 273 GB/s box, decode is memory-bound, so reading 3B
    active params/token instead of a dense 27B is a large speed win: MoE sparsity is exactly right
    for inference here (it is the wrong choice for TRAINING on this box, but that is a different
    regime). The 'mtp' variant adds multi-token prediction for further decode speedup.
  * vLLM continuous batching gives far higher throughput than a hand-rolled transformers loop, so we
    just fire concurrent HTTP requests and let the server schedule them.

Like every GPU job here, the SERVER must not share the GPU with training (vLLM grabs ~80% of memory
and would OOM the trainer, which has happened twice). So the flow is: pause Stage 1 -> sparkrun run
<model> -> this script -> sparkrun stop -> resume Stage 1.

    # 1. stand up the generator (in its own GPU window):
    sparkrun run @official/qwen3.6-35b-a3b-fp8-mtp-vllm --port 8000
    # 2. generate:
    python scripts/gen_synthetic_pairs.py --n 200000 --base-url http://127.0.0.1:8000/v1
    # smoke test against any running endpoint (e.g. the tiny qwen3-1.7b recipe):
    python scripts/gen_synthetic_pairs.py --n 8 --base-url http://127.0.0.1:8000/v1 --concurrency 4
"""
import argparse
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEDUP_DIR = PROJECT_DIR / "data" / "raw_dedup"
OUT_DIR = PROJECT_DIR / "data" / "pairs_synth"

# Same domain mixture as pretraining, but general_web is dropped: synthetic RETRIEVAL queries only
# make sense over the enterprise domains we are actually building for.
MIXTURE = {
    "financial_edgar": 0.34,
    "financial_pol": 0.10,
    "legal_caselaw": 0.30,
    "legal_contracts": 0.18,
    "legal_regulations": 0.08,
}

PERSONA = {
    "financial_edgar": "a financial analyst researching a company's SEC filings",
    "financial_pol": "a financial analyst researching regulatory and tax filings",
    "legal_caselaw": "a litigator researching case law",
    "legal_contracts": "a transactional lawyer reviewing contracts",
    "legal_regulations": "a compliance lawyer researching statutes and regulations",
}

# E5-mistral-style granularity: three query types per passage so the model learns short->long matching
# at more than one style. Each returns ONE line.
QUERY_TYPES = [
    ("specific", "a specific factual question that this passage directly answers"),
    ("topical", "a broader topical search query (a phrase, not a full sentence) about what this passage covers"),
    ("scenario", "a realistic question phrased the way a practitioner would actually type it into a search box"),
]


def sample_passages(n, min_chars, max_chars, seed):
    """Random-byte-seek sampling from the deduped corpus -- the same bounded approach as gosample,
    but we only need a few hundred thousand, so a light Python version is fine here."""
    rng = random.Random(seed)
    out = []
    for domain, frac in MIXTURE.items():
        path = DEDUP_DIR / f"{domain}.jsonl"
        if not path.exists():
            continue
        want = int(n * frac)
        size = path.stat().st_size
        got = 0
        with open(path, "rb") as f:
            tries = 0
            while got < want and tries < want * 20 + 500:
                tries += 1
                f.seek(rng.randint(0, max(1, size - max_chars * 4)))
                f.readline()
                line = f.readline()
                if not line:
                    continue
                try:
                    text = json.loads(line).get("text", "")
                except Exception:
                    continue
                if len(text) < min_chars:
                    continue
                s = rng.randint(0, max(0, len(text) - max_chars))
                chunk = text[s : s + max_chars].strip()
                # snap to sentence-ish boundaries
                if s > 0 and " " in chunk[:40]:
                    chunk = chunk[chunk.find(" ") + 1 :]
                if len(chunk) >= min_chars:
                    out.append({"domain": domain, "passage": chunk})
                    got += 1
    rng.shuffle(out)
    return out


def build_messages(persona, passage, qtype_desc):
    return [
        {"role": "system",
         "content": "You write realistic search queries. Output ONLY the query text, no preamble, "
                    "no quotes, no explanation, no reasoning."},
        {"role": "user",
         "content": f"You are {persona}. Read the passage below and write {qtype_desc}. The query "
                    f"must be answerable from the passage but must NOT copy long phrases from it "
                    f"verbatim.\n\nPassage:\n\"\"\"\n{passage[:1600]}\n\"\"\"\n\nQuery:"},
    ]


def clean_query(text):
    text = text.strip().strip('"').strip()
    text = re.sub(r"^(query|question|search)\s*[:\-]\s*", "", text, flags=re.I)
    text = text.split("\n")[0].strip()  # first line only
    return text


def generate_one(client, model_name, messages, max_new_tokens):
    """One chat completion against the OpenAI-compatible server. Returns cleaned query or None."""
    try:
        r = client.post("/chat/completions", json={
            "model": model_name,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            # Qwen3 'thinking' models will emit long reasoning unless disabled; both vLLM and sglang
            # accept this passthrough. Harmless for non-thinking models.
            "chat_template_kwargs": {"enable_thinking": False},
        })
        r.raise_for_status()
        return clean_query(r.json()["choices"][0]["message"]["content"])
    except Exception:
        return None


def discover_model(client):
    """Ask the server which model it is serving (so we don't hardcode the name)."""
    try:
        return client.get("/models").json()["data"][0]["id"]
    except Exception:
        return "default"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000, help="passages to sample (x query-types = pairs)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1",
                    help="OpenAI-compatible endpoint stood up by sparkrun")
    ap.add_argument("--model", default=None, help="model id (default: auto-discover from /models)")
    ap.add_argument("--concurrency", type=int, default=64, help="in-flight requests; vLLM batches them")
    ap.add_argument("--min-chars", type=int, default=500)
    ap.add_argument("--max-chars", type=int, default=1600)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--query-types", type=int, default=2, help="how many of the 3 types per passage")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    client = httpx.Client(base_url=args.base_url, timeout=120.0)
    try:
        client.get("/models")
    except Exception as e:
        raise SystemExit(f"no server at {args.base_url} ({e}). Start one first, e.g.:\n"
                         f"  sparkrun run @official/qwen3.6-35b-a3b-fp8-mtp-vllm --port 8000")
    model_name = args.model or discover_model(client)
    print(f"server: {args.base_url}  model: {model_name}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"sampling {args.n:,} passages ...")
    passages = sample_passages(args.n, args.min_chars, args.max_chars, args.seed)
    print(f"  got {len(passages):,}")

    qtypes = QUERY_TYPES[: args.query_types]
    jobs = [(pi, qi) for pi in range(len(passages)) for qi in range(len(qtypes))]
    print(f"generating {len(jobs):,} queries ({len(qtypes)} types x {len(passages):,} passages) "
          f"at concurrency {args.concurrency} ...")

    out_path = out_dir / "train.jsonl"
    lock = threading.Lock()
    t0 = time.time()
    counts = {"written": 0, "done": 0}

    def work(job):
        pi, qi = job
        p = passages[pi]
        q = generate_one(client, model_name, build_messages(PERSONA[p["domain"]], p["passage"],
                                                             qtypes[qi][1]), args.max_new_tokens)
        with lock:
            counts["done"] += 1
            if q and 8 <= len(q) <= 300:
                fout.write(json.dumps({
                    "anchor": f"[QUERY] {q}",
                    "positive": f"[PASSAGE] {p['passage']}",
                    "label": f"synth-{p['domain']}-{pi}",  # unique per passage: its queries are its only positives
                    "source": f"synth_{p['domain']}",
                    "qtype": qtypes[qi][0],
                }) + "\n")
                counts["written"] += 1
            if counts["done"] % 500 == 0:
                el = time.time() - t0
                rate = counts["done"] / max(el, 1e-9)
                eta = (len(jobs) - counts["done"]) / max(rate, 1e-9) / 3600
                print(f"  {counts['done']:,}/{len(jobs):,}  {rate:.1f} q/s  ETA {eta:.1f}h  "
                      f"(kept {counts['written']:,})", flush=True)

    with open(out_path, "w") as fout:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(work, jobs))

    print(f"\ndone: {counts['written']:,} synthetic pairs -> {out_path}  ({(time.time()-t0)/60:.1f} min)")
    print("next: feed through gopairs for BM25 hard negatives, then blend into Stage 2.")


if __name__ == "__main__":
    main()
