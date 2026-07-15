"""Taxonomy-driven synthetic training data for the financial/legal world.

WHY THIS, NOT CORPUS-GROUNDED QUERIES. Our real corpus (EDGAR 10-Ks, caselaw, pile-of-law) is narrow
and skewed: it barely contains the document types that matter for enterprise retrieval -- NDAs,
investment management agreements, credit facilities, M&A term sheets, fintech/SaaS contracts. So
generating queries over it cannot teach the model those document types. Instead we SYNTHESISE both
sides across a broad taxonomy, the way E5-mistral / Gecko / Qwen3-Embedding are built.

ONE CALL -> MANY TRIPLES. At 5M-pair scale, per-call overhead dominates, so each LLM call produces a
full training cluster as JSON:

    { "passage": "<a realistic clause/section from the sampled document type>",
      "queries": ["<specific factual>", "<topical keyword>", "<practitioner-phrased>"],
      "hard_negative": "<a passage from a RELATED but different clause/doc that looks similar but
                        does NOT answer the queries>" }

That yields len(queries) training triples (query, passage, hard_negative) per call -- ~3x fewer calls
than generating one triple at a time. The LLM-generated hard negative follows Shao 2025 / Kim & Baek
2025 (generated negatives beat BM25 for this), and gopairs can still add BM25 negatives across the
whole set afterwards.

VARIETY is the explicit goal, so we sample independently along several axes and inject them into the
prompt: document category -> subtype -> a focus clause/topic, plus sector, jurisdiction, party type,
and deal context. The cartesian product is enormous, and temperature adds within-cell diversity.

RESUMABLE. 5M pairs is a multi-day generation; it appends to a jsonl and records how many cluster
calls have completed, so a killed run continues instead of restarting.

Runs against an OpenAI-compatible server stood up by sparkrun (see gen_synthetic_pairs.py for the
server rationale -- 35B MoE, memory-bound decode favours the sparsity). Never shares the GPU with
training.

    sparkrun run @official/qwen3.6-35b-a3b-fp8-mtp-vllm --port 8000
    python scripts/gen_synthetic_taxonomy.py --target-pairs 5000000 --base-url http://127.0.0.1:8000/v1
"""
import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from pathlib import Path

import httpx

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "data" / "pairs_synth_taxonomy"

# --------------------------------------------------------------------------------------------------
# The taxonomy. Category -> document subtypes -> the clause/topic each subtype is generated around.
# Deliberately broad: this is what buys variety across the financial/legal world.
# --------------------------------------------------------------------------------------------------
TAXONOMY = {
    "M&A": {
        "docs": ["merger agreement", "share purchase agreement", "asset purchase agreement",
                 "letter of intent", "term sheet", "disclosure schedule", "escrow agreement",
                 "transition services agreement", "stockholders' representative agreement"],
        "clauses": ["representations and warranties", "material adverse change", "purchase price adjustment",
                    "earnout", "indemnification and escrow", "closing conditions", "non-compete and non-solicit",
                    "termination and break fee", "working capital adjustment", "R&W insurance",
                    "covenants between signing and closing", "disclosure schedule exceptions"],
    },
    "Investment Management": {
        "docs": ["investment management agreement", "limited partnership agreement",
                 "private placement memorandum", "subscription agreement", "side letter",
                 "sub-advisory agreement", "separately managed account agreement", "fund of funds agreement"],
        "clauses": ["management fee", "carried interest and distribution waterfall", "high-water mark",
                    "key person provision", "investment restrictions and guidelines", "redemption and lock-up",
                    "most favored nation", "GP clawback", "valuation policy", "capital calls and defaults",
                    "co-investment rights", "fee offset and expense allocation"],
    },
    "Lending & Credit": {
        "docs": ["credit agreement", "term loan facility", "revolving credit facility", "security agreement",
                 "guaranty", "intercreditor agreement", "promissory note", "mezzanine note purchase agreement"],
        "clauses": ["financial covenants", "negative covenants", "events of default", "borrowing base",
                    "affirmative covenants", "mandatory prepayment", "collateral and perfection",
                    "representations and warranties", "conditions precedent", "cash sweep",
                    "make-whole and prepayment premium", "MFN pricing"],
    },
    "NDAs & Confidentiality": {
        "docs": ["mutual NDA", "one-way NDA", "M&A due-diligence NDA", "clean team agreement",
                 "data protection addendum"],
        "clauses": ["definition of confidential information", "permitted disclosures", "residuals clause",
                    "term and survival", "return or destruction of information", "non-solicitation of employees",
                    "standstill", "injunctive relief", "carve-outs from confidentiality"],
    },
    "Fintech & Technology": {
        "docs": ["SaaS subscription agreement", "API license agreement", "payment processing agreement",
                 "banking-as-a-service agreement", "data sharing agreement", "embedded finance partnership",
                 "digital asset custody agreement", "master services agreement", "reseller agreement"],
        "clauses": ["service level agreement", "data security and breach notification", "limitation of liability",
                    "indemnification", "intellectual property ownership", "fees and payment terms",
                    "regulatory compliance and licensing", "termination for convenience", "audit rights",
                    "acceptable use", "uptime and credits", "sub-processor obligations"],
    },
    "Corporate & Securities": {
        "docs": ["shareholders agreement", "convertible note", "SAFE agreement", "warrant agreement",
                 "stock option plan", "underwriting agreement", "registration rights agreement",
                 "voting agreement", "prospectus", "offering memorandum"],
        "clauses": ["liquidation preference", "anti-dilution protection", "drag-along and tag-along",
                    "pre-emptive rights", "board composition", "protective provisions", "conversion mechanics",
                    "information rights", "vesting and acceleration", "lock-up", "use of proceeds", "risk factors"],
    },
    "Derivatives & Structured": {
        "docs": ["ISDA master agreement", "credit support annex", "swap confirmation", "repo agreement",
                 "securities lending agreement", "structured note term sheet"],
        "clauses": ["netting and set-off", "collateral and margin", "events of default and termination",
                    "close-out amount", "eligible collateral and haircuts", "cross-default",
                    "calculation agent", "payment netting", "credit events"],
    },
    "Real Estate & Project Finance": {
        "docs": ["commercial lease", "purchase and sale agreement", "commercial real estate loan agreement",
                 "project finance credit agreement", "development agreement", "ground lease"],
        "clauses": ["rent and escalation", "operating expenses and CAM", "assignment and subletting",
                    "casualty and condemnation", "completion guaranty", "debt service coverage covenant",
                    "reserves and cash management", "permitted transfers", "SNDA"],
    },
    "Regulatory & Compliance": {
        "docs": ["Form ADV", "compliance manual", "KYC/AML policy", "code of ethics", "proxy statement",
                 "10-K risk factors section", "MD&A section", "SEC comment letter response"],
        "clauses": ["conflicts of interest", "custody rule compliance", "best execution", "insider trading policy",
                    "suspicious activity monitoring", "advisory fee disclosure", "material risk factors",
                    "liquidity and capital resources", "critical accounting estimates", "related party transactions"],
    },
    "Asset Management Ops": {
        "docs": ["distribution agreement", "transfer agency agreement", "custody agreement",
                 "prime brokerage agreement", "model portfolio agreement", "administration agreement"],
        "clauses": ["standard of care", "indemnification and liability", "fees and expense reimbursement",
                    "termination and transition", "reporting and recordkeeping", "rehypothecation",
                    "margin and financing", "NAV calculation and error correction", "proxy voting"],
    },
}

SECTORS = ["technology", "healthcare", "energy", "financial services", "real estate", "manufacturing",
           "consumer/retail", "telecommunications", "biotech/pharma", "infrastructure", "media",
           "renewable energy", "private equity portfolio company", "hedge fund", "insurance"]
JURISDICTIONS = ["Delaware", "New York", "English law", "California", "Cayman Islands", "Luxembourg",
                 "Texas", "Ontario", "Singapore", "Delaware LLC", "Nevada"]
PARTIES = ["a private equity sponsor", "an institutional asset manager", "a growth-stage startup",
           "a commercial bank", "a family office", "a public company", "a fintech company",
           "a pension fund", "a sovereign wealth fund", "a hedge fund", "a venture capital firm",
           "an insurance company", "a REIT", "a broker-dealer"]
CONTEXTS = ["a cross-border transaction", "a distressed situation", "a first-time fund",
            "a syndicated deal", "a bilateral negotiation", "a highly negotiated side letter",
            "a middle-market deal", "a large-cap transaction", "an amendment and restatement",
            "a bespoke structure", "a standard-form agreement"]

QUERY_STYLES = ("a specific factual question a practitioner would ask that this passage answers",
                "a short topical keyword search query (a phrase, not a sentence) about this passage's subject",
                "a natural question phrased the way someone would actually type it into a search box")


def build_messages(category, subtype, clause, sector, jur, party, ctx, n_queries):
    persona = "an expert transactional attorney and financial analyst"
    system = (
        "You generate realistic financial/legal training data. You always respond with a single "
        "valid JSON object and nothing else -- no markdown, no code fences, no commentary."
    )
    user = f"""You are {persona}. Produce one realistic training cluster for a retrieval model.

Setting (use it to make the text specific and varied):
- Document category: {category}
- Document type: {subtype}
- Clause / topic focus: {clause}
- Sector: {sector}
- Governing law: {jur}
- A principal party: {party}
- Context: {ctx}

Return a JSON object with exactly these keys:
- "passage": a realistic 90-160 word excerpt from the "{subtype}" focused on "{clause}". Write it the
  way a real {subtype} reads -- defined terms, cross-references, appropriate legal/financial register.
  Do NOT include a heading or the document title; just the operative text.
- "queries": a list of {n_queries} DISTINCT search queries that this passage answers. Vary them:
  ({'; '.join(QUERY_STYLES[:n_queries])}). Do not copy long phrases from the passage verbatim.
- "hard_negative": a realistic 90-160 word excerpt that is SIMILAR in topic/document type (a related
  clause or a neighbouring provision) but that does NOT actually answer the queries -- a plausible
  wrong retrieval result.

Output only the JSON object."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_cluster(text):
    """Pull the JSON object out of the model output, tolerant of stray prose or code fences."""
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1] if text.count("```") >= 2 else text
        text = text.lstrip("json").strip()
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        obj = json.loads(text[a : b + 1])
    except Exception:
        return None
    p = (obj.get("passage") or "").strip()
    qs = obj.get("queries") or []
    hn = (obj.get("hard_negative") or "").strip()
    qs = [q.strip() for q in qs if isinstance(q, str) and 8 <= len(q.strip()) <= 300]
    if len(p) < 120 or not qs:
        return None
    return {"passage": p, "queries": qs, "hard_negative": hn if len(hn) >= 120 else None}


def sample_cell(rng):
    cat = rng.choice(list(TAXONOMY))
    sub = rng.choice(TAXONOMY[cat]["docs"])
    clause = rng.choice(TAXONOMY[cat]["clauses"])
    return dict(category=cat, subtype=sub, clause=clause, sector=rng.choice(SECTORS),
                jur=rng.choice(JURISDICTIONS), party=rng.choice(PARTIES), ctx=rng.choice(CONTEXTS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-pairs", type=int, default=5_000_000)
    ap.add_argument("--queries-per-cluster", type=int, default=3)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    client = httpx.Client(base_url=args.base_url, timeout=180.0)
    try:
        # tolerate a brief blip at startup rather than exiting immediately (the supervisor would just
        # re-run us, which is wasteful)
        ok = False
        for _ in range(6):
            try:
                client.get("/models"); ok = True; break
            except Exception:
                time.sleep(10)
        if not ok:
            raise SystemExit(f"no server at {args.base_url} after retries")
    except Exception as e:
        raise SystemExit(f"no server at {args.base_url} ({e})")
    model_name = args.model or client.get("/models").json()["data"][0]["id"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train.jsonl"
    prog_path = out_dir / "progress.json"

    # RESUME: count what is already on disk so a restart continues.
    have_pairs = 0
    if out_path.exists():
        with open(out_path) as f:
            have_pairs = sum(1 for _ in f)
    n_clusters_target = max(0, (args.target_pairs - have_pairs)) // args.queries_per_cluster + 1
    print(f"server {args.base_url} | model {model_name}")
    print(f"have {have_pairs:,} pairs; targeting {args.target_pairs:,} "
          f"-> ~{n_clusters_target:,} more cluster calls at {args.queries_per_cluster} q each")

    rng = random.Random(args.seed + have_pairs)  # different cells on resume
    lock = threading.Lock()
    t0 = time.time()
    counts = {"pairs": have_pairs, "calls": 0, "fail": 0}

    def work(_):
        # Bulletproof: NOTHING may propagate out of here, or ThreadPoolExecutor.map re-raises it in
        # the driving list() and kills the whole run. A transient server error must cost one cluster,
        # never the job.
        try:
            _work(_)
        except Exception:
            with lock:
                counts["calls"] += 1
                counts["fail"] += 1

    def _work(_):
        cell = sample_cell(random.Random(rng.random()))
        msgs = build_messages(**cell, n_queries=args.queries_per_cluster)
        try:
            r = client.post("/chat/completions", json={
                "model": model_name, "messages": msgs, "max_tokens": args.max_tokens,
                "temperature": 0.9, "top_p": 0.95,
                "chat_template_kwargs": {"enable_thinking": False},
            })
            r.raise_for_status()
            cl = parse_cluster(r.json()["choices"][0]["message"]["content"])
        except Exception:
            cl = None
        with lock:
            counts["calls"] += 1
            if cl is None:
                counts["fail"] += 1
            else:
                lab = f"synth-{cell['category']}-{cell['subtype']}-{counts['calls']}"
                for q in cl["queries"]:
                    rec = {"anchor": f"[QUERY] {q}", "positive": f"[PASSAGE] {cl['passage']}",
                           "label": lab, "source": f"synth_{cell['category'].split()[0].lower()}",
                           "meta": {k: cell[k] for k in ("category", "subtype", "clause")}}
                    if cl["hard_negative"]:
                        rec["negative_0"] = f"[PASSAGE] {cl['hard_negative']}"
                    fout.write(json.dumps(rec) + "\n")
                    counts["pairs"] += 1
            if counts["calls"] % 200 == 0:
                el = time.time() - t0
                cps = counts["calls"] / max(el, 1e-9)
                pps = (counts["pairs"] - have_pairs) / max(el, 1e-9)
                eta = (args.target_pairs - counts["pairs"]) / max(pps, 1e-9) / 3600
                fout.flush()
                prog_path.write_text(json.dumps({"pairs": counts["pairs"], "calls": counts["calls"],
                                                 "fail_rate": counts["fail"] / counts["calls"]}))
                print(f"  pairs {counts['pairs']:,}/{args.target_pairs:,} | {pps:.0f} pairs/s "
                      f"({cps:.0f} calls/s) | fail {counts['fail']/counts['calls']:.1%} | ETA {eta:.1f}h",
                      flush=True)

    with open(out_path, "a") as fout:
        # BOUNDED submission. The previous `pool.map(work, jobs())` over an UNBOUNDED generator was a
        # 60GB memory leak: pool.map eagerly pulls from the generator and queues pending futures far
        # faster than the slow API calls drain them, so millions of task objects pile up. That memory
        # blowout is what tripped earlyoom (configured --prefer python|vllm), which then killed the
        # vLLM server -- the real cause of every "crash". Keep at most 2x concurrency in flight.
        max_inflight = args.concurrency * 2
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            inflight = set()
            while counts["pairs"] < args.target_pairs:
                while len(inflight) < max_inflight and counts["pairs"] < args.target_pairs:
                    inflight.add(pool.submit(work, None))
                done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for f in inflight:
                f.result()

    print(f"\ndone: {counts['pairs']:,} pairs -> {out_path}  ({(time.time()-t0)/3600:.1f}h, "
          f"fail {counts['fail']/max(counts['calls'],1):.1%})")


if __name__ == "__main__":
    main()
