"""Sample text chunks for geometry distillation (scripts/distill_geometry.py).

Distillation needs raw TEXT, not packed token ids, because the teacher has its own tokenizer -- that
is exactly what makes the approach tokenizer-agnostic and lets our custom legal vocab survive. So we
sample from the deduped jsonl, not from data/packed/train.bin.

BOUNDED-WINDOW SAMPLING
-----------------------
The obvious implementation -- seek to a random offset, readline, json.loads, take a chunk -- is a
trap here. Record sizes in this corpus are wildly skewed:

    legal_caselaw       median record    15 KB
    financial_edgar     median record   179 KB
    legal_regulations   median record  3889 KB   (mean 10 MB)

so that implementation reads and JSON-parses a 10 MB document to extract 1,800 characters, using
0.02% of the bytes it moved. Measured: 180 chunks/s => 7.7 hours for 5M chunks.

Instead we read a small BOUNDED WINDOW at the random offset and recover the text from inside the
JSON string without ever parsing the enclosing record. Records are `{"text":"...","source":...}` with
`text` first, so a window landing mid-record is almost entirely escaped text; the only structure we
must respect is the first UNESCAPED `"`, which ends the value. Cost becomes O(chunks), independent of
document size.

This is also distributionally BETTER: uniform-by-byte sampling weights documents by length, matching
the token distribution the encoder was actually pretrained on. One-chunk-per-document would have
over-sampled short documents.

CPU + random IO only, no GPU and no memory bandwidth. Safe to run alongside training.

    python scripts/sample_distill_corpus.py --n-chunks 5000000
"""
import argparse
import hashlib
import json
import re
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

# control chars except \n \t \r -- artefacts of trimming a torn escape at a window edge
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEDUP_DIR = PROJECT_DIR / "data" / "raw_dedup"
OUT_DIR = PROJECT_DIR / "data" / "distill"

# Same mixture as the pretraining corpus (fractions of deduped bytes).
MIXTURE = {
    "financial_edgar": 0.24,
    "financial_pol": 0.07,
    "general_web": 0.24,
    "legal_caselaw": 0.29,
    "legal_contracts": 0.12,
    "legal_regulations": 0.04,
}


def first_unescaped_quote(s):
    """Index of the first `"` not preceded by an odd run of backslashes, else -1."""
    i = 0
    while True:
        i = s.find('"', i)
        if i < 0:
            return -1
        b = 0
        while i - 1 - b >= 0 and s[i - 1 - b] == "\\":
            b += 1
        if b % 2 == 0:
            return i
        i += 1


def unescape(frag):
    """json.loads a fragment of a JSON string value, trimming partial escapes at the edges."""
    for a in range(3):
        for b in range(3):
            piece = frag[a : len(frag) - b] if b else frag[a:]
            try:
                return json.loads('"' + piece + '"')
            except Exception:
                continue
    return None


def sample_file(job):
    domain, n_want, min_chars, max_chars, seed = job
    path = DEDUP_DIR / f"{domain}.jsonl"
    if not path.exists():
        return domain, [], 0
    size = path.stat().st_size
    win = max_chars * 3  # escapes expand; 3x gives ample slack to land max_chars of real text
    rng = np.random.default_rng(seed)

    out, misses, seen = [], 0, set()
    with open(path, "rb") as f:
        while len(out) < n_want and misses < n_want * 10 + 500:
            f.seek(int(rng.integers(0, max(1, size - win))))
            raw = f.read(win)
            if not raw:
                misses += 1
                continue
            s = raw.decode("utf-8", errors="ignore")

            # if we landed at a record start, skip past the key into the value
            k = s.find('{"text":"')
            if k >= 0:
                s = s[k + 9 :]
            # the value ends at the first unescaped quote (then comes ,"source":...)
            q = first_unescaped_quote(s)
            if q >= 0:
                s = s[:q]

            text = unescape(s)
            if not text or len(text) < min_chars:
                misses += 1
                continue
            # window edges can leave a torn escape or a torn utf-8 sequence. Measured at <1% of
            # chunks, but there is no reason to hand the teacher corrupt bytes.
            if "�" in text or re.search(r'\\[nrtu"]', text):
                misses += 1
                continue
            text = CONTROL.sub("", text)

            # snap to whitespace so the teacher never sees a half-word at either end
            text = text[:max_chars]
            c = text.find(" ")
            if 0 <= c < 40:
                text = text[c + 1 :]
            c = text.rfind(" ")
            if c > len(text) - 40:
                text = text[:c]
            text = text.strip()
            if len(text) < min_chars:
                misses += 1
                continue

            h = hashlib.blake2b(text.encode()[:512], digest_size=8).digest()
            if h in seen:
                misses += 1
                continue
            seen.add(h)
            out.append(text)
    return domain, out, misses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-chunks", type=int, default=5_000_000)
    ap.add_argument("--min-chars", type=int, default=400)
    ap.add_argument("--max-chars", type=int, default=1800, help="~450 teacher tokens")
    ap.add_argument("--val-chunks", type=int, default=4096, help="held out to measure geometry transfer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = args.n_chunks + args.val_chunks

    jobs = [(d, int(total * fr), args.min_chars, args.max_chars, args.seed + i)
            for i, (d, fr) in enumerate(MIXTURE.items())]

    t0 = time.time()
    rows = []
    with Pool(len(jobs)) as pool:  # one worker per domain file; the 10MB-record file is the long pole
        for domain, chunks, misses in pool.imap_unordered(sample_file, jobs):
            rows += [{"text": c, "domain": domain} for c in chunks]
            print(f"  {domain:<20} {len(chunks):>9,} chunks  ({misses:,} rejects)  "
                  f"[{(time.time()-t0)/60:.1f} min]", flush=True)

    rng = np.random.default_rng(args.seed)
    rng.shuffle(rows)
    val, train = rows[: args.val_chunks], rows[args.val_chunks :]

    for name, part in (("train", train), ("val", val)):
        p = out_dir / f"{name}.jsonl"
        with open(p, "w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
        chars = sum(len(r["text"]) for r in part)
        print(f"\n{name:<6} {len(part):>9,} chunks  {chars/1e9:.3f}B chars  -> {p}")

    el = time.time() - t0
    print(f"\ndone in {el/60:.1f} min  ({len(rows)/max(el,1e-9):,.0f} chunks/s)")


if __name__ == "__main__":
    main()
