"""Generate a phone/GitHub-viewable status page for the synthetic-data generation.

Mirrors status_page.py (the training dashboard) but for the generation run: progress toward the pair
target, live rate + ETA, taxonomy coverage, health of the vLLM server, and a couple of sample pairs
so the quality is visible at a glance. Published to the `status` branch by run_gen_status_daemon.sh.

Reads ONLY on-disk artifacts and the log -- never touches the generator or the server.
"""
import json
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA = PROJECT_DIR / "data" / "pairs_synth_taxonomy" / "train.jsonl"
PROG = PROJECT_DIR / "data" / "pairs_synth_taxonomy" / "progress.json"
GEN_LOG = PROJECT_DIR / "gensyn.log"
TARGET = 1_000_000
HF = "https://huggingface.co/datasets/sakshamio/financial-legal-synthetic-retrieval"


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        return ""


def main():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n = int(sh(f"wc -l < {DATA}") or 0)
    pct = 100 * n / TARGET

    server_up = bool(sh("curl -s http://127.0.0.1:8000/v1/models 2>/dev/null | grep -o '\"id\"' | head -1"))
    gen_alive = bool(sh("pgrep -f '[g]en_synthetic_taxonomy.py'"))
    health = "🟢 generating" if server_up and gen_alive else (
        "🟠 server up, generator idle" if server_up else "🔴 server down")

    # rate: read the progress file the generator writes, else fall back to log
    rate = None
    fail = None
    if PROG.exists():
        try:
            p = json.loads(PROG.read_text())
            fail = p.get("fail_rate")
        except Exception:
            pass
    # rate from last two log ticks
    ticks = sh(f"grep -oE '[0-9.]+ pairs/s' {GEN_LOG} 2>/dev/null | tail -1")
    if ticks:
        try:
            rate = float(ticks.split()[0])
        except Exception:
            pass
    eta = (TARGET - n) / rate / 3600 if rate else None

    filled = int(pct / 100 * 28)
    bar = "█" * filled + "░" * (28 - filled)

    # taxonomy coverage
    cats = Counter()
    samples = []
    if DATA.exists():
        with open(DATA) as f:
            for i, line in enumerate(f):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                cats[(r.get("meta") or {}).get("category", "?")] += 1
                if i < 4000 and len(samples) < 3 and i % 800 == 0:
                    samples.append(r)

    gpu = sh("nvidia-smi --query-gpu=utilization.gpu,power.draw,temperature.gpu --format=csv,noheader")
    disk = sh(f"df -h {PROJECT_DIR} | tail -1 | awk '{{print $4}}'")

    md = [f"# Synthetic data generation — {health}\n",
          f"`{bar}` **{pct:.1f}%**\n",
          "| | |", "| --- | --- |",
          f"| **Pairs** | **{n:,}** / {TARGET:,} |",
          f"| **Rate** | {f'{rate:.1f} pairs/s' if rate else '—'} |",
          f"| **ETA** | {f'{eta:.1f} h ({eta/24:.1f} d)' if eta else '—'} |",
          f"| Generator | Qwen3.6-27B dense FP8 (via sparkrun/vLLM) |",
          f"| Parse-fail rate | {f'{fail:.1%}' if fail is not None else '—'} |",
          f"| Dataset (live) | [HF ↗]({HF}) |",
          f"| GPU | {gpu} |",
          f"| Disk free | {disk} |\n",
          "## Taxonomy coverage\n",
          "| category | pairs |", "| --- | --- |"]
    for c, v in cats.most_common():
        md.append(f"| {c} | {v:,} |")

    if samples:
        md.append("\n## Sample pairs\n")
        for r in samples:
            meta = r.get("meta") or {}
            md.append(f"**{meta.get('subtype','?')}** · *{meta.get('clause','?')}*")
            md.append(f"- Q: {r['anchor'][:140]}")
            md.append(f"- P: {r['positive'][:180]}…")
            if r.get("negative_0"):
                md.append(f"- ✗ hard-neg: {r['negative_0'][:140]}…")
            md.append("")

    md.append(f"<sub>Updated {now} · generation is synthetic (Qwen3.6-27B dense); passages are model-written, "
              f"not authentic documents.</sub>")

    (PROJECT_DIR / "GEN_STATUS.md").write_text("\n".join(md))
    print(f"{n:,} pairs ({pct:.1f}%) — GEN_STATUS.md written")


if __name__ == "__main__":
    main()
