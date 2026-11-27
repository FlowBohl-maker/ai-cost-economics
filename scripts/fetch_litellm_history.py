"""Reconstruct LiteLLM's model_prices_and_context_window.json as of the 1st of each month.

Uses a blobless clone (full commit history, blobs fetched on demand) of
BerriAI/litellm, finds the last commit before 00:00 UTC on the 1st of every month
from 2023-10 to the current month, extracts price / context fields for text
models and writes .cache/litellm_monthly.parquet (long format, one row per
month x LiteLLM key).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache"
REPO_URL = "https://github.com/BerriAI/litellm.git"
PATHS = [  # the file moved/was mirrored over time; try in order
    "model_prices_and_context_window.json",
    "litellm/model_prices_and_context_window_backup.json",
]
TEXT_MODES = {"chat", "completion", "responses", None}


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "--git-dir", str(repo / ".git"), *args],
                          capture_output=True, text=True)


def months(start=(2023, 10)):
    today = datetime.now(timezone.utc)
    y, m = start
    while (y, m) <= (today.year, today.month):
        yield f"{y:04d}-{m:02d}-01"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def per_m(v):
    return float(v) * 1_000_000 if isinstance(v, (int, float)) and v >= 0 else None


def extract(doc: dict, snap: str, commit: str) -> list[dict]:
    rows = []
    for key, v in doc.items():
        if key == "sample_spec" or not isinstance(v, dict):
            continue
        if v.get("mode") not in TEXT_MODES:
            continue
        inp, out = per_m(v.get("input_cost_per_token")), per_m(v.get("output_cost_per_token"))
        if inp is None and out is None:
            continue
        ctx = v.get("max_input_tokens") or v.get("max_tokens")  # old files only have max_tokens
        rows.append({
            "snapshot_date": pd.Timestamp(snap),
            "model_id": key,
            "provider": v.get("litellm_provider"),
            "input_price_per_mtok": inp,
            "output_price_per_mtok": out,
            "cache_read_price_per_mtok": per_m(v.get("cache_read_input_token_cost")),
            "cache_write_price_per_mtok": per_m(v.get("cache_creation_input_token_cost")),
            "context_length": ctx if isinstance(ctx, int) else None,
            "litellm_commit": commit[:10],
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=CACHE / "litellm", help="local clone (created if missing)")
    args = ap.parse_args()
    if not (args.repo / ".git").exists():
        args.repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", "--single-branch",
                        REPO_URL, str(args.repo)], check=True)

    rows = []
    for snap in months():
        c = git(args.repo, "rev-list", "-1", "--first-parent", f"--before={snap}T00:00:00Z", "HEAD").stdout.strip()
        if not c:
            print(f"{snap}: no commit", file=sys.stderr); continue
        for p in PATHS:
            r = git(args.repo, "show", f"{c}:{p}")
            if r.returncode == 0:
                break
        else:
            print(f"{snap}: file missing at {c}", file=sys.stderr); continue
        doc = json.loads(r.stdout)
        got = extract(doc, snap, c)
        rows += got
        print(f"{snap} {c[:8]} {p.split('/')[-1][:25]:25} {len(got)} text models")
    df = pd.DataFrame(rows)
    CACHE.mkdir(exist_ok=True)
    df.to_parquet(CACHE / "litellm_monthly.parquet", index=False)
    print(f"wrote {len(df)} rows, {df.model_id.nunique()} keys")
    return 0


if __name__ == "__main__":
    sys.exit(main())
