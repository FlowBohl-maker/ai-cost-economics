"""Fetch OpenRouter model prices and append a daily snapshot.

Writes the raw API response to data/raw/YYYY-MM-DD.json and upserts rows into
data/prices.parquet. Re-running on the same (UTC) day replaces that day's
rows, so it never creates duplicates.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd
import requests

URL = "https://openrouter.ai/api/v1/models"
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PARQUET = ROOT / "data" / "prices.parquet"

COLUMNS = {
    "snapshot_date": "datetime64[ns]",
    "model_id": "string",
    "provider": "string",
    "input_price_per_mtok": "float64",
    "output_price_per_mtok": "float64",
    "cache_read_price_per_mtok": "float64",
    "cache_write_price_per_mtok": "float64",
    "context_length": "Int64",
}


def per_million(value) -> float | None:
    """Convert a per-token price string to USD per million tokens.

    Returns None when missing or not a real price (OpenRouter uses "-1" for
    variable-priced routers such as openrouter/auto).
    """
    if value is None or value == "":
        return None
    try:
        price = Decimal(str(value))
    except InvalidOperation:
        return None
    if price < 0:
        return None
    return float(price * 1_000_000)


def to_rows(payload: dict, snapshot_date: str) -> pd.DataFrame:
    rows = []
    for m in payload["data"]:
        pricing = m.get("pricing") or {}
        model_id = m["id"]
        rows.append(
            {
                "snapshot_date": pd.Timestamp(snapshot_date),
                "model_id": model_id,
                "provider": model_id.split("/", 1)[0],
                "input_price_per_mtok": per_million(pricing.get("prompt")),
                "output_price_per_mtok": per_million(pricing.get("completion")),
                "cache_read_price_per_mtok": per_million(pricing.get("input_cache_read")),
                "cache_write_price_per_mtok": per_million(pricing.get("input_cache_write")),
                "context_length": m.get("context_length"),
            }
        )
    return pd.DataFrame(rows).astype(COLUMNS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="override snapshot date (YYYY-MM-DD, default: today UTC)")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild the parquet file from data/raw/*.json only (no API call)")
    args = parser.parse_args()
    if args.rebuild:
        frames = [to_rows(json.loads(f.read_text()), f.stem) for f in sorted(RAW_DIR.glob("*.json"))]
        combined = pd.concat(frames, ignore_index=True).sort_values(["snapshot_date", "model_id"])
        combined = combined.drop_duplicates(["snapshot_date", "model_id"], keep="last").reset_index(drop=True)
        combined.to_parquet(PARQUET, index=False)
        print(f"rebuilt {PARQUET.relative_to(ROOT)} from {len(frames)} raw files: {len(combined)} rows")
        return 0
    snapshot_date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    resp = requests.get(URL, timeout=60)
    resp.raise_for_status()
    payload = resp.json()  # validates JSON and shape before anything is written
    if not payload.get("data"):
        print("API returned no models; refusing to write.", file=sys.stderr)
        return 1

    # Raw response, byte-for-byte as received.
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / f"{snapshot_date}.json").write_bytes(resp.content)

    new = to_rows(payload, snapshot_date)
    if PARQUET.exists():
        old = pd.read_parquet(PARQUET)
        old = old[old["snapshot_date"] != pd.Timestamp(snapshot_date)]  # idempotent re-run
        combined = pd.concat([old.reindex(columns=list(COLUMNS)).astype(COLUMNS), new], ignore_index=True)
    else:
        combined = new
    combined = combined.sort_values(["snapshot_date", "model_id"]).reset_index(drop=True)
    combined = combined.drop_duplicates(["snapshot_date", "model_id"], keep="last")

    tmp = PARQUET.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp, index=False)
    tmp.replace(PARQUET)

    print(f"{snapshot_date}: {len(new)} models; {len(combined)} total rows in {PARQUET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
