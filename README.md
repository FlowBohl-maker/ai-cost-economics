# ai-cost-economics

Daily snapshots of model pricing from the [OpenRouter models API](https://openrouter.ai/api/v1/models),
for tracking how AI costs change over time.

A GitHub Actions workflow ([daily-fetch.yml](.github/workflows/daily-fetch.yml)) runs the
script every day at 06:00 UTC and commits any new data. It can also be started manually
from the Actions tab (`workflow_dispatch`).

## Run locally

Requires [uv](https://docs.astral.sh/uv/) (Python 3.12 is installed/selected automatically).

```bash
uv run python scripts/fetch_prices.py
```

Optional: `--date YYYY-MM-DD` overrides the snapshot date (default: today, UTC).

The script is safe to run repeatedly on the same day: the raw file is overwritten and
that day's rows in the parquet file are replaced, never duplicated.

## Data format

### `data/raw/YYYY-MM-DD.json`

The unmodified API response for that day (`{"data": [...], ...}`).

### `data/prices.parquet`

One row per model per snapshot date. Unique key: `(snapshot_date, model_id)`.

| column | type | description |
|---|---|---|
| `snapshot_date` | datetime64 | UTC date of the snapshot |
| `model_id` | string | OpenRouter model id, e.g. `anthropic/claude-sonnet-4` |
| `provider` | string | Part of `model_id` before the first `/` |
| `input_price_per_mtok` | float64 | USD per million input (prompt) tokens |
| `output_price_per_mtok` | float64 | USD per million output (completion) tokens |
| `cache_read_price_per_mtok` | float64 | USD per million cached-input read tokens; null if the model has no cache pricing |
| `context_length` | Int64 | Context window in tokens |

OpenRouter reports prices per token as strings; they are multiplied by 1,000,000.
Missing prices and OpenRouter's `-1` marker (variable pricing, e.g. `openrouter/auto`)
become null.

```python
import pandas as pd
df = pd.read_parquet("data/prices.parquet")
```
