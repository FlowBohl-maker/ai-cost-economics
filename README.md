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

## Historical price dataset

Monthly price history (1st of each month, 2023-10 → current month) built from third-party
sources, to complement the daily OpenRouter snapshots.

```bash
uv run python scripts/fetch_litellm_history.py   # blobless clone of BerriAI/litellm -> .cache/
# download Epoch into .cache/ (see "Sources"), then:
uv run python scripts/build_history.py
```

The committed data is built from LiteLLM, Epoch and OpenRouter only. llm-prices.com declares no
licence, so it is opt-in for local use: download it into `.cache/llm-prices/` and run
`build_history.py --with-llm-prices`. That also writes `data/price_conflicts.csv` and `.md`
(disagreements over 5% between the two price sources). Don't commit or publish a build that
includes it.

`.cache/` (clone, downloads) is git-ignored; only derived files in `data/` are committed.

### Outputs

| file | content |
|---|---|
| `data/price_history.parquet` | Long format, same columns as `prices.parquet` plus `source` (`litellm` / `llm_prices`), `canonical_id`, `eci`, `inferred_start`, `price_conflict`. Unique key: `(snapshot_date, source, model_id)`. `model_id` is the source's own ID; join on `canonical_id` across sources and to OpenRouter. |
| `data/model_map_overrides.csv` | Hand-reviewed mappings (`source, source_id, canonical_id, note`). Applied before automatic matching; add a row here to resolve a review item. |
| `data/model_map.csv` | Source ID → canonical ID for LiteLLM, llm-prices, OpenRouter and Epoch, with `match_method`, `needs_review`, `candidate_canonical_id`, `note`. **Review the rows with `needs_review = true`.** |
| `data/models.csv` | One row per canonical model: first/last month, sources, vendor, ECI (+ CI), Epoch model name. |

### Method notes

- **LiteLLM**: the last commit before 00:00 UTC on the 1st of each month; text models only
  (`mode` chat / completion / responses). Context = `max_input_tokens`, falling back to
  `max_tokens` for early files. One row per LiteLLM key, so hosting variants (Bedrock, Azure, …)
  appear as separate `model_id`s with their own `provider`.
- **llm-prices**: a record applies to the 1st of a month when `from_date <= month < to_date`.
  Most records have no `from_date`; for those the start is **inferred** (Epoch release month if
  matched, else first LiteLLM appearance, else the current month only) and `inferred_start = true`.
  Exact duplicate records in the source file are dropped.
  Context-length price tiers (e.g. `claude-sonnet-4-200k`) are not emitted; they're listed in
  `model_map.csv` as `tier_variant`. No context window in this source (null).
- **Canonical ID** = `vendor/slug`: provider/region prefixes, version suffixes, `:free`/`:batch`-style
  suffixes and date stamps are stripped and dots in versions become dashes.
- **Dated snapshots** (`gpt-4o-2024-05-13`, `mistral-large-2407`, `gpt-3.5-turbo-1106`) merge into one
  canonical model when they share a price. When their prices differ by more than 5 %, each keeps its
  stamp (`openai/gpt-4o-20240513`) and undated aliases (`gpt-4o`, `-latest`) stay on the undated ID.
  Prices are compared per stamp at that stamp's latest month, using the model maker's own LiteLLM rows
  when those cover at least two stamps, otherwise all providers.
- Near-matches (fuzzy) are **never merged automatically**; they get `match_method = fuzzy_candidate`
  and a `candidate_canonical_id` for review. Approved ones go into `model_map_overrides.csv`
  (`match_method = manual_override`).
- **Conflicts**: llm-prices vs first-party LiteLLM rows (e.g. provider `openai` for OpenAI models)
  per canonical model, month and price field. A conflict is when no first-party LiteLLM row agrees
  within 5 %. Both rows are kept; both get `price_conflict = true`. LiteLLM→llm-prices comparison
  only covers vendors with a defined first-party provider.
- **ECI**: Epoch's Capabilities Index attached only on an exact normalised-name match. Epoch models
  whose names collapse to the same ID (several dated snapshots) are *not* attached
  (`ambiguous_epoch_variants`); fuzzy matches are listed for review.

## Token price curve

```bash
uv run python scripts/build_curve.py              # v2 (default)
uv run python scripts/build_curve.py --version v1 # or v0; earlier versions still reproduce
```

Reads `price_history.parquet`, `prices.parquet`, `model_map.csv` and `models.csv`. Builds a monthly
blended price (3:1 input:output, USD per 1M tokens) per model, then an index per capability band: the
cheapest model with Epoch ECI ≥ 120, 130, 140, 150 and 160.

| file | content |
|---|---|
| `data/curves/curve_v2.csv`, `.json` | 36-month forward curve per band. Scenarios: `base` = no change (the published base case), `decline` = the 2024–25 decline resumes (÷1.7 per year), `fast_decline`. The JSON adds history and per-band state, for the website. |
| `data/curves/tier_history_v2.csv` | Monthly index per band with the cheapest model behind it. |
| `data/curves/panel_v2.parquet` | Model-month panel the indices are built from. |
| `reports/curve_v2.md` | Method, results, backtests, judgement calls, limitations. Also `curve_v0.md` and `curve_v1.md`. |
| `reports/curve_v2_*.png`, `_results.json`, `_rolling_backtest.csv` | Charts, all reported numbers, backtest errors. |

"No change" is the base case because it beat every fitted trend in the backtests. See
`reports/curve_v2.md`.

### Sources, licences and attribution

| source | licence | attribution / caveats |
|---|---|---|
| [BerriAI/litellm](https://github.com/BerriAI/litellm) `model_prices_and_context_window.json` | MIT (outside `litellm/enterprise/`; the JSON is in the MIT part), © 2023 Berri AI | Keep the MIT copyright notice: "Copyright (c) 2023 Berri AI". |
| [llm-prices.com](https://www.llm-prices.com/) `historical-v1.json` (repo [simonw/llm-prices](https://github.com/simonw/llm-prices)) | **No licence declared** (no LICENSE file; GitHub reports none) | Credit Simon Willison / llm-prices.com. Redistribution terms are unclear, so the committed data excludes it (opt-in with `--with-llm-prices` for local builds only). |
| [Epoch AI benchmark data](https://epoch.ai/data/benchmark_data.zip) | CC BY 4.0 | Cite: Epoch AI, "Capabilities & benchmarking". Published online at epoch.ai. Retrieved from https://epoch.ai/benchmarks. Indicate that values were joined to price data and ECI is unmodified. |
| [OpenRouter models API](https://openrouter.ai/api/v1/models) | Public API; subject to OpenRouter's terms of service | Used for daily snapshots and as an ID source in the map. |
