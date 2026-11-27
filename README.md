# ai-cost-economics

How the cost of AI work changes over time. The project has two layers:

| Layer | What it does | Where |
|---|---|---|
| **1. Core: token price curve** | Token prices only: historical prices per model, then a monthly price per tier and token type (input, output, cache read, cache write), projected 36 months forward under three scenarios. Knows nothing about workflows. | `scripts/`, `data/`, `reports/curve_*` |
| **2. Workflow layer: token profiles** | A workflow is a YAML profile of assumed token volumes per task. A cost function combines a profile with the curve to give cost per task and per month. | `workflows/` |

Live model runs are not needed. They are an optional calibration step for profiles
(see [Optional: calibrating a profile](#optional-calibrating-a-profile)).

Layer 1 starts from daily snapshots of model pricing from the
[OpenRouter models API](https://openrouter.ai/api/v1/models). The models API is public and needs no
account or key.

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
| `cache_write_price_per_mtok` | float64 | USD per million tokens written to the prompt cache (`input_cache_write`); null if not listed |
| `context_length` | Int64 | Context window in tokens |

OpenRouter reports prices per token as strings; they are multiplied by 1,000,000.
`uv run python scripts/fetch_prices.py --rebuild` rebuilds the parquet file from `data/raw/` without an API call.
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
| `data/price_history.parquet` | Long format, same columns as `prices.parquet` (cache-write prices from LiteLLM's `cache_creation_input_token_cost`) plus `source` (`litellm` / `llm_prices`), `canonical_id`, `eci`, `inferred_start`, `price_conflict`. Unique key: `(snapshot_date, source, model_id)`. `model_id` is the source's own ID; join on `canonical_id` across sources and to OpenRouter. |
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
uv run python scripts/build_curve.py              # v3 (default): tiers, base cases, blended curve
uv run python scripts/build_curve.py --version v2 # or v0, v1; earlier versions still reproduce
uv run python scripts/build_curve_v4.py           # v4: v3 split by token type, plus the capability lookup
```

Reads `price_history.parquet`, `prices.parquet`, `model_map.csv` and `models.csv`. Builds a monthly
blended price per model (3:1 input:output, USD per 1M tokens), then an index per tier:

| Tier | What it measures |
|---|---|
| `frontier` | Median price of models within 5 ECI points of the best model priced that month (released in the last 12 months) |
| `mid` | The same, for models 5–15 ECI points below the best |
| `eci>=120` … `eci>=160` | Cheapest model with Epoch ECI at or above the threshold |

A hosted model's price comes from one **reference provider**, chosen as the provider with the longest
price history for that model, not from the median across providers. The maker's own price is used
where one exists.

**Base case per tier.** A structural-break test checks whether prices have hit a floor:
- **Floor supported:** the base is flat.
- **Otherwise:** the base follows the fitted decline.
- **Rising trend:** the base is held flat; it never projects a rise.
- **Bands priced by the same model:** they share a base case.

The base level is a 3-month median.

Base case as of 2026-11:

| Tier | Base level | Base case |
|---|---|---|
| frontier | about $9 | flat |
| mid | about $4 | flat |
| ECI ≥ 140 to ≥ 160 | $0.20 | ÷1.3 per year |
| ECI ≥ 120 and ≥ 130 | $0.0875 | flat (floor since Sep 2025) |

| file | content |
|---|---|
| `data/curves/curve_v3.csv`, `.json` | 36-month forward curve per tier. Scenarios: `base` (rule above), `fast` = constant-capability decline (÷1.86 per year), `slow` = no change. The JSON adds history and each tier's base case with its reason, for the website. |
| `data/curves/tier_history_v3.csv` | Monthly index per tier, with the cheapest model behind each band. |
| `data/curves/panel_v3.parquet` | Model-month panel the indices are built from, with price source. |
| `reports/curve_v3.md` | Method, floor tests, results, backtests, judgement calls, limitations. Also `curve_v0.md` to `curve_v2.md`. |
| `reports/curve_v3_*.png`, `_results.json`, `_rolling_rates.csv` | Charts, all reported numbers, rolling decline rates. |

### Curve v4: prices per token type and by capability

v4 keeps v3's tiers, base levels and rates, and v3's files stay unchanged. It adds three things:
- **Separate input, output, cache-read and cache-write prices** per tier, scenario and month
  (months 0–36). Reasoning tokens are billed at the output price. Cache prices come from the same source
  as the model's input price (its maker, or v3's reference provider). Where that source lists none, those
  tokens are priced as input.
- **A capability lookup:** the cheapest model at or above each whole ECI threshold (120–169), each
  month since 2023-10, and projected forward.
- **A `small` median tier.** It is built like v3's mid and frontier tiers, from models 15–30 ECI points
  below the best model, released in the last 12 months, using v3's base-case rule. The rule finds no
  significant trend, so the tier is held flat at $1.06 blended.

**How tier prices are built.**
- **`eci>=N` tiers:** every token type comes from one model, the cheapest at or above the threshold.
- **`small`, `mid` and `frontier`:** every token type comes from the **median model by blended price**.
  - With an even number of models, the middle two are averaged.
  - Models tied at the median price are averaged too.
  - The base level is the month, among the last three, whose median blended price is v3's base level.
  - So the blended price equals v3's exactly, and input, output, cache-read and cache-write prices come
    from the same model or models.
  - Until 2026-11, each token type was a separate median across models. That mixed price structures from
    different models.

Details and before/after numbers are in [`reports/curve_v4.md`](reports/curve_v4.md).

| Tier (2026-11) | Input | Output | Cache read | Cache write | Priced from |
|---|---|---|---|---|---|
| ECI ≥ 120, ≥ 130 | $0.05 | $0.20 | $0.05 | $0.05 | gpt-oss-20b (its reference provider lists no cache prices) |
| ECI ≥ 140 to ≥ 160 | $0.10 | $0.50 | $0.01 | $0.125 | Claude Haiku 5.5 |
| small (median model) | $0.50 | $2.75 | $0.32 | $0.50 | Kimi K2 Thinking and Qwen 3.8 27B, averaged |
| mid (median model) | $2.00 | $10.00 | $0.20 | $2.33 | Claude Sonnet 5, Kimi K3.1 Pro and GPT-6 Sol (tied) |
| frontier (median model) | $4.50 | $22.50 | $0.35 | $5.625 | Claude Opus 5 and Opus 5.5, averaged (Oct 2026) |

| file | content |
|---|---|
| `data/curves/curve_v4.csv`, `.json` | Per tier, scenario and month: `input`, `output`, `cache_read`, `cache_write`, `blended` (USD per 1M tokens). `months_ahead = 0` is today's base level. |
| `data/curves/tier_history_v4.csv` | v3's monthly tier history plus `small`, with a price per token type and the `median_models` behind each median-tier month. |
| `data/curves/capability_v4.csv` | Capability lookup: `month`, `eci_min`, `scenario` (`history` for past months), cheapest `model`, prices per type, `rate_from`. |
| `reports/curve_v4.md`, `curve_v4_capability.png`, `curve_v4_results.json` | Method, checks, judgement calls, chart. |

## Workflow layer: token profiles

A **profile** describes one workflow as token volumes per task. Price data stays in the curve;
the profile holds only the volumes and which price applies.

```bash
uv run python -m workflows.run_profiles                     # every profile in workflows/profiles/
uv run python -m workflows.run_profiles my_profiles/ --out out/ --from 2025-01 --to 2027-12
```

This writes `profile_costs.csv` and `profile_costs.png`.
- **CSV:** one row per profile, month, scenario and **price view**.
  - `price_view` (with `price_case` and `price_view_label`) says which view a row's numbers come from:
    `median` (central price case) or `cheapest` (low price case).
  - `cost_per_task_{low,central,high}_volume` and `monthly_cost_*_volume` give the profile's token
    volumes at that view's prices.
  - `band_up_*` gives the same view one band up.
  - `notes` and `warnings` explain a missing view and flag a threshold near a band edge.
- **Chart:** one panel per price view, titled with the view.
The months default to today through +36 months; earlier months are priced from the curve's history.
Output goes to `reports/workflows/` by default. That folder is git-ignored until real profiles exist.

### Capability bands and the two price views

Profiles should choose a **capability band**. The band edges and each band's median tier are defined
once, in [`config/capability_bands.yaml`](config/capability_bands.yaml). Every band is priced two
ways, and the cost output shows both:

| View | Price case | What it prices |
|---|---|---|
| `cheapest` | **low** | The cheapest model whose ECI is at or above the band's minimum, month by month (capability lookup). This is the cheapest way to get the capability. |
| `median` | **central** | The median model of the band's median tier: models released in the last 12 months within a fixed number of ECI points of the best model. This is what a typical current model in the band costs. |

| Band | Min. ECI | Low case today: cheapest model | Blended | Median tier | Central case today: blended |
|---|---|---|---|---|---|
| floor | 120 | gpt-oss-20b | $0.0875 | none | n/a |
| small | 140 | Claude Haiku 5.5 | $0.20 | `small` (15–30 points below the best) | $1.06 |
| mid | 154 | Claude Haiku 5.5 | $0.20 | `mid` (5–15 below) | $4.00 |
| frontier | 164 | GPT-6.1 Terra | $2.00 | `frontier` (0–5 below) | $9.00 |

- **Token volumes vs price cases.** Each view has its own low/central/high token volumes. The volume
  levels and the price cases are separate: the central volumes at the low price case are "a typical
  task on the cheapest qualifying model".
- **Floor has no median view.** Only one model released in the last 12 months scores between 120 and
  140, so a floor profile gets the low case only, with a note.
- **Small and mid share a low case today.** Claude Haiku 5.5 (ECI 160.3) costs $0.20 blended and clears
  both edges.
- **Median tiers are relative.** They are defined relative to the best model each month; the band
  edges are absolute. In 2026-11 the two match (the best model is ECI 169.2: 30, 15 and 5 points below
  are 139, 154 and 164). Over the forward curve the median tiers are flat in the base case, so they
  don't drift. For past months they can differ.
- **Fixed edges.** The edges are absolute ECI values, set against the 2026-11 frontier (best model:
  ECI 169.2). A band therefore tracks a fixed capability level as it gets cheaper. Review the edges
  when the frontier moves.

### Profile schema (YAML)

```yaml
name: my-workflow                 # required, unique
description: What one task is.    # required
example: false                    # true = made-up volumes; flagged in every output
tokens_per_task:                  # per task, summed over all model calls
  input:      {low: 2000, central: 6000, high: 15000}   # uncached input tokens (required)
  output:     {low: 300, central: 800, high: 2000}      # visible output tokens (required)
  reasoning:  {low: 0, central: 1500, high: 6000}       # billed at the output price (default 0)
  cache_read: {low: 0, central: 4000, high: 20000}      # input served from a prompt cache (default 0)
  cache_write: {low: 0, central: 1000, high: 4000}      # input written to a prompt cache (default 0)
model:                            # exactly one of:
  band: mid                       #   a capability band: floor | small | mid | frontier (recommended)
  # min_eci: 150                  #   or an exact threshold: the cheapest model with Epoch ECI >= 150
  # tier: mid                     #   or a curve tier: eci>=120 ... eci>=160, mid, frontier
effort: medium                    # none | low | medium | high (a label: the volumes should match it)
runs_per_month: {low: 500, central: 2000, high: 5000}  # optional; adds monthly cost
```

- **Writing a value.** Any volume can be one number (low = central = high) or `{low, central, high}`.
  Unknown keys are rejected, which catches typos.
- **Cache writes.** Tokens written to a prompt cache go in `cache_write`, not `input`. Some providers
  charge more than input for them (Anthropic and OpenAI list 1.25× for several models); where nothing
  is listed, they cost the input price.
- **Exact thresholds.** `min_eci` stays available. A value within 2 ECI points of a band edge (the
  `edge_warning_points` setting) gets a warning when the profile loads and in the cost output, because
  small changes to the threshold can change the price a lot (for example, ECI ≥ 160 costs $0.20 blended
  and ≥ 161 costs $2.00).
- **One effort per profile.** For a workflow run at two effort levels, write two profiles.
- **Example profile.**
  [`workflows/profiles/example_placeholder.yaml`](workflows/profiles/example_placeholder.yaml) is a
  placeholder with **made-up volumes**. It shows the format and is not an estimate of anything.

### Cost function

```python
from workflows.cost import cost
from workflows.profile import load_profile

c = cost(load_profile("workflows/profiles/example_placeholder.yaml"), "2027-06", "base")
v = c["views"]["median"]     # central price case; c["views"]["cheapest"] is the low price case
v["label"], v["basis"]       # "tier median model", "band mid: median of tier mid"
v["cost_per_task"]           # {"low": ..., "central": ..., "high": ...} token volumes, in USD
v["monthly_cost"]            # the same times runs_per_month, or None if the profile has none
c["band_up"]["views"]        # both views one band up; band_up is None at the top band
c["notes"], c["warnings"]    # e.g. "band floor has no median tier", band-edge warnings for an exact min_eci
```

- **Arguments.** `cost(profile, month, scenario)` takes a month (`"2027-06"`, a date or a Timestamp)
  and a scenario: `base`, `fast` or `slow`.
- **Months.** From today to +36 months, prices come from curve_v4. Earlier months come from history,
  and the scenario has no effect there.
- **Formula.** Cost per task = Σ over token types of tokens × price ÷ 1M, with reasoning tokens at the
  output price.
- **Views.** A band profile gets both views. A `min_eci` profile gets the cheapest model at its exact
  threshold, plus the median tier of the band the threshold falls in. A `tier` profile gets its tier's
  own view only: `eci>=N` tiers are cheapest-model indices (low), while `small`, `mid` and `frontier`
  are medians (central). A view that doesn't apply is `None`.
- **Ranges.** Low, central and high spread the token volumes. The two views give the price range for
  today. The scenario covers price changes over time.
  Monthly cost pairs low runs with low volumes, and likewise for high, so its low and high are bounds,
  not percentiles.
- **`min_eci`.** Rounded up to a whole ECI point. Below 120, the ECI ≥ 120 price is used, which is an
  upper bound.
- **One band up.** For `band` and `min_eci` profiles, `band_up` reprices the same volumes, under both
  views, at the next band, for the same month and scenario. This is a sensitivity: what the task costs if it
  turns out to need more capability. `min_eci` profiles count from the band their threshold falls in.
  `tier` profiles have no band-up, because tiers and bands are priced differently.

Tests: `uv run pytest`.

### Optional: calibrating a profile

`workflows/calibration/` runs four synthetic finance workflows against a model and logs tokens per
task:
- data Q&A over a 2k-token table
- screening of 200 companies
- summary of a public earnings call
- a portfolio-review agent of at least 4 turns, with fixed tool responses

All inputs are synthetic except the earnings call, which is a public SEC filing. It is read from
`.cache/` and not redistributed. Each run gets an automated quality score.

```bash
# estimate only (no calls)
uv run python -m workflows.calibration.run_task_tokens --adapter mock --models demo
# local model, no account
uv run python -m workflows.calibration.run_task_tokens --adapter ollama --models gpt-oss:20b --run --reps 1
# draft profiles from logged runs
uv run python -m workflows.calibration.run_task_tokens summarise --runs data/tasks/token_runs.parquet --out drafts/
```

- **Adapters.** Model access goes through one small adapter per provider
  ([`adapters.py`](workflows/calibration/adapters.py)). Two are included, and neither needs an
  account:
  - `mock`: offline and deterministic, for tests.
  - `ollama`: a local Ollama server.

  To add a provider, such as Gemini's free tier or a single vendor's API, add a class with `price()`
  and `chat()` and register it. That class reads its own key from the environment or from `.env`,
  which is git-ignored; never commit a key.
- **Spending cap.** `--run` refuses to start if the estimate, with a 1.5× margin, is over `--max-cost`
  (default $20). It stops before any run that could take spending over the cap.
- **Output.** Results go to `data/tasks/token_runs.parquet`, with a resume log in `.cache/tasks/`.
  `summarise` turns runs into draft profiles: low, central and high are the min, median and max over
  repetitions. Review the drafts and set `model.band` (or `min_eci` / `tier`) before moving them into
  `workflows/profiles/`.

## Sources, licences and attribution

| source | licence | attribution / caveats |
|---|---|---|
| [BerriAI/litellm](https://github.com/BerriAI/litellm) `model_prices_and_context_window.json` | MIT (outside `litellm/enterprise/`; the JSON is in the MIT part), © 2023 Berri AI | Keep the MIT copyright notice: "Copyright (c) 2023 Berri AI". |
| [llm-prices.com](https://www.llm-prices.com/) `historical-v1.json` (repo [simonw/llm-prices](https://github.com/simonw/llm-prices)) | **No licence declared** (no LICENSE file; GitHub reports none) | Credit Simon Willison / llm-prices.com. Redistribution terms are unclear, so the committed data excludes it (opt-in with `--with-llm-prices` for local builds only). |
| [Epoch AI benchmark data](https://epoch.ai/data/benchmark_data.zip) | CC BY 4.0 | Cite: Epoch AI, "Capabilities & benchmarking". Published online at epoch.ai. Retrieved from https://epoch.ai/benchmarks. Indicate that values were joined to price data and ECI is unmodified. |
| [OpenRouter models API](https://openrouter.ai/api/v1/models) | Public API; subject to OpenRouter's terms of service | Used for daily snapshots and as an ID source in the map. |
