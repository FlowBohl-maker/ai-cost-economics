# Token price curve v3: floor tests, reference-provider prices, frontier and mid tiers

Built by `uv run python scripts/build_curve.py` (v3 is now the default). v0–v2 still reproduce with
`--version`. Data runs to 2026-11 and comes from LiteLLM, Epoch and OpenRouter only; llm-prices.com
is no longer used (see the README). All numbers are in
[`curve_v3_results.json`](curve_v3_results.json).

**Bottom line.**
- **Floor confirmed for the cheapest bands.** The cheapest models at ECI ≥ 120 and ≥ 130 (both
  gpt-oss-20b) have cost $0.0875 per million tokens since **Sep 2025**. The break test is clear
  (p = 0.001), and the price hasn't moved in 15 months. **Base case: flat.**
- **No floor for ECI ≥ 140 to ≥ 160.** These bands also broke in Sep 2025 (p = 0.001), but they
  have kept falling slowly since: ÷1.29 per year, 95% CI ÷1.14 to ÷1.46. **Base case: ÷1.29 per year
  from $0.20.** ECI ≥ 150 and ≥ 160 are too new to test. All three bands are priced by the same model
  today (Claude Haiku 5.5), so they share the ECI ≥ 140 base case.
- **Mid and frontier tiers.** Models near the frontier cost dollars, not cents: about $4 (mid) and
  $9 (frontier) per million tokens. Neither tier shows a decline. **Base case: flat for both.** The
  frontier tier is noisy, so treat its base case as low confidence.

## What changed from v2

| | v2 | v3 |
|---|---|---|
| Price of a hosted model | Median across all LiteLLM hosting providers | One **reference provider** per model |
| Base price level | Latest month | **3-month median** |
| Tiers | Cheapest model at ECI ≥ 120 … ≥ 160 | The same bands, plus **mid** and **frontier** |
| Base case | "No change" for every band | **Per tier**, from a structural-break test |
| Price sources | LiteLLM, llm-prices, OpenRouter | LiteLLM, OpenRouter |

## Check 2: reference provider and 3-month median

**Reference provider.** A hosted model (one with no price from its maker, typically open-weight)
used to get the median across every LiteLLM hosting provider. That median moved whenever a
provider was added or dropped. For example, gpt-oss-20b gained darkbloom at $0.028 and oci at
$0.128 during 2026. Now each model has one reference provider, chosen as follows:
- The reference is the provider with the longest price history for that model.
- Ties go to a fixed list: together_ai, fireworks_ai, deepinfra, groq, bedrock, azure_ai,
  vertex_ai, nebius, novita.
- Gateways and resellers are never chosen (openrouter, vercel_ai_gateway and similar).
- A provider-month priced more than 20× the cheapest provider that month is treated as a units
  error and skipped. Example: wandb at $8,750.
- In months where the reference provider has no price, the median of the other providers is used.
  That applies to 5% of scored model-months.

Model makers' own prices are unchanged. Overall, 47% of scored model-months use the maker's own
price, 49% a reference provider and 5% the median fallback.

**Effect:** under v2, the ECI ≥ 120 floor appeared to move between $0.0875 and $0.1225 from month to
month. With a reference provider it is exactly $0.0875 every month since Sep 2025.

**3-month median.** The base price is the median of the last 3 months:
- **For the threshold bands,** it is the median of the **current cheapest model's own price**, not
  of the band index. Otherwise a band that just stepped down would start from the price of the
  model it replaced. For example, ECI ≥ 160's index was $4.50, $4.00 and $0.20 over the last 3
  months, but its cheapest model, Claude Haiku 5.5, has cost $0.20 since it launched.
- **For mid and frontier,** it is the median of the tier index.

## Check 3: tier coverage

v3 adds two tiers defined by distance from the best model priced that month, measured in ECI
points:

| Tier | Models included | Index |
|---|---|---|
| frontier | within 5 ECI points of the best | median price |
| mid | 5–15 ECI points below the best | median price |

Both use only models released in the last 12 months, so old listings don't distort the median
(the problem in v0). These tiers measure **what typical models near the frontier cost**, not the
price of a fixed capability level. Their capability rises as the frontier moves.

### All tiers

Prices are blended USD per 1M tokens (3:1 input:output).

| Tier | Current cheapest model / models in tier | Latest | Base price (3-mo median) | Full-history trend (95% CI) | Since break | Last 12 months | Base case |
|---|---|---|---|---|---|---|---|
| frontier | 8 models | $6.00 | **$9.00** | ×1.67/yr rising (×1.11–×2.49) | no significant break | – | **flat** (judgement call) |
| mid | 35 models | $4.00 | **$4.00** | ÷1.04 (×1.39 to ÷1.49): no trend | no significant break | ×2.1 rising | **flat** |
| ECI ≥ 160 | Claude Haiku 5.5 | $0.20 | **$0.20** | 7 months only | too short to test | – | **÷1.29/yr** (same model as ≥ 140) |
| ECI ≥ 150 | Claude Haiku 5.5 | $0.20 | **$0.20** | ÷40 (drop phase) | not significant | ÷40 | **÷1.29/yr** (same model as ≥ 140) |
| ECI ≥ 140 | Claude Haiku 5.5 | $0.20 | **$0.20** | ÷4.8 (÷2.8–÷8.2) | break Sep 2025, then ÷1.29 (÷1.14–÷1.46) | ÷1.44 | **÷1.29/yr** |
| ECI ≥ 130 | gpt-oss-20b | $0.0875 | **$0.0875** | ÷3.4 (÷1.3–÷8.9) | break Jul 2025 (p = 0.09), flat since Sep 2025 | flat | **flat** (same model as ≥ 120) |
| ECI ≥ 120 | gpt-oss-20b | $0.0875 | **$0.0875** | ÷1.9 (÷1.7–÷2.2) | break Sep 2025, then exactly flat | flat | **flat: floor** |

The full-history trends for ECI ≥ 140 to ≥ 160 are dominated by each band's one-off drop (see v1).
For those bands, the rate since the break is the relevant one.

## Check 1: is there a floor?

### Rolling decline rates

[`curve_v3_rolling_rates.png`](curve_v3_rolling_rates.png) plots, for every tier, the annual price
factor fitted over trailing 6-month and 12-month windows. Every value is in
[`curve_v3_rolling_rates.csv`](curve_v3_rolling_rates.csv).
- **ECI ≥ 120 and ≥ 130:** both windows reached exactly 1.0 (flat) in early 2026 and have stayed
  there.
- **ECI ≥ 140:** the rate recovered from the drop and now sits at ÷1.4–÷1.6 per year. Prices are
  still falling, just slowly.
- **ECI ≥ 150 and ≥ 160:** still inside their drop phase, so their rolling rates are extreme.
- **Mid and frontier:** the rolling rates swing between ÷10 and ×10, because the set of models in
  each tier keeps changing.

### Break test

For each tier I tested for one break in level and slope (a sup-F test), searched over every
month with at least 6 months of data on each side:
- **p-value:** from a parametric bootstrap (999 draws) under the "no break" alternative, a linear
  trend with AR(1) errors. The usual tables would overstate significance in short, autocorrelated
  series like these.
- **After the break:** the slope after the most likely break month is reported with a
  Newey-West CI.

| Tier | Most likely break | p-value | Slope since break (95% CI) | Months since | Verdict |
|---|---|---|---|---|---|
| ECI ≥ 120 | 2025-09 | **0.001** | flat (exactly) | 15 | **Floor supported since Sep 2025** |
| ECI ≥ 130 | 2025-07 | 0.09 | ÷1.36 (×1.03 to ÷1.91) | 17 | Not significant on its own. Since Sep 2025 it is the same model and price as ≥ 120. The test picks Jul 2025 because of the expensive months at the band's start ($5.25, then $1.93). |
| ECI ≥ 140 | 2025-09 | **0.001** | ÷1.29 (÷1.14–÷1.46) | 15 | **Break, but no floor:** still falling slowly |
| ECI ≥ 150 | 2026-06 | 0.16 | – | 6 | Too early: the band reached its current price only in Oct 2026 |
| ECI ≥ 160 | – | – | – | – | Too short to test (7 months) |
| mid | 2026-01 | 0.43 | – | 11 | No break, no trend: flat throughout |
| frontier | 2026-04 | 0.15 | – | 8 | No significant break. Noisy, with a hump in Apr–May 2026 when expensive "Pro" models were within 5 points of the best. |

**Answer:**
- **ECI ≥ 120 and ≥ 130 have hit a floor.** It started in **Sep 2025**, when gpt-oss-20b became
  the cheapest model at both levels. It is the only clear floor in the data.
- **ECI ≥ 140 broke from its steep drop in Sep 2025** to a slow decline. Its prices are still
  falling, about ÷1.3 per year.
- **ECI ≥ 150 and ≥ 160** can't be tested yet. They reached $0.20 only in Sep–Nov 2026.
- **Mid and frontier** have no decline to speak of. Typical prices near the frontier have mostly stayed
  at $3–10 per million tokens since 2024, with swings outside that range.

## Base case rule

Applied in this order:
1. **Same model.** A band whose cheapest model also sets a lower band's price takes that band's
   base case. They share a price today, so they should share a projection.
2. **Floor.** A significant break (p < 0.05) with no significant change since: flat.
3. **Break, still falling.** A significant break with a falling slope since: that rate continues.
4. **No break.** Use the full-history fitted rate if it falls significantly. Hold flat if the trend
   isn't significant.

A base case never projects a rise. A significantly rising trend is held flat and flagged. This
applies only to the frontier tier: its fitted trend is ×1.67 per year, but that comes from the
Apr–May 2026 hump, and the price has fallen since.

**Scenarios in `curve_v3.csv` and `.json`:**
- `base`: the rule above.
- `fast`: the constant-capability (hedonic) decline, ÷1.86 per year, unless the base is already
  steeper.
- `slow`: no change from the base price.

### Forward curve (USD per 1M tokens, from 2026-11)

| Tier | Base now | +12 months: base (fast) | +36 months: base (fast) |
|---|---|---|---|
| frontier | 9.00 | 9.00 (4.84) | 9.00 (1.40) |
| mid | 4.00 | 4.00 (2.15) | 4.00 (0.62) |
| ECI ≥ 140, ≥ 150, ≥ 160 | 0.200 | 0.156 (0.108) | 0.094 (0.031) |
| ECI ≥ 120, ≥ 130 | 0.0875 | 0.0875 (0.047) | 0.0875 (0.014) |

`slow` equals the base price throughout.

## Backtests

I ran the same rule on data up to each cutoff and compared the result with what happened next.
MALE is the mean absolute log error.

| Tier | From 2025-11, 12 months: rule chose | MALE | No change | From 2026-05, 6 months: rule chose | MALE | No change |
|---|---|---|---|---|---|---|
| ECI ≥ 120 | full-history rate ÷2.5 | 65% | **0%** | **floor (flat)** | **0%** | 0% |
| ECI ≥ 130 | same as ≥ 120 | 65% | **0%** | **floor (flat)** | **0%** | 0% |
| ECI ≥ 140 | full-history rate ÷38 | 530% | **13%** | rising, held flat | 26% | 26% |
| ECI ≥ 150 | – | | | full-history rate ÷3.3 | **176%** | 255% |
| ECI ≥ 160 | – | | | too short, flat | 216% | 216% |
| mid | full-history rate ÷2.5 | 693% | 378% | no trend (flat) | 25% | 25% |
| frontier | no trend (flat) | 219% | 219% | rising, held flat | 218% | 218% |

- **The rule can only see a floor about 6 months after it starts.** In Nov 2025, the ECI ≥ 120
  floor was 2 months old, so the rule used the full-history decline and missed the same way v1 did.
  By May 2026 it detected the floor and forecast the next 6 months exactly.
- **Mid and frontier are hard to forecast either way.** Their medians moved 3–5× within a year
  (mid's 3-month median: $0.73 in Nov 2025, $4.00 now), and "no change" misses by as much as the rule does.

## Judgement calls

1. **Frontier = within 5 ECI points; mid = 5–15 points below.** On the hedonic slope (each ECI point adds
   about 5% to the price), 5 points is worth about 30%. Wider windows would pull
   in the cheap floor models (Claude Haiku 5.5 is 9 points below the best).
2. **Median, not cheapest, for mid and frontier.** "Cheapest" would be one floor model again. The
   median answers "what do typical models at this level cost".
3. **Frontier held flat despite its rising fit.** See the base case rule.
4. **Shared base case for bands priced by the same model.** Without it, ECI ≥ 150 and ≥ 160 would
   inherit their drop-phase rates (÷40 and ÷600 per year).
5. **Reference provider by longest history.** Steady, but not the cheapest offer available. A
   buyer who shops around pays less: the cheapest gpt-oss-20b host is $0.028 (darkbloom), against
   $0.0875 at together_ai.
6. **Bootstrap null of a linear trend with AR(1) errors.** It is simple and transparent, but these
   series move in steps, which this null fits poorly. The p-values are a guide, not exact.

## Limitations

- **Few observations.** 7–32 months per tier, so test power is low. Only very sharp changes
  (ECI ≥ 120, ≥ 140) come out significant.
- **The frontier tier has 3–8 models.** One expensive "Pro" model can double the median.
- **The 3-month median lags real steps for mid and frontier.** Frontier's base ($9.00) is above its
  latest month ($6.00), because the tier has been falling since May 2026.
- **Earlier limitations still apply:** ECI coverage, no usage weighting, and price per token rather
  than per task.

## Files

- [`data/curves/curve_v3.csv`](../data/curves/curve_v3.csv) and
  [`curve_v3.json`](../data/curves/curve_v3.json): the forward curve. The JSON includes each tier's
  `base_case` with its rule and reason.
- [`data/curves/tier_history_v3.csv`](../data/curves/tier_history_v3.csv): monthly index per tier.
- [`data/curves/panel_v3.parquet`](../data/curves/panel_v3.parquet): the model-month panel, with
  price source.
- [`curve_v3_rolling_rates.csv`](curve_v3_rolling_rates.csv) and the results JSON: rolling rates,
  floor tests and backtests.
- Charts:
  - [rolling rates](curve_v3_rolling_rates.png)
  - [history](curve_v3_tier_history.png)
  - [forward curve](curve_v3_forward.png)
  - [backtest](curve_v3_backtest.png)
