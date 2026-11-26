# Token price curve v0

> **Data note (2026-11-26).** The numbers below were computed with llm-prices.com in the price data. That source declares no licence, so the committed data and curve files are now built without it (see the README). Re-running gives somewhat different numbers: the hedonic rate is unchanged at ÷1.9 per year; the tier backtest errors move (frontier 177% vs 90% for "no change"). The conclusions are unchanged.

Built by `uv run python scripts/build_curve.py` from `data/price_history.parquet` (LiteLLM + llm-prices),
the OpenRouter daily snapshots and Epoch ECI scores. All numbers below are in
[`curve_v0_results.json`](curve_v0_results.json). Data runs to 2026-11.

**Bottom line.** At constant capability, prices fall about **1.9× per year** (95% CI 1.4–2.5×).
The tier indices do **not** give a usable decline rate yet. The frontier tier is roughly flat. The
mid and small tier medians *rise*, because those tiers fill up with old, still-listed models
(details under Results). The forward curve follows the requested spec, but treat its mid and small
lines as an artefact of how the tiers are built, not as a forecast.

## Method

**Price measure.** Blended price = 0.75 × input + 0.25 × output (USD per 1M tokens), monthly (1st of
the month). Input and output are kept as separate columns in `data/curves/panel_v0.parquet`. Rows
where input or output is zero or missing are dropped. Each canonical model gets one price per
month, taken from the first available source in this order:
1. The model maker's own LiteLLM rows (for example provider `openai` for OpenAI models); median if
   there are several keys.
2. llm-prices.com.
3. Median across all LiteLLM hosting providers.
4. OpenRouter, as the monthly mean of the daily snapshots.

Shares of model-months with an ECI score: maker's own LiteLLM rows 44%, other LiteLLM providers
45%, llm-prices 11%, OpenRouter 0.1%. There is only one OpenRouter snapshot so far.

**a. Hedonic index.** Pooled OLS: `log(blended) = a + b·t + c·ECI`, with t in years. Standard errors
are clustered by model, because each model appears in many months. The annual price factor is
exp(b). Sample: 2,048 model-months from 169 models with an ECI score. The 3 model-months with
|studentized residual| > 3 in a first pass are excluded from every fit.

**b. Tier indices.** Each month, a model's ECI is divided by the best ECI among models priced that
month:

| Tier | ECI relative to best model that month |
|---|---|
| frontier | ≥ 0.90 |
| mid | 0.80–0.90 |
| small | < 0.80 |

The tier index is the monthly median blended price. Tier-months with fewer than 3 models are
dropped. The decline is fitted as `log(median) = a + b·t`, using OLS with Newey-West standard
errors (3 lags).

**Forward curve.** 36 months from 2026-11, per tier. The starting level is the geometric mean of
the last 3 monthly tier medians. Each scenario applies a constant log rate:

| Scenario | Rate |
|---|---|
| base | the fitted rate |
| fast | the steeper of the lower 95% bound and the last-12-month fitted rate |
| slow | the upper 95% bound |

Files: [`data/curves/curve_v0.csv`](../data/curves/curve_v0.csv) (long format) and
[`curve_v0.json`](../data/curves/curve_v0.json) (meta, fit, history and forecast per tier, for the
website).

**Backtest.** The same tier procedure, fitted only on months up to 2025-11 and projected to 2026-11.
Error is reported as the mean absolute log error (MALE), expressed in %.

## Results

### Annual price factor (below 1 = prices falling)

| View | Factor per year | 95% CI | R² | n |
|---|---|---|---|---|
| **Hedonic, constant ECI** | **0.54 (÷1.87)** | 0.40–0.71 (÷2.5 to ÷1.4) | 0.31 | 2,048 model-months |
| Frontier tier median | 0.87 | 0.69–1.09 | 0.08 | 34 months |
| Mid tier median | 1.51 | 0.75–3.02 | 0.12 | 31 months |
| Small tier median | 1.48 | 1.23–1.79 | 0.37 | 27 months |

- **ECI coefficient:** each extra ECI point raises the price by about 5.6% (the coefficient is
  0.054, CI 0.04–0.06). Roughly, 13 ECI points double the price.
- **Fit quality:** the tier fits have very low R². Month-to-month noise in the medians comes mostly
  from changes in which models are in each tier, not from price changes.

### Sensitivity of the hedonic rate (not used for the curve)

| Variant | Prices fall per year | 95% CI | n |
|---|---|---|---|
| Pooled (headline) | ÷1.9 | ÷1.4–÷2.5 | 2,048 |
| Launch price only (each model's first priced month) | ÷2.1 | ÷1.5–÷3.1 | 169 |
| Models ≤ 18 months since release only | ÷4.2 | ÷2.9–÷6.1 | 1,557 |

Old models keep their launch price while staying listed, so they pull the pooled rate toward zero.
The true constant-capability decline is most likely between 2× and 4× per year.

### Why the tier medians rise

In 2026-11 the median model age is 2 months in the frontier tier, 9 months in mid and **25 months in
small**. The small tier's most expensive members are GPT-4.5 ($94 blended), Claude 3 Opus ($30) and
o1-preview ($26). These were frontier models at launch. As the frontier moves up, they slide into
the lower tiers at their old prices. Restricting to models released in the last 18 months does not
fix this: the small tier still rises ×1.34 per year. That's because Epoch mostly scores notable
models, so genuinely new cheap small models rarely have an ECI.

### Forward curve (blended USD per 1M tokens)

| Tier | Now (anchor) | +12 months: base (fast–slow) | +36 months: base (fast–slow) |
|---|---|---|---|
| frontier | 4.28 | 3.71 (2.96–4.64) | 2.79 (1.42–5.47) |
| mid | 1.28 | 1.93 (0.96–3.87) | 4.38 (0.54–35.3) |
| small | 0.29 | 0.43 (0.24–0.52) | 0.94 (0.16–1.65) |

The mid and small base lines rise because of the tier artefact above. They should not be published.

### Backtest (fit to 2025-11, compared with 2025-12 to 2026-11)

| Tier | Rate fitted on training data | MALE | "No change" MALE | Bias | Error at 12 months | Months inside band |
|---|---|---|---|---|---|---|
| frontier | 0.59 per year | 147% | 86% | −60% (too low) | −70% ($1.45 vs $4.89) | 0 / 12 |
| mid | 3.37 per year | 51% | 52% | +51% | +28% | 10 / 12 |
| small | 2.03 per year | 68% | 25% | +45% | +34% | 1 / 12 |

"No change" is a naive benchmark: the price stays at its level at the cutoff (added in v1). It does
better than the fitted trend in every tier.

The tier method does not forecast well. The frontier tier fell in 2024–25 and then rose again in
2026, as new premium models (Claude Opus 5.x, GPT-6 Astra Pro and others) entered at higher price
points. Any log-linear fit misses a turn like that.

### Outliers and repricing events

- **Outliers:** one model is flagged, `qwen/qwen-3-235b-a22b-instruct` at $5,000/M for 3 months in
  2025-10 to 2025-12. That's a units error in LiteLLM, and it is excluded from the fits. See
  [`curve_v0_outliers.csv`](curve_v0_outliers.csv).
- **Repricing events:** a month-on-month change of 30% or more in a model's price series, counted
  only when both months come from a single source (the maker's own LiteLLM rows or llm-prices).
  There are 67 such events in [`curve_v0_repricing_events.csv`](curve_v0_repricing_events.csv).
  Notable ones among models with an ECI score:

  | Month | Model | Change |
  |---|---|---|
  | 2025-07 | o3 | −80% |
  | 2024-06, 2025-02, 2026-04 | mistral-large | −50%, −33%, −75% |
  | 2025-02 | o1-mini | −32% |
  | 2026-08 | claude-sonnet-5 | −33% |
  | 2026-09 | gemini-3.6-flash | −50% |

  The 2026-09 xAI rows (grok-4, grok-3 and grok-3-mini all set to an identical $1.5625) look like a
  bad LiteLLM snapshot rather than a real repricing. They also show up in the price conflict report.

## Charts

- [Price history by tier, log scale, with fitted lines](curve_v0_tier_history.png)
- [Quality-adjusted price, all model-months adjusted to ECI 135](curve_v0_hedonic.png)
- [Forward curve with the slow–fast scenario band](curve_v0_forward.png)
- [Backtest per tier](curve_v0_backtest.png)

The charts are static, light-mode PNGs: three tiers, in fixed colours blue, orange and aqua.

## Judgement calls

1. **Tier cut-offs.** I read "frontier = top 10%" as *within 10% of the best model's ECI* (≥ 0.90 ×
   best), not as the top decile of models. Mid and small are split at 0.80. The ECI scale has no
   meaningful zero, so these ratios are really fixed point gaps: 0.90 × 173 is about 17 ECI points
   today.
2. **Source priority.** The model maker's own price comes first, and llm-prices is preferred over
   hosting-provider medians.
3. **Undated aliases.** Undated aliases (such as `gpt-4o`) are separate models from their dated
   snapshots where the prices differ. This follows the model map rules.
4. **Clustered errors for the hedonic fit.** Without clustering, the standard error on the time
   coefficient would be 2.6× smaller (0.055 vs 0.143), because the same model appears in many
   months. The CI would be falsely tight.
5. **Thin tier-months.** Tier-months with fewer than 3 models are dropped. Before mid-2024 there are
   very few ECI-scored priced models: 2 in 2023-10 and 12 in 2024-06.
6. **Projection start.** The forward curve starts from a 3-month geometric mean, not the fitted
   line. That keeps it at today's observed level, but it is sensitive to the most recent months.
7. **Fast scenario.** "fast" uses the steeper of the CI bound and the last-12-month rate, as
   specified. For mid, that last-12-month rate is *rising* (×3.7 per year), so fast falls back to
   the CI bound.
8. **Outlier exclusion.** Outliers are excluded only from the fits. They remain in
   `panel_v0.parquet`.

## Limitations

- **Missing early models.** The best-known early models (GPT-4, GPT-4o, Claude 3.5 Sonnet) have no
  ECI score. Their Epoch entries are ambiguous dated variants that the model map deliberately did
  not resolve, so 2023–24 is thin and biased toward whichever models happened to match.
- **Stale listings.** LiteLLM keeps deprecated models listed, which inflates prices in older-model
  tiers (see Results).
- **Hosting-provider medians.** 45% of model-months are priced from hosting-provider medians, which
  include markups and regional surcharges.
- **ECI is a single point in time.** Each model has one ECI. Capability does not change over time,
  but the relative tiers do, which is the intended design.
- **No usage weighting.** Each model counts equally, whether or not anyone uses it.
- **Price only.** No accounting for token efficiency: reasoning models use far more output tokens
  per task.
- **OpenRouter data.** There is a single OpenRouter snapshot, so it adds almost nothing yet.

## Suggested changes for v1

1. **Use the hedonic rate as the published headline,** preferably on launch prices or recent models
   only.
2. **Replace relative-median tiers** with one of:
   - absolute ECI bands, for example "cheapest model with ECI ≥ 140", which is how Epoch measures
     this; or
   - the cheapest or lower-quartile price per tier, which ignores legacy listings.
3. **Match the ambiguous Epoch variants** to the new dated canonical IDs by release month to fill
   2023–24.
4. **Drop models from the panel after their deprecation date,** or after N months without any price
   change.
