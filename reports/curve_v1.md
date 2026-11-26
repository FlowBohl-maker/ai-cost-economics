# Token price curve v1: absolute ECI bands

> **Data note (2026-11-26).** The numbers below were computed with llm-prices.com in the price data. That source declares no licence, so the committed data and curve files are now built without it (see the README). Re-running gives somewhat different numbers: ECI ≥ 130 now fits at ÷3.0 per year instead of ÷2.0, and ECI ≥ 150 at ÷40 instead of ÷28; "no change" still wins every backtest. The conclusions are unchanged.

Built by `uv run python scripts/build_curve.py --version v1`. The script still reproduces v0 with
`--version v0`. Data, price measure, hedonic regression, scenarios and backtest are unchanged from
[v0](curve_v0.md). The only change is how tiers are defined. All numbers are in
[`curve_v1_results.json`](curve_v1_results.json).

**Bottom line.**
- **Historical decline:** the cheapest price for a fixed capability level fell 1.7× to 28× per year,
  and faster for higher capability levels. Unlike v0, every band is falling, and the fits are better
  (R² 0.34–0.85).
- **Not usable as a forecast:** each band collapses quickly after it is first reached, then levels
  off at about $0.10–0.20 per million tokens. A log-linear fit projects the collapse forward.
- **Backtest:** the fitted lines missed by 60–380% over the last 12 months. A naive "no change"
  forecast missed by only 12–18%. Don't publish the v1 forward curve as it stands.

## What changed from v0

| | v0 | v1 |
|---|---|---|
| Tier | ECI relative to the best model that month | Fixed ECI threshold: ≥ 120, ≥ 130, ≥ 140, ≥ 150 |
| Index | Median blended price of models in the tier | Blended price of the **cheapest** model at or above the threshold |
| Start of each series | First month with ≥ 3 models in the tier | First month with ≥ 3 scored models at or above the threshold |

The bands are nested: a model with ECI 156 counts toward all four. This answers "what does it cost
to get at least this much capability?", the measure Epoch uses. Because the index is the *cheapest*
model, old expensive models that are still listed no longer matter, which removes v0's main
artefact. The cheapest model behind every month is in
[`data/curves/tier_history_v1.csv`](../data/curves/tier_history_v1.csv).

## Results

### Annual decline (cheapest model at or above the threshold)

| Band | Falls per year | 95% CI | R² | Months fitted | Last 12 months | Price now |
|---|---|---|---|---|---|---|
| ECI ≥ 120 | ÷1.7 | ÷1.5–÷2.0 | 0.75 | 29 (from 2024-07) | ×1.3 (rising) | $0.098 |
| ECI ≥ 130 | ÷2.0 | ÷1.1–÷3.4 | 0.34 | 23 (from 2025-01) | ×1.4 (rising) | $0.104 |
| ECI ≥ 140 | ÷4.6 | ÷2.8–÷7.5 | 0.75 | 22 (from 2025-02) | ÷1.4 | $0.199 |
| ECI ≥ 150 | ÷28 | ÷11–÷72 | 0.85 | 13 (from 2025-11) | ÷39 | $0.200 |
| *Hedonic, constant ECI (from v0)* | *÷1.9* | *÷1.4–÷2.5* | *0.31* | | | |

"Price now" is the geometric mean of the last 3 months.

The pattern is the same for every band:
1. **First reached:** the threshold is first crossed by an expensive flagship. ECI ≥ 140 starts with
   o3-mini at $1.93. ECI ≥ 150 starts with GPT-5 at $3.44, followed by Kimi K2.6 at $1.71.
2. **Fast drop:** within 8–13 months a cheap model reaches the same level. For ECI ≥ 140 that was
   gpt-oss-120b at $0.27; for ECI ≥ 150 it was GPT-6 Luna and Claude Haiku 5.5 at $0.20.
3. **Floor:** the price then levels off. ECI ≥ 120 and ≥ 130 have sat at about $0.09–0.12 since
   Oct 2025.

The older bands (≥ 120, ÷1.7) end up close to the hedonic rate (÷1.9). The newer bands look steep
because their whole history is the drop phase.

### Forward curve (blended USD per 1M tokens)

| Band | Now | +12 months: base (fast–slow) | +36 months: base (fast–slow) |
|---|---|---|---|
| ECI ≥ 120 | 0.098 | 0.057 (0.048–0.067) | 0.019 (0.011–0.032) |
| ECI ≥ 130 | 0.104 | 0.053 (0.030–0.092) | 0.014 (0.0026–0.071) |
| ECI ≥ 140 | 0.199 | 0.044 (0.026–0.071) | 0.0021 (0.0005–0.0092) |
| ECI ≥ 150 | 0.200 | 0.0071 (0.0028–0.018) | 0.000009 (0.0000005–0.00015) |

Even the "slow" scenario falls faster than any band did over the last year, except ECI ≥ 150. The
≥ 140 and ≥ 150 projections (fractions of a cent per million tokens) are not credible. They project
a one-time drop as if it would continue every year.

### Backtest (fit to 2025-11, compared with 2025-12 to 2026-11)

| Band | Rate fitted on training data | MALE | "No change" MALE | Bias | Error at 12 months | Months inside band |
|---|---|---|---|---|---|---|
| ECI ≥ 120 | 0.43 per year | 60% | **12%** | −38% | −67% ($0.040 vs $0.123) | 1 / 12 |
| ECI ≥ 130 | 0.11 per year | 238% | **13%** | −70% | −92% ($0.010 vs $0.123) | 0 / 12 |
| ECI ≥ 140 | 0.04 per year | 380% | **18%** | −79% | −94% ($0.011 vs $0.200) | 0 / 12 |
| ECI ≥ 150 | not tested: under 6 months of data before the cutoff | | | | | |

Every band's projection is far too low, and "no change" beats the fitted trend everywhere. I also
tried the hedonic rate as the forecast for every band, fitted to 2025-11 (÷3.4 per year). It too
was too low: MALE 98%, 101% and 65% for ≥ 120, ≥ 130 and ≥ 140. Over the last year the cheapest
prices have been flat at each capability level. At the top of the range, newer and better models
reach the same $0.10–0.20 price point, rather than any one level getting cheaper.

## Judgement calls (in addition to v0's)

1. **Thresholds 120 / 130 / 140 / 150.** Each has at least 13 fitted months. ECI ≥ 160 has only 9
   months with any model, so it was left out. The bands are nested, not exclusive.
2. **Cheapest model, not median.** This is robust to legacy listings, but sensitive to one
   mispriced row. Outliers (|studentized residual| > 3) are excluded, as in v0. A single data error
   at the bottom of a band would still move the index.
3. **Start when 3 scored models qualify.** Without this rule, ECI ≥ 120 starts in 2024-02 with
   Claude 3 Opus alone at $30, and the fitted decline is ÷4.3 instead of ÷1.7. That earlier start
   reflects which models Epoch happened to score, not market prices.
4. **Hosting-provider prices.** Floor prices are mostly open-weight models (gpt-oss, Qwen, Phi-4,
   Gemma) priced as the median across hosting providers. The cheapest single provider would be
   lower still.

## Limitations

v0's limitations still apply: no early GPT-4/GPT-4o ECI scores, stale listings, no usage weighting,
price only. In addition:
- **Steps, not a slope.** The cheapest-price series moves in steps when a new model launches, so a
  straight line on a log scale is a poor description of it.
- **ECI coverage bias.** Epoch mostly scores notable models. A cheaper model may already meet a
  threshold without having a score.
- **Short series.** ECI ≥ 150 has 13 months of history and no backtest.

## Suggested next step

The data supports a **step-then-floor** model better than a constant rate. Two simple options, both
still free of ARIMA/GARCH:
- **Fit on the floor phase only.** Use each band's data from the month its cheapest price first
  falls below a set share of its starting price. This measures how fast floors move.
- **One rate for all bands, plus a lag.** Use the rate from the older bands (≥ 120: ÷1.7, close to
  the hedonic ÷1.9) for every band. Treat a newly reached threshold as converging to the current
  floor price within 8–13 months (the gap observed for ≥ 140 and ≥ 150).

Either should be judged on the same backtest, with "no change" as the benchmark to beat.
