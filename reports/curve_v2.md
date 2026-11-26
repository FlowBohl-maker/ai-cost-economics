# Token price curve v2: one floor rate plus a catch-up lag

> **Data note (2026-11-26).** The numbers below were computed with llm-prices.com in the price data. That source declares no licence, so the committed data and curve files are now built without it (see the README). Re-running gives somewhat different numbers: the rolling backtest gives 77% for v2 and 53% for "no change", against 82% and 53% here. The conclusions are unchanged.

Built by `uv run python scripts/build_curve.py --version v2` (now the default). v0 and v1 still
reproduce with `--version v0` / `--version v1`, with unchanged numbers. Bands and index are the same
as [v1](curve_v1.md): the cheapest model at or above each ECI threshold. Only the forecast changes.
All numbers are in [`curve_v2_results.json`](curve_v2_results.json).

**Bottom line.**
- **Better than v1:** v2 removes v1's runaway projections. On the 12-month backtest its error is
  74% for ECI ≥ 130, against 238% for v1, and 34% for ECI ≥ 140, against 380%. The 36-month curve now
  ends at about $0.02–0.04, not $0.000009.
- **Still worse than "no change":** v2 loses to a flat forecast on both the 12-month backtest and a
  rolling 6-month backtest (MALE 82% vs 53%).
- **Both parts fail:**
  - **Floor rate:** the floors have not fallen since late 2025, so any declining rate loses.
  - **Lag:** the catch-up lag is too variable (1 to 9 months) to time the drop.
- **Published base case: no change.** The curve files now hold three scenarios: `base` keeps
  today's price flat, `decline` is v2's rate-plus-lag forecast ("the 2024–25 decline resumes") and
  `fast_decline` is its fast variant.

## Method

Each band is in one of two phases:

| Phase | When | Forecast |
|---|---|---|
| **Floor** | the band's price has been within 3× of the band below at least once | latest price × the floor rate |
| **Drop** | the band has never been within 3× of the band below | holds its latest price, then **steps** down to the band below's path once the catch-up lag has passed |

- **Floor rate.** The log-linear rate of the lowest band (ECI ≥ 120), applied to every band in its
  floor phase. Scenarios use v0's rules: fast = the steeper of the lower 95% bound and the last-12-month
  rate, slow = the upper bound.
- **Catch-up lag.** Months from a band's first month until it is within 3× of the band below.
  Base = median observed lag, fast = shortest, slow = longest. A band past its due date is expected
  to step down next month.
- **Anchor.** The latest month's price, not v1's 3-month average. In a step model, prices sit flat
  between steps. A 3-month average would have started ECI ≥ 160 at $0.54 instead of today's $0.20.
- **ECI ≥ 160 added.** It has only 8 months of data, too few for v1's per-band fit. v2 needs no
  per-band fit.

### Observed lags

| Band | First month | Price then | Within 3× of band below | Lag |
|---|---|---|---|---|
| ECI ≥ 130 | 2025-01 | $1.93 (o3-mini) | 2025-02 | 1 month |
| ECI ≥ 140 | 2025-02 | $1.93 (o3-mini) | 2025-09 | 7 months |
| ECI ≥ 150 | 2025-11 | $3.44 (GPT-5) | 2026-08 | 9 months |
| ECI ≥ 160 | 2026-04 | $11.25 | 2026-10 | 6 months |

The median is 6.5 months and the range is 1–9.

## Forward curve (blended USD per 1M tokens, from 2026-11)

| Scenario | What it assumes |
|---|---|
| **base** | No change: each band's latest monthly price held flat. It beat every fitted forecast in both backtests. |
| decline | The decline resumes: the ECI ≥ 120 rate (÷1.73 per year) for every band in its floor phase. A band still in its drop phase steps down after the median lag. |
| fast_decline | The steeper of the rate's 95% CI bound and its last-12-month rate (÷2.05 per year), with the shortest observed lag. |

v1's "slow" scenario is dropped. At ÷1.46 per year it fell between `base` and `decline`.

Every band is currently in its floor phase:

| Band | Now = base | +12 months: decline (fast) | +36 months: decline (fast) |
|---|---|---|---|
| ECI ≥ 120, ≥ 130 | 0.123 | 0.071 (0.060) | 0.024 (0.014) |
| ECI ≥ 140, ≥ 150, ≥ 160 | 0.200 | 0.116 (0.098) | 0.039 (0.023) |

The base anchors at one month's price. For ECI ≥ 120 that month is at the top of its recent noise
($0.0875–0.1225, see Limitations), so the base could plausibly sit about 30% lower.

The lag part only matters when a new threshold is reached. That will likely be ECI ≥ 170, which so
far has 2 scored models, both priced at $16.

## Backtests

### 12 months (fit to 2025-11, compared with 2025-12 to 2026-11)

MALE = mean absolute log error, in %.

| Band | Phase at cutoff | v2 MALE | v1 MALE | "No change" MALE | v2 with floors held flat |
|---|---|---|---|---|---|
| ECI ≥ 120 | floor | 71% | 60% | **8%** | 8% |
| ECI ≥ 130 | floor | 74% | 238% | **10%** | 10% |
| ECI ≥ 140 | floor | 34% | 380% | **18%** | 18% |
| ECI ≥ 150 | drop | 231% | not testable | 208% | **154%** |

- **"No change" baseline.** v2's "no change" starts from the latest month, so it differs from v1's
  column, which started from a 3-month average.
- **Floor rate.** The rate fitted to 2025-11 was ÷2.3 per year. The floors then stayed flat or rose
  slightly, so the forecast ends 41–67% too low.
- **ECI ≥ 150.** With only two lags known at the cutoff (1 and 7 months), the base lag was 4 months.
  The model expected the drop in 2026-03, but it came in 2026-08.

### Rolling origin, 6 months ahead (17 cutoffs, 2025-01 to 2026-05)

This is a second check, so the conclusion doesn't rest on a single cutoff.
[`curve_v2_rolling_backtest.csv`](curve_v2_rolling_backtest.csv) has every error.

| | v2 | Floors held flat (lag only) | "No change" |
|---|---|---|---|
| All bands | 82% | 67% | **53%** |
| Bands in floor phase | 28% | **19%** | **19%** |
| Bands in drop phase | 340% | 284% | **183%** |
| ECI ≥ 160 only (2 cutoffs) | 218% | **204%** | 326% |

- **Floor rate.** Adding a decline rate makes floor-phase forecasts worse (28% vs 19%).
- **Lag.** The step comes too early at most cutoffs: the lags observed so far have grown longer
  (1, then 7, then 9 months). It helped only for ECI ≥ 160.

## Judgement calls

1. **Convergence test: within 3× of the band below.** The observed spread after convergence ranges
   from 1× (≥ 150 vs ≥ 140 now) to about 2.5× (≥ 140 vs ≥ 130). A 2× test would mark ≥ 140 as
   converging only in 2026-07, which would make its lag 17 months.
2. **A band stays in its floor phase once it has converged.** Without this rule, ECI ≥ 140 flips
   between phases as its gap moves around 3×.
3. **Step, not glide.** An earlier run glided log-linearly to the floor. That run also lacked
   rule 2 and used a 3-month anchor, and it did worse (rolling MALE 103%). Real drops arrive in one
   or two steps.
4. **The step targets the band below exactly (1×).** Observed convergence lands at 1–2.6×, so this
   biases drop forecasts low.
5. **Censored lags are ignored.** A band that hasn't converged yet doesn't count toward the median.
   That biases the lag short, which is part of why the steps come too early.

## Limitations

- **Noisy floors.** Floor prices are medians across hosting providers. The set of providers
  LiteLLM lists changes from month to month. Example: gpt-oss-20b moves between $0.0875 and $0.1225
  with no repricing. Floors therefore swing ±40%, which no trend can forecast.
- **Very few lag observations.** There are only four, and each depends on when Epoch scored a cheap
  model, not just when one existed.
- **v1's limitations still apply.** These are ECI coverage, missing early flagship scores and
  price-only measurement.

## What this suggests

The price at each capability level has followed a step pattern: expensive at first, then near
$0.10–0.20 within a year, then flat. Over the last 12 months, cheaper capability came from **new
bands reaching the floor**, not from the floor falling. That points to two changes for the
published curve:
- **Headline the time to reach the floor.** Track how long after a capability level first appears
  it becomes available at $0.20. The observed lags are 1, 7, 9 and 6 months. Don't headline a
  constant decline rate per band.
- **Forecast with "no change" as the base case.** This is now done in the curve files, with v2's
  ÷1.7 per year as the "decline resumes" scenario.

A cheaper data fix should come first: price floors from a fixed provider, or from the cheapest
first-party or major provider, rather than the median across all hosting providers. That would
remove the ±40% noise.

## Files

- [`data/curves/curve_v2.csv`](../data/curves/curve_v2.csv) and
  [`curve_v2.json`](../data/curves/curve_v2.json): the forward curve. The JSON includes each band's
  phase and anchor under `forecast_state`.
- [`data/curves/tier_history_v2.csv`](../data/curves/tier_history_v2.csv): the cheapest model behind
  each month, including ECI ≥ 160.
- Charts:
  - [history](curve_v2_tier_history.png)
  - [forward curve](curve_v2_forward.png)
  - [backtest](curve_v2_backtest.png)
