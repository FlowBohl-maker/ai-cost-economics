# Token price curve v4: prices per token type, plus a capability lookup

Built by `uv run python scripts/build_curve_v4.py`, after `build_curve.py` (v3). v4 reads v3's files and
never writes them. v3's tiers, base levels and base-case rates are unchanged; v4 only adds detail. All
numbers are in [`curve_v4_results.json`](curve_v4_results.json).

**What v4 adds.**
1. **Separate prices for input, output and cache-read tokens**, per tier and scenario, for months 0–36.
   v3 stored only a blended price (3:1 input:output).
2. **A capability lookup:** for every whole ECI threshold from 120 to 169, the cheapest model at or above
   it, each month since 2023-10, and projected 36 months forward under each scenario.

**Check.** v4 rebuilds v3's tier history exactly (largest difference 0), and its blended base prices
match v3's base levels exactly for every tier.

## Prices today (2026-11), USD per 1M tokens

| Tier | Input | Output | Cache read | Blended (= v3) | Base case (from v3) |
|---|---|---|---|---|---|
| ECI ≥ 120, ≥ 130 | 0.05 | 0.20 | 0.05 (no cache price listed) | 0.0875 | flat |
| ECI ≥ 140, ≥ 150, ≥ 160 | 0.10 | 0.50 | 0.01 | 0.20 | ÷1.29 per year |
| mid | 2.00 | 10.00 | 0.25 | 4.00 | flat |
| frontier | 4.50 | 22.50 | 0.225 | 9.00 | flat |

Output costs 4–5× input in every tier today. That ratio has been stable: the median over each tier's
history is 4–6×.

## Method

- **Input and output.**
  - Bands (ECI ≥ N): the current cheapest model's own input and output prices, each the median of its
    last 3 months. This is v3's base-level rule applied to each price separately.
  - Mid and frontier: the median of the tier's monthly median input and output prices over the last 3
    months. A median per type need not come from the same model as the blended median. Today the
    blended price still matches v3 exactly.
- **Cache read.**
  - Source: the cache-read ÷ input ratio listed by the source the model's input price comes from. That
    is the maker's own LiteLLM rows, v3's reference provider, the median of hosting providers, or
    OpenRouter.
  - No listed price: where that source lists no cache-read price, cache reads are priced as ordinary
    input (no discount), and `cache_read_listed = false`.
  - Effect: gpt-oss-20b (ECI ≥ 120 and ≥ 130) has no discount, because its reference provider,
    together_ai, lists no cache price, while Fireworks and Groq list about 50%. In the mid and
    frontier tiers, 86% and 100% of the models list a cache price.
- **Forward curve.** Each token type moves at its tier's v3 annual factor in each scenario
  (`base`, `fast`, `slow`). v3 fits rates on blended prices only. The stable output:input ratio
  suggests input and output prices move together.
- **Capability lookup** (`capability_v4.csv`).
  - History: for each threshold and month, the cheapest model with ECI at or above it (lowest blended
    price), whatever the number of qualifying models. `n_models` gives the count. Months before
    2026-11 have `scenario = history`.
  - Forward: today's cheapest model at or above the threshold, starting from the median of its last
    3 months. Its rate comes from the v3 band priced by the same model (v3's "same model" rule).
    Otherwise it comes from the nearest v3 band below the threshold.

### Capability lookup today

| Minimum ECI | Cheapest model | Input | Output | Cache read | Rate taken from |
|---|---|---|---|---|---|
| 120–134 | gpt-oss-20b | 0.05 | 0.20 | 0.05 | ECI ≥ 130 band (same model): flat |
| 135–139 | gpt-5-nano | 0.05 | 0.40 | 0.005 | ECI ≥ 130 band (nearest below): flat |
| 140–160 | Claude Haiku 5.5 | 0.10 | 0.50 | 0.01 | ECI ≥ 160 band (same model): ÷1.29 per year |
| 161–167 | GPT-6.1 Terra | 1.00 | 5.00 | 0.05 | ECI ≥ 160 band (nearest below): ÷1.29 per year |
| 168 | GPT-6.1 Sol | 2.00 | 10.00 | 0.10 | ECI ≥ 160 band: ÷1.29 per year |
| 169 | GPT-6 Astra | 10.00 | 50.00 | 1.00 | ECI ≥ 160 band: ÷1.29 per year |

![Capability lookup](curve_v4_capability.png)

## Judgement calls

1. **One rate for all token types.** v3's rates were fitted to blended prices. Separate fits per type
   would double the number of noisy series, and the output:input ratio has barely moved.
2. **No cache price listed means no discount.** This overstates cache-read cost where a provider
   offers caching but LiteLLM doesn't list it. It is the conservative choice for a cost model.
3. **Thresholds above 160 use the ECI ≥ 160 rate.** The models there are frontier-priced
   ($2–20 blended). v2 found that a newly reached capability level falls in steps, 1–9 months after it
   first appears, and that the timing can't be forecast. v4 doesn't model a step. A smooth ÷1.29 per
   year is a middle path between the frontier tier (flat) and a step to $0.20.
4. **Cache writes are not priced.** Some providers charge for writing to the cache (about 1.25× input).
   Few sources list it, so profiles should count those tokens as input.

## Limitations

- **Thin data at the top.** 14 priced models qualify at ECI ≥ 161, but only 3 at ≥ 168 and 1 at ≥ 169.
  The price jumps tenfold from ≥ 160 ($0.20, Claude Haiku 5.5) to ≥ 161 ($2.00). A profile that sets
  `min_eci` near 160 is sensitive to the exact ECI scores.
- **v3's limitations still apply.** These are price only, no usage weighting, and ECI coverage. See
  [curve_v3.md](curve_v3.md).

## Files

| File | Content |
|---|---|
| [`data/curves/curve_v4.csv`](../data/curves/curve_v4.csv), [`.json`](../data/curves/curve_v4.json) | Per tier, scenario and month (months_ahead 0–36): `input`, `output`, `cache_read`, `blended`, `annual_factor`. The JSON adds each tier's base per type and v3's reason. |
| [`data/curves/tier_history_v4.csv`](../data/curves/tier_history_v4.csv) | v3's monthly tier history with input, output and cache-read prices. |
| [`data/curves/capability_v4.csv`](../data/curves/capability_v4.csv) | Capability lookup: `month`, `eci_min`, `scenario` (`history` for past months), `model`, `model_eci`, prices per type, `rate_from`. |
| [`curve_v4_capability.png`](curve_v4_capability.png) | The chart above. |
