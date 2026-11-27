# Token price curve v4: prices per token type, plus a capability lookup

Built by `uv run python scripts/build_curve_v4.py`, after `build_curve.py` (v3). v4 reads v3's files and
never writes them. v3's tiers, base levels and base-case rates are unchanged; v4 only adds detail. All
numbers are in [`curve_v4_results.json`](curve_v4_results.json).

**What v4 adds.**
1. **Separate prices for input, output, cache-read and cache-write tokens**, per tier and scenario, for
   months 0–36. v3 stored only a blended price (3:1 input:output).
2. **A capability lookup:** for every whole ECI threshold from 120 to 169, the cheapest model at or above
   it, each month since 2023-10, and projected 36 months forward under each scenario.
3. **A `small` median tier.** It covers models 15–30 ECI points below the best priced model that month,
   released in the last 12 months, and is built and projected exactly like v3's mid and frontier tiers.
   v3's base-case rule finds no significant break and no significant trend (p = 0.16 for a break), so
   it is held flat. A tier 30–50 points below has no recent models, so there is no floor median tier.

**Check.** v4 rebuilds v3's tier history exactly (largest difference 0). Its blended prices match v3's
index in every month and its base levels in every tier.

## Prices today (2026-11), USD per 1M tokens

| Tier | Input | Output | Cache read | Cache write | Blended (= v3) | Base case (from v3) |
|---|---|---|---|---|---|---|
| ECI ≥ 120, ≥ 130 | 0.05 | 0.20 | 0.05 (none listed) | 0.05 (none listed) | 0.0875 | flat |
| ECI ≥ 140, ≥ 150, ≥ 160 | 0.10 | 0.50 | 0.01 | 0.125 | 0.20 | ÷1.29 per year |
| small (v4 only) | 0.50 | 2.75 | 0.32 | 0.50 | 1.0625 (v3's method) | flat |
| mid | 2.00 | 10.00 | 0.20 | 2.33 | 4.00 | flat |
| frontier | 4.50 | 22.50 | 0.35 | 5.625 | 9.00 | flat |

Output costs 4–5× input in every tier today. That ratio has been stable: the median over each tier's
history is 4–6×.

## Method

- **One price structure per tier.** Every token type in a tier comes from the same model or models.
  - Bands (ECI ≥ N): the current cheapest model's own prices, each the median of its last 3 months.
    This is v3's base-level rule.
  - Small, mid and frontier: the **median model by blended price**, as set out in the next section.
- **Cache read and cache write.**
  - Source: the cache-read ÷ input and cache-write ÷ input ratios listed by the source the model's input
    price comes from. That
    is the maker's own LiteLLM rows, v3's reference provider, the median of hosting providers, or
    OpenRouter.
  - No listed price: where that source lists no price, those tokens are priced as ordinary input,
    and `cache_read_listed` / `cache_write_listed = false`. A cache write can cost more than input:
    Anthropic and OpenAI list 1.25× input for Claude Haiku 5.5 and the GPT-6 models.
  - Data: cache-write prices come from LiteLLM's `cache_creation_input_token_cost` (3,284 model-months,
    all from LiteLLM) and OpenRouter's `input_cache_write`. Adding them changed no other value in
    `price_history.parquet` or `prices.parquet`.
  - Effect: gpt-oss-20b (ECI ≥ 120 and ≥ 130) has no discount, because its reference provider,
    together_ai, lists no cache price, while Fireworks and Groq list about 50%. In the mid and
    frontier tiers, 86% and 100% of the models list a cache-read price; 40% and 88% list a cache-write
    price.
  - In the small tier, one of the two median models (Kimi K2 Thinking) has no cache price at its
    reference provider. Its cache reads are priced at its $0.60 input price, which is why small's cache
    read is $0.32.
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

### Median tiers: one model's prices, not four separate medians

**Before (until 2026-11).** Each token type was a separate median:
- Each month, the median input price across the tier's models, the median output price, the median
  cache-read price and the median cache-write price.
- Then, for each type, the median of the last 3 months.

So a tier could combine one model's input price with another model's cache price. Its blended price
matched v3 only by chance.

**After.** Each tier takes the full price set of its **median model by blended price**:
- **Each month:** sort the tier's models by blended price and take the middle one. With an even count,
  average the middle two. Models tied at a middle price are averaged first, so the result doesn't
  depend on row order.
- **Base level:** v3's base level is the median of the last 3 monthly medians, so it is one of those
  months. The base takes that month's price set, or the later month if two tie.
- **Result:** the blended price equals v3's index exactly, and every token type comes from the same
  model or models.

**Why this option, not median ratios.** The other option was to apply median ratios (output ÷ input,
cache ÷ input) to the median input price.
- **Blended price:** ratios on the median input price don't reproduce v3's blended index. The
  median-model price set does, in every month.
- **Real price lists:** a median model's prices are an actual price list, so cache discounts and
  cache-write surcharges keep the pattern a real provider uses. Median ratios can mix one model's
  output ratio with another's cache ratio.
- **Traceability:** the models behind each month are recorded (`median_models` in
  `tier_history_v4.csv`).

**Before and after, base level (USD per 1M tokens).**

| Tier | | Input | Output | Cache read | Cache write | Blended | Median models |
|---|---|---|---|---|---|---|---|
| small | before | 0.55 | 2.50 | 0.15 | 0.425 | 1.0375 (≠ 1.0625) | separate medians |
| small | after | 0.50 | 2.75 | 0.32 | 0.50 | 1.0625 | Kimi K2 Thinking, Qwen 3.8 27B (Nov 2026) |
| mid | before | 2.00 | 10.00 | 0.25 | 2.00 | 4.00 | separate medians |
| mid | after | 2.00 | 10.00 | 0.20 | 2.33 | 4.00 | Claude Sonnet 5, Kimi K3.1 Pro, GPT-6 Sol, tied (Nov 2026) |
| frontier | before | 4.50 | 22.50 | 0.225 | 5.625 | 9.00 | separate medians |
| frontier | after | 4.50 | 22.50 | 0.35 | 5.625 | 9.00 | Claude Opus 5, Claude Opus 5.5 (Oct 2026) |

- **Input and output:** almost unchanged. Most models in these tiers charge output at 5× input, so the
  separate medians happened to line up.
- **Cache prices:** these change most.
  - Mid cache read falls from 12.5% to 10% of input, and cache write rises from 1.0× to 1.17×. Two of
    the three tied models list a 1.25× write price.
  - Frontier cache read rises from 5% to 7.8% of input.
- **Small:** it didn't exist before. Separate medians would have given $1.04 blended, not the $1.06
  its index shows.

### Capability lookup today

| Minimum ECI | Cheapest model | Input | Output | Cache read | Cache write | Rate taken from |
|---|---|---|---|---|---|---|
| 120–134 | gpt-oss-20b | 0.05 | 0.20 | 0.05 | 0.05 | ECI ≥ 130 band (same model): flat |
| 135–139 | gpt-5-nano | 0.05 | 0.40 | 0.005 | 0.05 | ECI ≥ 130 band (nearest below): flat |
| 140–160 | Claude Haiku 5.5 | 0.10 | 0.50 | 0.01 | 0.125 | ECI ≥ 160 band (same model): ÷1.29 per year |
| 161–167 | GPT-6.1 Terra | 1.00 | 5.00 | 0.05 | 1.25 | ECI ≥ 160 band (nearest below): ÷1.29 per year |
| 168 | GPT-6.1 Sol | 2.00 | 10.00 | 0.10 | 2.50 | ECI ≥ 160 band: ÷1.29 per year |
| 169 | GPT-6 Astra | 10.00 | 50.00 | 1.00 | 12.50 | ECI ≥ 160 band: ÷1.29 per year |

The workflow layer's capability bands (`config/capability_bands.yaml`) read this lookup at each band's
minimum ECI: floor 120, small 140, mid 154 and frontier 164.

![Capability lookup](curve_v4_capability.png)

### Where the frontier cache-read price comes from

Now $0.35. Before the change to median models it was $0.225.
- **Base month.** v3's frontier base level is the median of three monthly median blended prices:
  $10.625 (Sep), $9.00 (Oct) and $6.00 (Nov 2026). That is October's $9.00, so the base takes
  October's median-model prices.
- **October.** Six models were in the tier. Sorted by blended price: GPT-6.1 Sol $4, Claude Sonnet 5.5
  $4, Claude Opus 5.5 $8, Claude Opus 5 $10, Claude Fable 5.1 $20 and GPT-6 Astra $20. The middle two
  are Opus 5.5 and Opus 5. Averaging their full price lists gives input $4.50, output $22.50, cache
  read ($0.20 + $0.50) ÷ 2 = $0.35 and cache write ($5.00 + $6.25) ÷ 2 = $5.625. The blended price is
  $9.00, v3's base level.
- **Source.** These are Anthropic's own list prices, from LiteLLM's first-party rows (provider
  `anthropic`). No hosting provider is involved.
- **The old $0.225.** That was a separate median of cache-read prices: October's middle two by cache
  price were Sonnet 5.5 or Opus 5.5 ($0.20) and Fable 5.1 ($0.25). Their input prices were not the
  tier's input price.
- **Lag.** The 3-month median lags. Nov 2026's median models alone would give $0.15: Opus 5.5's $0.20
  averaged with $0.10, the cache-read price of all three models tied at $4 (Sonnet 5.5, Gemini 4 Argon
  and GPT-6.1 Sol).

## Judgement calls

1. **One rate for all token types.** v3's rates were fitted to blended prices. Separate fits per type
   would double the number of noisy series, and the output:input ratio has barely moved.
2. **No cache price listed means no discount.** This overstates cache-read cost where a provider
   offers caching but LiteLLM doesn't list it. It is the conservative choice for a cost model.
3. **Thresholds above 160 use the ECI ≥ 160 rate.** The models there are frontier-priced
   ($2–20 blended). v2 found that a newly reached capability level falls in steps, 1–9 months after it
   first appears, and that the timing can't be forecast. v4 doesn't model a step. A smooth ÷1.29 per
   year is a middle path between the frontier tier (flat) and a step to $0.20.
4. **No cache-write price listed means the input price.** That is right for providers that charge
   nothing extra for writes. It understates cost where a provider charges about 1.25× input but
   LiteLLM doesn't list it.
5. **The small median tier is relative, like mid and frontier.** A median over fixed ECI 140–154 falls
   ÷3.3 per year, because as the frontier moves, more cheap new models land in that range. v3's rule
   would project that decline, and the median would drop below the cheapest model within about 22
   months. That repeats v1's mistake of projecting a one-off collapse. Defined relative to the best
   model, the tier shows no significant trend.

## Limitations

- **Thin data at the top.** 14 priced models qualify at ECI ≥ 161, but only 3 at ≥ 168 and 1 at ≥ 169.
  The price jumps tenfold from ≥ 160 ($0.20, Claude Haiku 5.5) to ≥ 161 ($2.00). A profile that sets
  `min_eci` near 160 is sensitive to the exact ECI scores.
- **v3's limitations still apply.** These are price only, no usage weighting, and ECI coverage. See
  [curve_v3.md](curve_v3.md).

## Files

| File | Content |
|---|---|
| [`data/curves/curve_v4.csv`](../data/curves/curve_v4.csv), [`.json`](../data/curves/curve_v4.json) | Per tier, scenario and month (months_ahead 0–36): `input`, `output`, `cache_read`, `cache_write`, `blended`, `annual_factor`. The JSON adds each tier's base per type and v3's reason. |
| [`data/curves/tier_history_v4.csv`](../data/curves/tier_history_v4.csv) | v3's monthly tier history with input, output and cache-read prices. |
| [`data/curves/capability_v4.csv`](../data/curves/capability_v4.csv) | Capability lookup: `month`, `eci_min`, `scenario` (`history` for past months), `model`, `model_eci`, prices per type, `rate_from`. |
| [`curve_v4_capability.png`](curve_v4_capability.png) | The chart above. |
