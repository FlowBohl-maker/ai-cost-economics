"""Token price curve: hedonic index, ECI tier indices, 36-month forward curve, 12-month backtest.

  --version v0  tiers relative to the best model each month, index = tier median price
  --version v1  absolute ECI thresholds, index = cheapest model at or above the threshold
  --version v2  v1 bands, forecast = one floor rate for every band plus a catch-up lag for newly reached bands
  --version v3  v2 bands priced from one reference provider per model, plus frontier and mid tiers;
                base case per tier from a structural-break (floor) test

Inputs:  data/price_history.parquet, data/prices.parquet (OpenRouter daily), data/model_map.csv, data/models.csv
Outputs: data/curves/curve_<v>.csv, curve_<v>.json, panel_<v>.parquet, tier_history_<v>.csv,
         reports/curve_<v>_*.png, reports/curve_<v>_results.json (numbers quoted in reports/curve_<v>.md)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_history import FIRST_PARTY  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT, REP = ROOT / "data", ROOT / "data" / "curves", ROOT / "reports"

INPUT_WEIGHT = 0.75            # 3:1 input:output
TIERS = [("frontier", 0.90, 9.99), ("mid", 0.80, 0.90), ("small", 0.0, 0.80)]  # ECI / best-ECI that month
MIN_MODELS = 3                  # tier-months with fewer models are not used
HORIZON = 36
BACKTEST_MONTHS = 12
ANCHOR_MONTHS = 3               # projection starts at the geometric mean of the last 3 tier medians
REPRICE_THRESHOLD = 0.30        # |month-on-month change| in blended price
RECENT_MONTHS = 18              # sensitivity: only models released within this many months
T0 = pd.Timestamp("2023-10-01")

ECI_THRESHOLDS = [120, 130, 140, 150]  # v1: >= 13 fitted months each; >= 160 has only 8
V2_THRESHOLDS = [120, 130, 140, 150, 160]  # v2 needs no fit per band, so >= 160 can be included
GAP = 3.0                       # v2: a band is in its floor phase once its price is within 3x of the band below
FALLBACK_LAG = 9                # v2: catch-up lag (months) when no band has converged yet
ROLL_HORIZON = 6                # v2: rolling-origin backtest horizon (months)
# v3: one reference provider per model for hosted (non-first-party) prices
REF_PREFERENCE = ["together_ai", "fireworks_ai", "deepinfra", "groq", "bedrock", "bedrock_converse", "azure_ai",
                  "vertex_ai", "nebius", "novita"]  # tie-break among providers with equally long price histories
GATEWAYS = {"openrouter", "vercel_ai_gateway", "aihubmix", "github_copilot", "opencode_go", "ollama"}  # resellers
BROKEN_KEY_FACTOR = 20          # a provider-month priced > 20x the cheapest provider that month is a units error
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # dataviz reference categorical slots 1-5, fixed order
# v3: tiers that answer "what do models near the frontier cost", by ECI points below the best priced model that month
GAP_TIERS = [("mid", 5, 15), ("frontier", 0, 5)]
GAP_TIER_MAX_AGE = 12           # months since release; keeps old, still-listed models out of the medians
GAP_COLORS = {"mid": "#008300", "frontier": "#6250d6"}  # reference slots 6 (green) and 7 (violet)
BREAK_MIN_SIDE = 6              # months on each side of a candidate break
BREAK_ALPHA = 0.05
N_BOOT = 999
BASE_MEDIAN_MONTHS = 3

# set in main() from --version
VERSION, TIER_NAMES, COLORS, INDEX_LABEL = "v1", [], {}, ""
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def years(d) -> np.ndarray:
    return ((pd.to_datetime(pd.Series(d)) - T0).dt.days / 365.25).to_numpy()


# ---------------------------------------------------------------- panel


def reference_prices(h: pd.DataFrame) -> pd.DataFrame:
    """v3: one hosting provider per model instead of the median over all of them.

    The reference is the provider (gateways excluded) with the most months of prices for the model; ties go
    to REF_PREFERENCE order, then name. Months where it has no price fall back to the median of the others.
    """
    keys = ["snapshot_date", "canonical_id"]
    o = h[(h.input_price_per_mtok > 0) & (h.output_price_per_mtok > 0)].copy()
    o["bl"] = INPUT_WEIGHT * o.input_price_per_mtok + (1 - INPUT_WEIGHT) * o.output_price_per_mtok
    o = o[o.bl <= BROKEN_KEY_FACTOR * o.groupby(keys).bl.transform("min")]
    per = o.groupby(keys + ["provider"], as_index=False)[["input_price_per_mtok", "output_price_per_mtok"]].median()
    cov = per[~per.provider.isin(GATEWAYS)].groupby(["canonical_id", "provider"]).size().rename("n").reset_index()
    cov["pref"] = cov.provider.map({p: i for i, p in enumerate(REF_PREFERENCE)}).fillna(len(REF_PREFERENCE))
    ref = (cov.sort_values(["canonical_id", "n", "pref", "provider"], ascending=[True, False, True, True])
           .drop_duplicates("canonical_id").set_index("canonical_id").provider)
    per["is_ref"] = per.provider.eq(per.canonical_id.map(ref))
    has_ref = per.groupby(keys).is_ref.transform("any")
    fallback = per[~has_ref].groupby(keys, as_index=False)[["input_price_per_mtok", "output_price_per_mtok"]].median()
    return pd.concat([per[per.is_ref].drop(columns="is_ref").assign(tier_src="litellm_ref"),
                      fallback.assign(tier_src="litellm_median")], ignore_index=True)


def build_panel(ref_provider: bool = False) -> pd.DataFrame:
    """One blended price per canonical model per month.

    Source priority: model maker's own LiteLLM rows (median over keys) > llm-prices > other LiteLLM providers
    (median of all, or with ref_provider one reference provider per model) > OpenRouter (monthly mean).
    """
    h = pd.read_parquet(DATA / "price_history.parquet")
    h = h[h.canonical_id.notna()].copy()
    vendor = h.canonical_id.str.split("/").str[0]
    h["first_party"] = (h.source == "litellm").to_numpy() & np.array(
        [p in FIRST_PARTY.get(v, set()) for p, v in zip(h.provider.fillna(""), vendor)], dtype=bool)
    h["tier_src"] = np.select(
        [h.first_party, h.source == "llm_prices", h.source == "litellm"], ["litellm_first_party", "llm_prices", "litellm_other"], "")

    if ref_provider:
        other = h.tier_src == "litellm_other"
        h = pd.concat([h[~other], reference_prices(h[other])], ignore_index=True)

    orp = pd.read_parquet(DATA / "prices.parquet")
    mp = pd.read_csv(DATA / "model_map.csv")
    orp = orp.merge(mp[mp.source == "openrouter"][["source_id", "canonical_id"]], left_on="model_id", right_on="source_id")
    orp["snapshot_date"] = orp.snapshot_date.dt.to_period("M").dt.to_timestamp()
    orp = orp.groupby(["snapshot_date", "canonical_id", "model_id"], as_index=False)[
        ["input_price_per_mtok", "output_price_per_mtok"]].mean()
    orp["tier_src"] = "openrouter"

    allp = pd.concat([h[["snapshot_date", "canonical_id", "input_price_per_mtok", "output_price_per_mtok", "tier_src"]],
                      orp[["snapshot_date", "canonical_id", "input_price_per_mtok", "output_price_per_mtok", "tier_src"]]])
    allp = allp[(allp.input_price_per_mtok > 0) & (allp.output_price_per_mtok > 0)]  # drop free / zero / missing
    rank = {"litellm_first_party": 0, "llm_prices": 1, "litellm_other": 2, "litellm_ref": 2, "litellm_median": 2, "openrouter": 3}
    allp["src_rank"] = allp.tier_src.map(rank)
    best = allp.groupby(["snapshot_date", "canonical_id"]).src_rank.transform("min")
    allp = allp[allp["src_rank"] == best]
    panel = allp.groupby(["snapshot_date", "canonical_id"], as_index=False).agg(
        input=("input_price_per_mtok", "median"), output=("output_price_per_mtok", "median"),
        price_source=("tier_src", "first"))
    panel["blended"] = INPUT_WEIGHT * panel.input + (1 - INPUT_WEIGHT) * panel.output

    models = pd.read_csv(DATA / "models.csv", parse_dates=["epoch_release_date"])[["canonical_id", "eci", "epoch_release_date"]]
    panel = panel.merge(models, on="canonical_id", how="left")
    panel = panel.rename(columns={"snapshot_date": "month"})
    panel["t"] = years(panel.month)
    panel["age_months"] = (panel.month - panel.epoch_release_date).dt.days / 30.44
    return panel.sort_values(["month", "canonical_id"]).reset_index(drop=True)


def assign_tiers(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel[panel.eci.notna()].copy()
    p["eci_best"] = p.groupby("month").eci.transform("max")
    p["eci_rel"] = p.eci / p.eci_best
    p["tier"] = pd.cut(p.eci_rel, [lo for _, lo, _ in TIERS[::-1]] + [9.99],
                       labels=[n for n, _, _ in TIERS[::-1]], right=False).astype(str)
    return p


# ---------------------------------------------------------------- fits


def decline(slope: float, lo: float, hi: float) -> dict:
    """Annual price multiplier exp(b) and the equivalent 'falls N x per year'."""
    return {"annual_factor": float(np.exp(slope)), "annual_factor_ci95": [float(np.exp(lo)), float(np.exp(hi))],
            "falls_x_per_year": float(np.exp(-slope)), "falls_x_per_year_ci95": [float(np.exp(-hi)), float(np.exp(-lo))],
            "log_slope_per_year": float(slope), "log_slope_ci95": [float(lo), float(hi)]}


def fit_hedonic(p: pd.DataFrame):
    X = sm.add_constant(p[["t", "eci"]])
    res = sm.OLS(np.log(p.blended), X).fit(cov_type="cluster", cov_kwds={"groups": p.canonical_id.astype("category").cat.codes})
    lo, hi = res.conf_int().loc["t"]
    out = decline(res.params["t"], lo, hi)
    out.update({"eci_coef": float(res.params["eci"]), "eci_coef_ci95": [float(x) for x in res.conf_int().loc["eci"]],
                "r_squared": float(res.rsquared), "n_obs": int(res.nobs), "n_models": int(p.canonical_id.nunique()),
                "se": "clustered by model"})
    return res, out


def tier_index(p: pd.DataFrame) -> pd.DataFrame:
    g = p.groupby(["month", "tier"]).agg(index_price=("blended", "median"), median_input=("input", "median"),
                                         median_output=("output", "median"), n_models=("canonical_id", "nunique"),
                                         eci_best=("eci_best", "first")).reset_index()
    g["used"] = g.n_models >= MIN_MODELS
    g["t"] = years(g.month)
    return g


def band_index(p: pd.DataFrame) -> pd.DataFrame:
    """v1: each month, the cheapest blended price among models with ECI >= threshold (nested bands)."""
    rows = []
    for thr in ECI_THRESHOLDS:
        q = p[p.eci >= thr]
        if q.empty:
            continue
        cheapest = q.loc[q.groupby("month").blended.idxmin()]
        n = q.groupby("month").canonical_id.nunique()
        for r in cheapest.itertuples():
            rows.append({"month": r.month, "tier": f"eci>={thr}", "index_price": r.blended, "index_input": r.input,
                         "index_output": r.output, "cheapest_model": r.canonical_id, "cheapest_eci": r.eci,
                         "price_source": r.price_source, "n_models": int(n[r.month])})
    g = pd.DataFrame(rows)
    # a band starts once >= MIN_MODELS scored models qualify (earlier months reflect Epoch coverage, not the market)
    start = g[g.n_models >= MIN_MODELS].groupby("tier").month.min()
    g["used"] = g.month >= g.tier.map(start)
    g["t"] = years(g.month)
    return g


def fit_line(df: pd.DataFrame):
    X = sm.add_constant(df[["t"]])
    lags = max(1, min(3, len(df) // 4))
    res = sm.OLS(np.log(df.index_price), X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    lo, hi = res.conf_int().loc["t"]
    out = decline(res.params["t"], lo, hi)
    out.update({"r_squared": float(res.rsquared), "n_months": int(res.nobs),
                "first_month": f"{df.month.min():%Y-%m}", "last_month": f"{df.month.max():%Y-%m}",
                "se": f"Newey-West HAC, {lags} lags"})
    return res, out


def fit_tiers(idx: pd.DataFrame, upto: pd.Timestamp | None = None) -> dict:
    fits = {}
    for tier in TIER_NAMES:
        d = idx[(idx.tier == tier) & idx.used]
        if upto is not None:
            d = d[d.month <= upto]
        if len(d) < 6:
            fits[tier] = None
            continue
        res, out = fit_line(d)
        last = d.month.max()
        d12 = d[d.month > last - pd.DateOffset(months=12)]
        slope12 = sm.OLS(np.log(d12.index_price), sm.add_constant(d12[["t"]])).fit().params["t"] if len(d12) >= 6 else np.nan
        out["last12_log_slope_per_year"] = float(slope12)
        out["last12_annual_factor"] = float(np.exp(slope12))
        b, (lo, hi) = res.params["t"], res.conf_int().loc["t"]
        fast = min(lo, slope12) if np.isfinite(slope12) else lo
        out["scenario_log_slopes"] = {"base": float(b), "fast": float(fast), "slow": float(hi)}
        out["scenario_annual_factors"] = {k: float(np.exp(v)) for k, v in out["scenario_log_slopes"].items()}
        tail = d.sort_values("month").tail(ANCHOR_MONTHS)
        out["anchor_month"] = f"{last:%Y-%m}"
        out["anchor_price"] = float(np.exp(np.log(tail.index_price).mean()))
        fits[tier] = {"res": res, **out}
    return fits


def project(fit: dict, months: int) -> pd.DataFrame:
    start = pd.Timestamp(fit["anchor_month"])
    rows = []
    for k in range(months + 1):
        m = start + pd.DateOffset(months=k)
        dt = k / 12
        row = {"month": m, "months_ahead": k}
        for sc, b in fit["scenario_log_slopes"].items():
            row[sc] = fit["anchor_price"] * np.exp(b * dt)
        rows.append(row)
    return pd.DataFrame(rows)


def months_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return (b.year - a.year) * 12 + b.month - a.month


def band_lags(idx: pd.DataFrame, upto: pd.Timestamp) -> dict:
    """v2: months from a band's first month until its price is within GAP x of the band below."""
    out = {}
    for lower, upper in zip(TIER_NAMES, TIER_NAMES[1:]):
        a = idx[(idx.tier == upper) & idx.used & (idx.month <= upto)].set_index("month").index_price
        if a.empty:
            continue
        b = idx[(idx.tier == lower) & (idx.month <= upto)].set_index("month").index_price.reindex(a.index)
        ratio = a / b
        hit = ratio[ratio <= GAP]
        start = a.index.min()
        out[upper] = {"start": f"{start:%Y-%m}", "start_price": float(a.iloc[0]), "start_gap": float(ratio.iloc[0]),
                      "converged": f"{hit.index.min():%Y-%m}" if len(hit) else None,
                      "lag_months": months_between(start, hit.index.min()) if len(hit) else None}
    return out


def lag_forecast(idx: pd.DataFrame, upto: pd.Timestamp, months: int, rate: float | None = None):
    """v2: every band in its floor phase moves at the lowest band's fitted rate (scenarios from its CI).
    A band that has never been within GAP x of the band below is in its drop phase: it holds its current
    price, then steps down to the band below's path at start + lag months (base = median observed lag,
    fast = shortest, slow = longest; an overdue band steps next month).
    rate: override the log slope per year for all scenarios (sensitivity)."""
    rfit = fit_tiers(idx, upto=upto)[TIER_NAMES[0]]
    slopes = dict(rfit["scenario_log_slopes"]) if rate is None else {k: rate for k in ("base", "fast", "slow")}
    lags = band_lags(idx, upto)
    obs = [v["lag_months"] for v in lags.values() if v["lag_months"] is not None]
    lag = ({"base": float(np.median(obs)), "fast": float(min(obs)), "slow": float(max(obs))} if obs
           else dict.fromkeys(("base", "fast", "slow"), float(FALLBACK_LAG)))
    ks = np.arange(months + 1)
    proj, info, prev, prev_anchor = {}, {}, None, None
    for tier in TIER_NAMES:
        d = idx[(idx.tier == tier) & idx.used & (idx.month <= upto)].sort_values("month")
        if d.empty or d.month.max() < upto:
            break  # nested bands: if this one isn't live yet, no higher one is
        anchor = float(d.index_price.iloc[-1])  # step model: prices sit flat between steps, so use the latest month
        gap = anchor / prev_anchor if prev is not None else 1.0
        phase = "drop" if tier in lags and lags[tier]["converged"] is None else "floor"
        paths, due = {}, {}
        for sc in ("base", "fast", "slow"):
            if phase == "floor":
                paths[sc] = anchor * np.exp(slopes[sc] * ks / 12)
            else:
                due[sc] = max(1.0, np.ceil(lag[sc] - months_between(d.month.min(), upto)))
                paths[sc] = np.where(ks < due[sc], anchor, prev[sc])
        proj[tier] = pd.DataFrame({"month": [upto + pd.DateOffset(months=int(k)) for k in ks], "months_ahead": ks, **paths})
        info[tier] = {"anchor_month": f"{upto:%Y-%m}", "anchor_price": anchor, "gap_to_band_below": gap, "phase": phase,
                      "months_to_converge": due or None, "first_month": f"{d.month.min():%Y-%m}"}
        prev, prev_anchor = paths, anchor
    meta = {"rate_from": TIER_NAMES[0], "rate_fit_window": f"{rfit['first_month']}..{rfit['last_month']}",
            "scenario_annual_factors": {k: float(np.exp(v)) for k, v in slopes.items()},
            "lag_months": lag, "lags_observed": lags}
    return proj, info, meta


def score(pr: pd.DataFrame, act: pd.DataFrame, anchor: float) -> dict | None:
    """Compare a projection with actual index prices; MALE = exp(mean |log error|) - 1."""
    j = pr.merge(act, on="month")
    j = j[j.months_ahead > 0]
    if j.empty:
        return None
    le = np.log(j.base / j.index_price)
    end = j.iloc[-1]
    return {"months_compared": int(len(j)),
            "mean_abs_log_error_pct": float((np.exp(le.abs().mean()) - 1) * 100),
            "bias_pct": float((np.exp(le.mean()) - 1) * 100),
            "endpoint_month": f"{end.month:%Y-%m}", "endpoint_projected": float(end.base), "endpoint_actual": float(end.index_price),
            "endpoint_error_pct": float((end.base / end.index_price - 1) * 100),
            "share_within_band": float(((j.index_price <= j.slow) & (j.index_price >= j.fast)).mean()),
            # naive benchmark: price stays at the anchor level
            "no_change_mean_abs_log_error_pct": float((np.exp(np.log(anchor / j.index_price).abs().mean()) - 1) * 100)}


def gap_tier_index(p: pd.DataFrame) -> pd.DataFrame:
    """v3: median blended price of models within lo..hi ECI points below the best priced model that month,
    released in the last GAP_TIER_MAX_AGE months."""
    q = p.copy()
    q["gap"] = q.groupby("month").eci.transform("max") - q.eci
    rel = q.epoch_release_date.fillna(q.groupby("canonical_id").month.transform("min"))
    q["age"] = (q.month.dt.year - rel.dt.year) * 12 + q.month.dt.month - rel.dt.month
    rows = []
    for name, lo, hi in GAP_TIERS:
        sel = q[(q.gap >= lo) & (q.gap < hi) & (q.age <= GAP_TIER_MAX_AGE)]
        g = sel.groupby("month").agg(index_price=("blended", "median"), index_input=("input", "median"),
                                     index_output=("output", "median"), n_models=("canonical_id", "nunique")).reset_index()
        rows.append(g.assign(tier=name))
    g = pd.concat(rows, ignore_index=True)
    g["used"] = g.n_models >= MIN_MODELS
    g["t"] = years(g.month)
    return g


def _chow_f(y: np.ndarray, t: np.ndarray, k: int) -> float:
    """F statistic for a break in level and slope after observation k."""
    ssr0 = sm.OLS(y, sm.add_constant(t)).fit().ssr
    ssr1 = sum(sm.OLS(z, sm.add_constant(tt)).fit().ssr for z, tt in ((y[:k], t[:k]), (y[k:], t[k:])))
    return (ssr0 - ssr1) / 2 / max(ssr1 / (len(y) - 4), 1e-12)


def _slope(y, t, hac=True):
    """Annual log slope with 95% CI (HAC, 2 lags); a series with no variation gets a zero-width CI."""
    r = sm.OLS(y, sm.add_constant(t)).fit()
    if r.ssr < 1e-12:
        return float(r.params[1]), (float(r.params[1]), float(r.params[1]))
    r = sm.OLS(y, sm.add_constant(t)).fit(cov_type="HAC", cov_kwds={"maxlags": 2}) if hac else r
    lo, hi = r.conf_int()[1]
    return float(r.params[1]), (float(lo), float(hi))


def floor_test(d: pd.DataFrame, rng) -> dict:
    """v3: has the series flattened? Sup-F (Quandt) test for one break in level and slope, searched over every
    month with >= BREAK_MIN_SIDE months on each side. p-value by parametric bootstrap under the no-break null
    (linear trend + AR(1) errors), since the series is short and autocorrelated."""
    d = d.sort_values("month")
    y, t = np.log(d.index_price.to_numpy()), d.t.to_numpy()
    b, ci = _slope(y, t)
    out = {"n_months": int(len(y)), "first_month": f"{d.month.min():%Y-%m}",
           "full_annual_factor": float(np.exp(b)), "full_ci95": [float(np.exp(ci[0])), float(np.exp(ci[1]))]}
    ks = list(range(BREAK_MIN_SIDE, len(y) - BREAK_MIN_SIDE + 1))
    if not ks:
        return {**out, "testable": False}
    F, k = max((_chow_f(y, t, k), k) for k in ks)
    r0 = sm.OLS(y, sm.add_constant(t)).fit()
    e = r0.resid
    rho = float(np.clip(np.corrcoef(e[:-1], e[1:])[0, 1], -0.95, 0.95)) if e.std() > 0 else 0.0
    sig = float(np.std(e[1:] - rho * e[:-1]))
    sims = np.empty(N_BOOT)
    for i in range(N_BOOT):
        u = np.empty(len(y))
        u[0] = rng.normal(0, sig / np.sqrt(1 - rho ** 2))
        for j in range(1, len(y)):
            u[j] = rho * u[j - 1] + rng.normal(0, sig)
        sims[i] = max(_chow_f(r0.fittedvalues + u, t, kk) for kk in ks)
    pre, _ = _slope(y[:k], t[:k], hac=False)
    post, pci = _slope(y[k:], t[k:])
    return {**out, "testable": True, "break_month": f"{d.month.iloc[k]:%Y-%m}", "sup_f": float(F),
            "p_value": float((np.sum(sims >= F) + 1) / (N_BOOT + 1)), "ar1_rho": rho,
            "pre_annual_factor": float(np.exp(pre)), "post_annual_factor": float(np.exp(post)),
            "post_ci95": [float(np.exp(pci[0])), float(np.exp(pci[1]))], "post_months": int(len(y) - k)}


def base_cases(p: pd.DataFrame, idx: pd.DataFrame, tests: dict, upto: pd.Timestamp, hed_factor: float) -> dict:
    """v3: base-case rate and starting price per tier, with the reason.

    Rule, in order:
    1. A band whose cheapest model also sets a lower band's price takes that band's base case (same model, same price).
    2. Significant break (p < BREAK_ALPHA) and post-break CI includes "no change": floor, held flat.
    3. Significant break, post-break CI excludes "no change": the post-break rate continues if it falls.
    4. No significant break: the full-history fitted rate if its CI excludes "no change" and it falls; flat if the CI
       includes "no change".
    A base case never projects a rise: a significantly rising trend is held flat and flagged (a judgement call).
    Starting price: the median of the last BASE_MEDIAN_MONTHS months: of the current cheapest model's own price for
    bands (so a new cheapest model is not averaged with the one it replaced), of the tier index for mid and frontier.
    """
    bands = [t for t in TIER_NAMES if t.startswith("eci>=")]
    out = {}
    for tier in bands + [n for n, _, _ in GAP_TIERS]:
        d = idx[(idx.tier == tier) & idx.used & (idx.month <= upto)].sort_values("month")
        if d.empty or d.month.max() < upto:
            continue
        tst = tests.get(tier) or {}
        res = {"tier": tier}
        if tier in bands:
            model = d.cheapest_model.iloc[-1]
            mp = p[(p.canonical_id == model) & (p.month <= upto)].sort_values("month").tail(BASE_MEDIAN_MONTHS)
            res.update(cheapest_model=model, base_price=float(mp.blended.median()))
            shared = next((b for b in bands[:bands.index(tier)] if b in out and out[b].get("cheapest_model") == model), None)
        else:
            res.update(base_price=float(d.index_price.tail(BASE_MEDIAN_MONTHS).median()), n_models=int(d.n_models.iloc[-1]))
            shared = None
        res["latest_price"] = float(d.index_price.iloc[-1])
        if shared:
            res.update(rule="same model", annual_factor=out[shared]["annual_factor"], floor=out[shared]["floor"],
                       reason=f"priced by the same model as {shared}; takes its base case")
        elif tst.get("testable") and tst["p_value"] < BREAK_ALPHA:
            lo, hi = tst["post_ci95"]
            if lo <= 1 + 1e-9 and hi >= 1 - 1e-9:
                res.update(rule="floor", annual_factor=1.0, floor=True,
                           reason=f"break in {tst['break_month']} (p={tst['p_value']:.3f}); no significant change since")
            elif tst["post_annual_factor"] > 1:
                res.update(rule="rising trend held flat", annual_factor=1.0, floor=False,
                           reason=f"break in {tst['break_month']} (p={tst['p_value']:.3f}); rising since (x{tst['post_annual_factor']:.2f}/yr), held flat as a judgement call")
            else:
                res.update(rule="post-break rate", annual_factor=tst["post_annual_factor"], floor=False,
                           reason=f"break in {tst['break_month']} (p={tst['p_value']:.3f}); still changing at x{tst['post_annual_factor']:.2f}/yr since")
        elif tst:
            lo, hi = tst["full_ci95"]
            lo, hi = lo - 1e-9, hi + 1e-9
            why = "no significant break" if tst.get("testable") else "too short to test for a break"
            if lo <= 1 <= hi:
                res.update(rule="no trend", annual_factor=1.0, floor=False, reason=f"{why}; full-history trend not significant")
            elif tst["full_annual_factor"] < 1:
                res.update(rule="full-history rate", annual_factor=tst["full_annual_factor"], floor=False, reason=f"{why}; full-history fit")
            else:
                res.update(rule="rising trend held flat", annual_factor=1.0, floor=False,
                           reason=f"{why}; full-history trend rises (x{tst['full_annual_factor']:.2f}/yr), held flat as a judgement call")
        else:
            res.update(rule="no data", annual_factor=1.0, floor=False, reason="too short to fit")
        # scenarios: fast = constant-capability (hedonic) decline unless the base is already steeper; slow = no change
        res["scenario_annual_factors"] = {"base": res["annual_factor"], "fast": min(res["annual_factor"], hed_factor), "slow": 1.0}
        res["anchor_month"] = f"{upto:%Y-%m}"
        out[tier] = res
    return out


def project_base(b: dict, start: pd.Timestamp, months: int) -> pd.DataFrame:
    ks = np.arange(months + 1)
    return pd.DataFrame({"month": [start + pd.DateOffset(months=int(k)) for k in ks], "months_ahead": ks,
                         **{sc: b["base_price"] * f ** (ks / 12) for sc, f in b["scenario_annual_factors"].items()}})


def rolling_rates(idx: pd.DataFrame, windows=(6, 12)) -> pd.DataFrame:
    """Annual price factor from a log-linear fit over each trailing window (only full windows)."""
    rows = []
    for tier in TIER_NAMES:
        d = idx[(idx.tier == tier) & idx.used].sort_values("month").reset_index(drop=True)
        for w in windows:
            for i in range(w - 1, len(d)):
                win = d.iloc[i - w + 1:i + 1]
                if months_between(win.month.iloc[0], win.month.iloc[-1]) != w - 1:
                    continue  # gap in the series
                b = sm.OLS(np.log(win.index_price.to_numpy()), sm.add_constant(win.t.to_numpy())).fit().params[1]
                rows.append({"month": d.month.iloc[i], "tier": tier, "window_months": w, "annual_factor": float(np.exp(b))})
    return pd.DataFrame(rows)


def chart_rolling(rr: pd.DataFrame, tests: dict, path):
    tiers = [t for t in TIER_NAMES[::-1] if t in set(rr.tier)]
    fig, axes = plt.subplots(1, len(tiers), figsize=(3.1 * len(tiers), 3.9), dpi=150, sharey=True)
    for ax, tier in zip(np.atleast_1d(axes), tiers):
        c = COLORS[tier]
        for w, ls in ((12, "-"), (6, ":")):
            d = rr[(rr.tier == tier) & (rr.window_months == w)]
            ax.plot(d.month, d.annual_factor.clip(1e-3, 1e3), color=c, lw=1.8 if w == 12 else 1.2, ls=ls, label=f"{w}-month window")
        ax.axhline(1, color=INK2, lw=0.8)
        tst = tests.get(tier, {})
        if tst.get("testable"):
            ax.axvline(pd.Timestamp(tst["break_month"]), color=INK2, lw=0.8, ls="--")
        sub = f"break {tst['break_month']}, p={tst['p_value']:.3f}" if tst.get("testable") else "too short to test"
        style(ax, f"{tier}\n{sub}", ylabel="Annual price factor (log; 1 = flat)" if ax is np.atleast_1d(axes)[0] else "")
        ax.title.set_fontsize(10)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"×{v:g}"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left")
    fig.suptitle("Rolling decline rate by tier (below 1 = prices falling; dashed line = most likely break month)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ---------------------------------------------------------------- diagnostics


def outliers(p: pd.DataFrame, res) -> pd.DataFrame:
    infl = res.get_influence()
    r = p.assign(resid=res.resid, student=infl.resid_studentized_internal)
    o = r[r.student.abs() > 3]
    return (o.groupby("canonical_id").agg(months=("month", "size"), first=("month", "min"), last=("month", "max"),
                                          blended=("blended", "median"), eci=("eci", "first"),
                                          resid_log=("resid", "mean"))
            .assign(price_vs_fit=lambda d: np.exp(d.resid_log)).sort_values("resid_log", key=abs, ascending=False))


def repricing(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_values(["canonical_id", "month"]).copy()
    g = p.groupby("canonical_id")
    p["prev_blended"], p["prev_month"], p["prev_source"] = g.blended.shift(), g.month.shift(), g.price_source.shift()
    p = p[(p.prev_month == p.month - pd.DateOffset(months=1)) & (p.prev_source == p.price_source)
          & p.price_source.isin(["litellm_first_party", "llm_prices"])]  # medians over hosting keys shift with the key set
    p["change"] = p.blended / p.prev_blended - 1
    ev = p[p.change.abs() >= REPRICE_THRESHOLD]
    return ev[["month", "canonical_id", "prev_blended", "blended", "change", "price_source", "eci"]].sort_values("month")


# ---------------------------------------------------------------- charts


def rate_label(x: float) -> str:
    """x = 'falls N x per year'; show rising prices as a multiplier instead of ÷0.7."""
    return f"÷{x:.1f}/yr" if x >= 1 else f"×{1 / x:.1f}/yr (rising)"


def usd(v: float) -> str:
    return f"${v:.3g}" if v >= 0.001 else f"${v:.1e}"


def end_labels(ax, items, gap=0.05):
    """items: (x, y, text). Nudge labels apart vertically (log axis, gap = share of the axis height)."""
    lo, hi = ax.get_ylim()
    min_ratio = (hi / lo) ** gap
    items = sorted(items, key=lambda i: i[1])
    ys = []
    for _, y, _ in items:
        ys.append(max(y, ys[-1] * min_ratio) if ys else y)
    for (x, y, text), ly in zip(items, ys):
        ax.annotate(text, (x, y), xytext=(x, ly), textcoords="data", color=INK, fontsize=9, va="center",
                    ha="left", annotation_clip=False)
    ax.margins(x=0.02)


def legend_below(ax):
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)


def style(ax, title, ylabel="Blended price, USD per 1M tokens (log)"):
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))


def chart_history(p, idx, fits, path):
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150)
    labels = []
    if VERSION != "v0":
        ax.scatter(p.month, p.blended, s=8, color=INK2, alpha=0.12, lw=0, zorder=1, label="individual models")
    for tier in TIER_NAMES:
        c = COLORS[tier]
        if VERSION == "v0":
            pts = p[p.tier == tier]
            ax.scatter(pts.month, pts.blended, s=8, color=c, alpha=0.18, lw=0, zorder=1)
        d = idx[(idx.tier == tier) & idx.used]
        lab = INDEX_LABEL if tier.startswith("eci>=") or VERSION == "v0" else "median of recent models"
        ax.plot(d.month, d.index_price, color=c, lw=2, marker="o", ms=4, zorder=3, label=f"{tier} — {lab}")
        f = fits.get(tier)
        if VERSION in ("v2", "v3") and len(d):
            labels.append((d.month.iloc[-1] + pd.Timedelta(days=12), d.index_price.iloc[-1], f"{tier}: {usd(d.index_price.iloc[-1])}"))
        elif f:
            ax.plot(d.month, np.exp(f["res"].fittedvalues), color=c, lw=1.2, ls="--", zorder=2)
            labels.append((d.month.iloc[-1] + pd.Timedelta(days=12), d.index_price.iloc[-1], f"{tier}: {rate_label(f['falls_x_per_year'])}"))
    style(ax, "Token price by capability tier (dots: individual models; dashed: log-linear fit)" if VERSION == "v0"
          else "Cheapest model at or above each ECI threshold (dashed: log-linear fit)" if VERSION == "v1"
          else "Cheapest model at or above each ECI threshold" if VERSION == "v2"
          else "Frontier and mid tiers (median price) and cheapest model at each ECI threshold")
    end_labels(ax, labels)
    legend_below(ax)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def chart_hedonic(p, res, out, path):
    eci_ref = float(p.eci.median())
    adj = p.blended * np.exp(-out["eci_coef"] * (p.eci - eci_ref))
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
    ax.scatter(p.month, adj, s=9, color=PALETTE[0], alpha=0.25, lw=0)
    tt = np.linspace(p.t.min(), p.t.max(), 50)
    line = np.exp(res.params["const"] + res.params["t"] * tt + res.params["eci"] * eci_ref)
    ax.plot(T0 + pd.to_timedelta(tt * 365.25, unit="D"), line, color=INK, lw=2)
    ax.annotate(f"÷{out['falls_x_per_year']:.1f} per year at constant ECI "
                f"(95% CI ÷{out['falls_x_per_year_ci95'][0]:.1f}–÷{out['falls_x_per_year_ci95'][1]:.1f}), R²={out['r_squared']:.2f}",
                (0.01, 0.03), xycoords="axes fraction", fontsize=9, color=INK)
    style(ax, f"Quality-adjusted price: each model-month adjusted to ECI {eci_ref:.0f}")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def chart_forward(idx, fits, curve, path):
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150)
    labels = []
    for tier in TIER_NAMES:
        f = fits.get(tier)
        if not f:
            continue
        c = COLORS[tier]
        d = idx[(idx.tier == tier) & idx.used]
        ax.plot(d.month, d.index_price, color=c, lw=1.6, marker="o", ms=3, label=f"{tier} — history")
        cv = curve[curve.tier == tier].pivot(index="month", columns="scenario", values="blended_price_per_mtok")
        if VERSION == "v2":
            ax.fill_between(cv.index, cv.fast_decline, cv.base, color=c, alpha=0.12, lw=0)
            ax.plot(cv.index, cv.base, color=c, lw=2)
            ax.plot(cv.index, cv.decline, color=c, lw=1.2, ls="--")
        else:
            ax.fill_between(cv.index, cv.slow, cv.fast, color=c, alpha=0.15, lw=0)
            ax.plot(cv.index, cv.base, color=c, lw=2, ls="--" if VERSION != "v3" else "-")
        labels.append((cv.index[-1] + pd.Timedelta(days=12), cv.base.iloc[-1], f"{tier}: {usd(cv.base.iloc[-1])}"))
    ax.axvline(pd.Timestamp(next(f for f in fits.values() if f)["anchor_month"]), color=INK2, lw=0.8, ls=":")
    style(ax, {"v2": "Forward curve v2: base = no change (solid); decline resumes (dashed); shaded down to fast decline",
               "v3": "Forward curve v3: base case per tier (solid); shaded from no change down to the hedonic decline"}.get(
        VERSION, f"Forward curve {VERSION}: base (dashed) with slow–fast scenario band, 36 months"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: usd(v)))
    end_labels(ax, labels)
    legend_below(ax)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def chart_backtest(idx, bt, path):
    tiers = [t for t in bt]
    fig, axes = plt.subplots(1, len(tiers), figsize=(4.2 * len(tiers), 4.4), dpi=150, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, tier in zip(axes, tiers):
        c = COLORS[tier]
        d = idx[(idx.tier == tier) & idx.used]
        ax.plot(d.month, d.index_price, color=c, lw=1.6, marker="o", ms=3, label="actual")
        pr = bt[tier]["projection"]
        ax.fill_between(pr.month, pr.slow, pr.fast, color=c, alpha=0.15, lw=0)
        ax.plot(pr.month, pr.base, color=INK, lw=1.4, ls="--", label="projected (base)")
        ax.axvline(pr.month.iloc[0], color=INK2, lw=0.8, ls=":")
        style(ax, f"{tier}: MALE {bt[tier]['mean_abs_log_error_pct']:.0f}% (no change {bt[tier]['no_change_mean_abs_log_error_pct']:.0f}%)", ylabel="" if ax is not axes[0] else None)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left")
    fig.suptitle("Backtest: fit to data up to 12 months ago, project forward (band = slow–fast; MALE = mean abs. log error)", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ---------------------------------------------------------------- main


def main() -> int:
    global VERSION, TIER_NAMES, COLORS, INDEX_LABEL, ECI_THRESHOLDS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", choices=["v0", "v1", "v2", "v3"], default="v3")
    VERSION = ap.parse_args().version
    if VERSION in ("v2", "v3"):
        ECI_THRESHOLDS = V2_THRESHOLDS
    TIER_NAMES = [n for n, _, _ in TIERS] if VERSION == "v0" else [f"eci>={t}" for t in ECI_THRESHOLDS]
    COLORS = dict(zip(TIER_NAMES, PALETTE))
    if VERSION == "v3":
        TIER_NAMES += [n for n, _, _ in GAP_TIERS]
        COLORS.update(GAP_COLORS)
    INDEX_LABEL = "monthly median" if VERSION == "v0" else "cheapest model"
    make_index = (tier_index if VERSION == "v0" else band_index if VERSION != "v3"
                  else lambda q: pd.concat([band_index(q), gap_tier_index(q)], ignore_index=True))
    OUT.mkdir(parents=True, exist_ok=True)
    REP.mkdir(exist_ok=True)
    panel = build_panel(ref_provider=VERSION == "v3")
    p = assign_tiers(panel)
    last = p.month.max()

    # outliers: |studentized residual| > 3 in a first hedonic pass; excluded from every fit below
    first_res, _ = fit_hedonic(p)
    outl = outliers(p, first_res)
    stud = first_res.get_influence().resid_studentized_internal
    p["outlier"] = np.abs(np.asarray(stud)) > 3
    p_all, p = p, p[~p.outlier].copy()

    hed_res, hed = fit_hedonic(p)
    hed["n_outlier_obs_excluded"] = int(p_all.outlier.sum())
    idx = make_index(p)
    fits = fit_tiers(idx)

    actual = lambda tier: idx[(idx.tier == tier) & idx.used][["month", "index_price"]]
    cutoff = last - pd.DateOffset(months=BACKTEST_MONTHS)
    bt, v2 = {}, {}
    rows = []
    if VERSION in ("v0", "v1"):
        # forward curve: each tier's own fitted rate
        for tier, f in fits.items():
            if not f:
                continue
            pr = project(f, HORIZON)
            for sc in ("base", "fast", "slow"):
                rows.append(pr.assign(tier=tier, scenario=sc, blended_price_per_mtok=pr[sc], annual_factor=f["scenario_annual_factors"][sc])
                            [["month", "months_ahead", "tier", "scenario", "blended_price_per_mtok", "annual_factor"]])
        fwd_fits = fits
        # backtest
        for tier, f in fit_tiers(idx, upto=cutoff).items():
            if not f:
                continue
            pr = project(f, BACKTEST_MONTHS)
            sc_ = score(pr, actual(tier), f["anchor_price"])
            if sc_:
                bt[tier] = {"projection": pr, "fit_window": f"{f['first_month']}..{f['last_month']}",
                            "train_annual_factor": f["annual_factor"], **sc_}
    elif VERSION == "v2":
        # forward curve. Published base case = no change (it beat every fitted forecast in the backtests).
        # The rate-plus-lag model supplies the "decline resumes" scenarios.
        proj, info, fmeta = lag_forecast(idx, last, HORIZON)
        f_ = fmeta["scenario_annual_factors"]
        for tier, pr in proj.items():
            for sc, col, af in (("base", None, 1.0), ("decline", "base", f_["base"]), ("fast_decline", "fast", f_["fast"])):
                price = info[tier]["anchor_price"] if col is None else pr[col]
                rows.append(pr.assign(tier=tier, scenario=sc, blended_price_per_mtok=price, annual_factor=af)
                            [["month", "months_ahead", "tier", "scenario", "blended_price_per_mtok", "annual_factor"]])
        fwd_fits = info
        # 12-month backtest
        bproj, binfo, bmeta = lag_forecast(idx, cutoff, BACKTEST_MONTHS)
        flat_proj, _, flat_meta = lag_forecast(idx, cutoff, BACKTEST_MONTHS, rate=0.0)  # lag only, floors flat
        for tier, pr in bproj.items():
            sc_ = score(pr, actual(tier), binfo[tier]["anchor_price"])
            if sc_:
                lag_only = score(flat_proj[tier], actual(tier), binfo[tier]["anchor_price"])
                bt[tier] = {"projection": pr, **binfo[tier], **sc_,
                            "lag_only_mean_abs_log_error_pct": lag_only["mean_abs_log_error_pct"]}
        # rolling-origin backtest: every cutoff with ROLL_HORIZON months of actuals after it
        roll = []
        origins = pd.date_range(idx[idx.used].month.min() + pd.DateOffset(months=6), last - pd.DateOffset(months=ROLL_HORIZON), freq="MS")
        for c in origins:
            for variant, rate in (("v2", None), ("lag_only", 0.0)):
                rp, ri, _ = lag_forecast(idx, c, ROLL_HORIZON, rate=rate)
                for tier, pr in rp.items():
                    j = pr.merge(actual(tier), on="month")
                    j = j[j.months_ahead > 0]
                    roll.append(pd.DataFrame({"origin": c, "tier": tier, "variant": variant, "phase": ri[tier]["phase"],
                                              "h": j.months_ahead, "log_err": np.log(j.base / j.index_price),
                                              "log_err_nochange": np.log(ri[tier]["anchor_price"] / j.index_price)}))
        roll = pd.concat(roll, ignore_index=True)
        male = lambda x: float((np.exp(x.abs().mean()) - 1) * 100)
        def summarise(g):
            return {"origins": int(g.origin.nunique()), "n": int(len(g)), "mean_abs_log_error_pct": male(g.log_err),
                    "no_change_mean_abs_log_error_pct": male(g.log_err_nochange), "bias_pct": float((np.exp(g.log_err.mean()) - 1) * 100)}
        v2 = {"forecast_meta": fmeta, "backtest_meta": bmeta, "backtest_lag_only_meta": flat_meta,
              "rolling": {"horizon_months": ROLL_HORIZON, "origins": f"{origins.min():%Y-%m}..{origins.max():%Y-%m}",
                          **{var: {"all": summarise(g), "by_tier": {t: summarise(h) for t, h in g.groupby("tier")},
                                   "by_phase": {ph: summarise(h) for ph, h in g.groupby("phase")}}
                             for var, g in roll.groupby("variant")}}}
        roll.to_csv(REP / f"curve_{VERSION}_rolling_backtest.csv", index=False)
    else:
        # v3: base case per tier from the floor (structural-break) test
        rng = np.random.default_rng(0)
        tests = {t: floor_test(idx[(idx.tier == t) & idx.used], rng) for t in TIER_NAMES
                 if len(idx[(idx.tier == t) & idx.used]) >= BREAK_MIN_SIDE}
        bases = base_cases(p, idx, tests, last, hed["annual_factor"])
        for tier, b in bases.items():
            pr = project_base(b, last, HORIZON)
            for sc in ("base", "fast", "slow"):
                rows.append(pr.assign(tier=tier, scenario=sc, blended_price_per_mtok=pr[sc], annual_factor=b["scenario_annual_factors"][sc])
                            [["month", "months_ahead", "tier", "scenario", "blended_price_per_mtok", "annual_factor"]])
        fwd_fits = bases
        # backtests: the same rule, run on data up to 12 and 6 months ago
        bt_runs = {}
        for horizon in (BACKTEST_MONTHS, 6):
            c = last - pd.DateOffset(months=horizon)
            idx_bt, p_bt = idx[idx.month <= c], p[p.month <= c]
            tests_bt = {t: floor_test(idx_bt[(idx_bt.tier == t) & idx_bt.used], rng) for t in TIER_NAMES
                        if len(idx_bt[(idx_bt.tier == t) & idx_bt.used]) >= BREAK_MIN_SIDE}
            bases_bt = base_cases(p_bt, idx_bt, tests_bt, c, fit_hedonic(p_bt)[1]["annual_factor"])
            run = {}
            for tier, b in bases_bt.items():
                pr = project_base(b, c, horizon)
                sc_ = score(pr, actual(tier), b["base_price"])
                if sc_:
                    latest = score(pr.assign(base=b["latest_price"]), actual(tier), b["latest_price"])
                    run[tier] = {"projection": pr, **{k: v for k, v in b.items() if k != "scenario_annual_factors"}, **sc_,
                                 "latest_month_no_change_mean_abs_log_error_pct": latest["mean_abs_log_error_pct"]}
            bt_runs[horizon] = {"cutoff": f"{c:%Y-%m}", "floor_tests": tests_bt, "tiers": run}
        bt = bt_runs[BACKTEST_MONTHS]["tiers"]
        tests_bt, bases_bt = bt_runs[BACKTEST_MONTHS]["floor_tests"], None
        rr = rolling_rates(idx)
        rr.assign(month=rr.month.dt.strftime("%Y-%m")).to_csv(REP / f"curve_{VERSION}_rolling_rates.csv", index=False)
        latest_rr = rr.sort_values("month").groupby(["tier", "window_months"]).annual_factor.last().unstack()
        v2 = {"floor_tests": tests, "base_cases": bases,
              "backtests": {f"{h}m": {"cutoff": r["cutoff"], "floor_tests": r["floor_tests"],
                                      "tiers": {t: {k: v for k, v in x.items() if k != "projection"} for t, x in r["tiers"].items()}}
                            for h, r in bt_runs.items()},
              "rolling_latest": {t: {f"{w}m": float(r[w]) for w in latest_rr.columns if pd.notna(r[w])} for t, r in latest_rr.iterrows()},
              "hedonic_annual_factor": hed["annual_factor"]}
    curve = pd.concat(rows, ignore_index=True)
    curve = curve[curve.months_ahead > 0]

    # sensitivities (not used for the curve)
    recent = p[p.age_months <= RECENT_MONTHS]
    _, hed_recent = fit_hedonic(recent)
    launch = p[p.age_months >= -1].sort_values("month").groupby("canonical_id").head(1)  # first priced month
    _, hed_launch = fit_hedonic(launch)
    idx_recent = make_index(recent)
    fits_recent = fit_tiers(idx_recent)
    sens = {"hedonic_recent_only": hed_recent, "hedonic_launch_price": hed_launch,
            "tiers_recent_only": {t: {k: v for k, v in f.items() if k != "res"} if f else None for t, f in fits_recent.items()},
            "tiers_recent_only_months_used": idx_recent[idx_recent.used].groupby("tier").month.size().to_dict(),
            "tier_median_age_months_last": p[p.month == last].groupby("tier").age_months.median().round(1).to_dict()}

    rep = repricing(panel)

    # ---- write data
    panel.drop(columns="t").to_parquet(OUT / f"panel_{VERSION}.parquet", index=False)
    idx.drop(columns="t").to_csv(OUT / f"tier_history_{VERSION}.csv", index=False)
    curve.assign(month=curve.month.dt.strftime("%Y-%m-%d")).to_csv(OUT / f"curve_{VERSION}.csv", index=False)
    outl.to_csv(REP / f"curve_{VERSION}_outliers.csv")
    rep.to_csv(REP / f"curve_{VERSION}_repricing_events.csv", index=False)

    strip = lambda f: {k: v for k, v in f.items() if k != "res"}
    meta = {
        "version": VERSION, "index": INDEX_LABEL, "generated_from_month": f"{last:%Y-%m}", "units": "USD per 1M tokens, blended 3:1 input:output",
        "tiers": ({n: f"ECI >= {lo:.0%} of best model that month" if n == "frontier" else f"{lo:.0%}-{hi:.0%} of best" if lo else f"< {hi:.0%} of best"
                   for n, lo, hi in TIERS} if VERSION == "v0" else
                  {**{f"eci>={t}": f"cheapest model with Epoch ECI >= {t}" for t in ECI_THRESHOLDS},
                   **({n: f"median price of models {lo}-{hi} ECI points below the best priced model that month, released in the last {GAP_TIER_MAX_AGE} months"
                       for n, lo, hi in GAP_TIERS} if VERSION == "v3" else {})}),
        "scenarios": ({"base": "fitted log-linear rate", "fast": "steeper of 95% CI bound and last-12-month rate",
                       "slow": "shallower 95% CI bound"} if VERSION in ("v0", "v1") else
                      {"base": "per tier: flat where a floor is supported by the break test, else the fitted rate (see base_case.reason)",
                       "fast": f"constant-capability (hedonic) decline, x{hed['annual_factor']:.2f}/yr, unless the base is steeper",
                       "slow": "no change from the base price"} if VERSION == "v3" else
                      {"base": "no change: latest monthly price held flat (beat every fitted forecast in the backtests)",
                       "decline": f"decline resumes: {TIER_NAMES[0]} fitted rate for every band; bands not yet near the band below step down after the median observed catch-up lag",
                       "fast_decline": "steeper of the rate's 95% CI bound and its last-12-month rate; shortest observed catch-up lag"}),
        "anchor": ("latest monthly index value" if VERSION == "v2" else
                   f"median of the last {BASE_MEDIAN_MONTHS} months: the current cheapest model's own price for bands, the tier index for mid and frontier"
                   if VERSION == "v3" else f"geometric mean of last {ANCHOR_MONTHS} monthly index values"),
    }
    js = {"meta": meta, "tiers": {}}
    for tier, f in fwd_fits.items():
        if not f:
            continue
        cv = curve[curve.tier == tier].pivot(index="month", columns="scenario", values="blended_price_per_mtok")
        hist = idx[(idx.tier == tier) & idx.used]
        js["tiers"][tier] = {
            {"v2": "forecast_state", "v3": "base_case"}.get(VERSION, "fit"): {k: v for k, v in strip(f).items() if k not in ("scenario_log_slopes",)},
            "history": [{"month": f"{r.month:%Y-%m}", "index_price": round(r.index_price, 6), "n_models": int(r.n_models)} for r in hist.itertuples()],
            "forecast": [{"month": f"{m:%Y-%m}", **{k: round(float(r[k]), 6) for k in cv.columns}} for m, r in cv.iterrows()],
        }
    (OUT / f"curve_{VERSION}.json").write_text(json.dumps(js, indent=1))

    results = {"hedonic": hed, "tiers": {t: strip(f) for t, f in fits.items() if f},
               "backtest": {t: {k: v for k, v in b.items() if k != "projection"} for t, b in bt.items()},
               "backtest_cutoff": f"{cutoff:%Y-%m}", **({VERSION: v2} if v2 else {}),
               "panel": {"model_months": int(len(panel)), "models": int(panel.canonical_id.nunique()),
                         "model_months_with_eci": int(len(p)), "models_with_eci": int(p.canonical_id.nunique()),
                         "first_month": f"{panel.month.min():%Y-%m}", "last_month": f"{last:%Y-%m}",
                         "price_source_share": p.price_source.value_counts(normalize=True).round(3).to_dict()},
               "tier_months_used": idx[idx.used].groupby("tier").month.agg(["min", "max", "count"]).astype(str).to_dict("index"),
               "n_outlier_models": int(len(outl)), "n_repricing_events": int(len(rep)), "sensitivity": sens}
    (REP / f"curve_{VERSION}_results.json").write_text(json.dumps(results, indent=1, default=str))

    # ---- charts
    chart_history(p, idx, fits, REP / f"curve_{VERSION}_tier_history.png")
    chart_hedonic(p, hed_res, hed, REP / f"curve_{VERSION}_hedonic.png")
    chart_forward(idx, fwd_fits, curve, REP / f"curve_{VERSION}_forward.png")
    chart_backtest(idx, bt, REP / f"curve_{VERSION}_backtest.png")
    if VERSION == "v3":
        chart_rolling(rr, tests, REP / f"curve_{VERSION}_rolling_rates.png")

    print(json.dumps({k: results[k] for k in ("panel", "tier_months_used", "n_outlier_models", "n_repricing_events")}, indent=1, default=str))
    print("hedonic:", {k: hed[k] for k in ("falls_x_per_year", "falls_x_per_year_ci95", "r_squared", "eci_coef", "n_obs")})
    for t, f in fits.items():
        if f:
            print(t, {k: f[k] for k in ("falls_x_per_year", "falls_x_per_year_ci95", "r_squared", "n_months", "last12_annual_factor", "scenario_annual_factors", "anchor_price")})
    for t, b in results["backtest"].items():
        print("bt", t, b)
    if v2 and VERSION == "v2":
        print(json.dumps({k: v2[k] for k in ("forecast_meta", "rolling")}, indent=1, default=str))
    print("sens hedonic recent:", {k: hed_recent[k] for k in ("falls_x_per_year", "falls_x_per_year_ci95", "r_squared", "n_obs")})
    print("sens hedonic launch:", {k: hed_launch[k] for k in ("falls_x_per_year", "falls_x_per_year_ci95", "r_squared", "n_obs")})
    for t, f in fits_recent.items():
        print("sens tier recent", t, f and {k: f[k] for k in ("falls_x_per_year", "falls_x_per_year_ci95", "r_squared", "n_months")})
    print("age", sens["tier_median_age_months_last"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
