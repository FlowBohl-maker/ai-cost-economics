"""Token price curve: hedonic index, ECI tier indices, 36-month forward curve, 12-month backtest.

  --version v0  tiers relative to the best model each month, index = tier median price
  --version v1  absolute ECI thresholds, index = cheapest model at or above the threshold
  --version v2  v1 bands, forecast = one floor rate for every band plus a catch-up lag for newly reached bands

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
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # dataviz reference categorical slots 1-5, fixed order

# set in main() from --version
VERSION, TIER_NAMES, COLORS, INDEX_LABEL = "v1", [], {}, ""
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def years(d) -> np.ndarray:
    return ((pd.to_datetime(pd.Series(d)) - T0).dt.days / 365.25).to_numpy()


# ---------------------------------------------------------------- panel


def build_panel() -> pd.DataFrame:
    """One blended price per canonical model per month.

    Source priority: model maker's own LiteLLM rows (median over keys) > llm-prices > median of all
    LiteLLM providers > OpenRouter (monthly mean of daily snapshots).
    """
    h = pd.read_parquet(DATA / "price_history.parquet")
    h = h[h.canonical_id.notna()].copy()
    vendor = h.canonical_id.str.split("/").str[0]
    h["first_party"] = (h.source == "litellm").to_numpy() & np.array(
        [p in FIRST_PARTY.get(v, set()) for p, v in zip(h.provider.fillna(""), vendor)], dtype=bool)
    h["tier_src"] = np.select(
        [h.first_party, h.source == "llm_prices", h.source == "litellm"], ["litellm_first_party", "llm_prices", "litellm_other"], "")

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
    rank = {"litellm_first_party": 0, "llm_prices": 1, "litellm_other": 2, "openrouter": 3}
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
        ax.plot(d.month, d.index_price, color=c, lw=2, marker="o", ms=4, zorder=3, label=f"{tier} — {INDEX_LABEL}")
        f = fits.get(tier)
        if VERSION == "v2" and len(d):
            labels.append((d.month.iloc[-1] + pd.Timedelta(days=12), d.index_price.iloc[-1], f"{tier}: {usd(d.index_price.iloc[-1])}"))
        elif f:
            ax.plot(d.month, np.exp(f["res"].fittedvalues), color=c, lw=1.2, ls="--", zorder=2)
            labels.append((d.month.iloc[-1] + pd.Timedelta(days=12), d.index_price.iloc[-1], f"{tier}: {rate_label(f['falls_x_per_year'])}"))
    style(ax, "Token price by capability tier (dots: individual models; dashed: log-linear fit)" if VERSION == "v0"
          else "Cheapest model at or above each ECI threshold (dashed: log-linear fit)" if VERSION == "v1"
          else "Cheapest model at or above each ECI threshold")
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
            ax.plot(cv.index, cv.base, color=c, lw=2, ls="--")
        labels.append((cv.index[-1] + pd.Timedelta(days=12), cv.base.iloc[-1], f"{tier}: {usd(cv.base.iloc[-1])}"))
    ax.axvline(pd.Timestamp(next(f for f in fits.values() if f)["anchor_month"]), color=INK2, lw=0.8, ls=":")
    style(ax, f"Forward curve {VERSION}: base (dashed) with slow–fast scenario band, 36 months" if VERSION != "v2" else
          "Forward curve v2: base = no change (solid); decline resumes (dashed); shaded down to fast decline")
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
    ap.add_argument("--version", choices=["v0", "v1", "v2"], default="v2")
    VERSION = ap.parse_args().version
    if VERSION == "v2":
        ECI_THRESHOLDS = V2_THRESHOLDS
    TIER_NAMES = [n for n, _, _ in TIERS] if VERSION == "v0" else [f"eci>={t}" for t in ECI_THRESHOLDS]
    COLORS = dict(zip(TIER_NAMES, PALETTE))
    INDEX_LABEL = "monthly median" if VERSION == "v0" else "cheapest model"
    make_index = tier_index if VERSION == "v0" else band_index
    OUT.mkdir(parents=True, exist_ok=True)
    REP.mkdir(exist_ok=True)
    panel = build_panel()
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
    if VERSION != "v2":
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
    else:
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
                  {f"eci>={t}": f"cheapest model with Epoch ECI >= {t}" for t in ECI_THRESHOLDS}),
        "scenarios": ({"base": "fitted log-linear rate", "fast": "steeper of 95% CI bound and last-12-month rate",
                       "slow": "shallower 95% CI bound"} if VERSION != "v2" else
                      {"base": "no change: latest monthly price held flat (beat every fitted forecast in the backtests)",
                       "decline": f"decline resumes: {TIER_NAMES[0]} fitted rate for every band; bands not yet near the band below step down after the median observed catch-up lag",
                       "fast_decline": "steeper of the rate's 95% CI bound and its last-12-month rate; shortest observed catch-up lag"}),
        "anchor": f"geometric mean of last {ANCHOR_MONTHS} monthly index values" if VERSION != "v2" else "latest monthly index value",
    }
    js = {"meta": meta, "tiers": {}}
    for tier, f in fwd_fits.items():
        if not f:
            continue
        cv = curve[curve.tier == tier].pivot(index="month", columns="scenario", values="blended_price_per_mtok")
        hist = idx[(idx.tier == tier) & idx.used]
        js["tiers"][tier] = {
            "fit" if VERSION != "v2" else "forecast_state": {k: v for k, v in strip(f).items() if k not in ("scenario_log_slopes",)},
            "history": [{"month": f"{r.month:%Y-%m}", "index_price": round(r.index_price, 6), "n_models": int(r.n_models)} for r in hist.itertuples()],
            "forecast": [{"month": f"{m:%Y-%m}", **{k: round(float(r[k]), 6) for k in cv.columns}} for m, r in cv.iterrows()],
        }
    (OUT / f"curve_{VERSION}.json").write_text(json.dumps(js, indent=1))

    results = {"hedonic": hed, "tiers": {t: strip(f) for t, f in fits.items() if f},
               "backtest": {t: {k: v for k, v in b.items() if k != "projection"} for t, b in bt.items()},
               "backtest_cutoff": f"{cutoff:%Y-%m}", **({"v2": v2} if v2 else {}),
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

    print(json.dumps({k: results[k] for k in ("panel", "tier_months_used", "n_outlier_models", "n_repricing_events")}, indent=1, default=str))
    print("hedonic:", {k: hed[k] for k in ("falls_x_per_year", "falls_x_per_year_ci95", "r_squared", "eci_coef", "n_obs")})
    for t, f in fits.items():
        if f:
            print(t, {k: f[k] for k in ("falls_x_per_year", "falls_x_per_year_ci95", "r_squared", "n_months", "last12_annual_factor", "scenario_annual_factors", "anchor_price")})
    for t, b in results["backtest"].items():
        print("bt", t, b)
    if v2:
        print(json.dumps({k: v2[k] for k in ("forecast_meta", "rolling")}, indent=1, default=str))
    print("sens hedonic recent:", {k: hed_recent[k] for k in ("falls_x_per_year", "falls_x_per_year_ci95", "r_squared", "n_obs")})
    print("sens hedonic launch:", {k: hed_launch[k] for k in ("falls_x_per_year", "falls_x_per_year_ci95", "r_squared", "n_obs")})
    for t, f in fits_recent.items():
        print("sens tier recent", t, f and {k: f[k] for k in ("falls_x_per_year", "falls_x_per_year_ci95", "r_squared", "n_months")})
    print("age", sens["tier_median_age_months_last"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
