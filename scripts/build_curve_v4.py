"""Token price curve v4: curve v3 split into input, output, cache-read and cache-write prices, plus a capability lookup.

  uv run python scripts/build_curve_v4.py      # run after build_curve.py (v3); v3's files are read, never written

v4 keeps v3's tiers, base levels and base-case rates. It adds:
1. One monthly price per token type (input, output, cache read, cache write) per tier and scenario, months 0-36.
   Each type moves at its tier's v3 annual factor (v3 fits rates on the blended price only).
2. A capability lookup: for every integer ECI threshold, the cheapest model at or above it each month
   (history), and its price per token type projected 36 months forward under each scenario.

Cache-read and cache-write prices: the cache-read / input and cache-write / input ratios listed by the same
price source the model's input price comes from (maker's own LiteLLM rows, the v3 reference provider, or
OpenRouter). Where that source lists no such price, those tokens are priced as ordinary input and
cache_read_listed / cache_write_listed = False. A cache write can cost more than input (e.g. 1.25x).

Outputs: data/curves/curve_v4.csv, curve_v4.json, tier_history_v4.csv, capability_v4.csv,
         reports/curve_v4_capability.png, reports/curve_v4_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_curve as bc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT, REP = ROOT / "data", ROOT / "data" / "curves", ROOT / "reports"
TYPES = ["input", "output", "cache_read", "cache_write"]
CACHE_TYPES = {"cache_read": "cache_read_price_per_mtok", "cache_write": "cache_write_price_per_mtok"}
SCENARIOS = ["base", "fast", "slow"]
CAP_MIN = 120  # lowest threshold in the capability lookup (v3's lowest band)


def v3_setup():
    bc.VERSION = "v3"
    bc.ECI_THRESHOLDS = bc.V2_THRESHOLDS
    bc.TIER_NAMES = [f"eci>={t}" for t in bc.ECI_THRESHOLDS] + [n for n, _, _ in bc.GAP_TIERS]


def cache_ratios(panel: pd.DataFrame) -> pd.DataFrame:
    """Cache-read / input and cache-write / input ratios per (month, model), from the same price source as the
    panel's input price. Unlisted: ratio 1 (priced as input). Cache reads never cost more than input."""
    h = pd.read_parquet(DATA / "price_history.parquet")
    h = h[h.canonical_id.notna() & (h.source == "litellm") & (h.input_price_per_mtok > 0)].copy()
    vendor = h.canonical_id.str.split("/").str[0]
    fp = np.array([p in bc.FIRST_PARTY.get(v, set()) for p, v in zip(h.provider.fillna(""), vendor)], dtype=bool)
    other = h[~fp]
    ref = bc.reference_prices(other)
    ref = ref[ref.tier_src == "litellm_ref"][["snapshot_date", "canonical_id", "provider"]].drop_duplicates()
    rows = [h[fp].assign(src="litellm_first_party"),
            other.merge(ref, on=["snapshot_date", "canonical_id", "provider"]).assign(src="litellm_ref"),
            other[~other.provider.isin(bc.GATEWAYS)].assign(src="litellm_median")]
    orp = pd.read_parquet(DATA / "prices.parquet")
    mp = pd.read_csv(DATA / "model_map.csv")
    orp = orp.merge(mp[mp.source == "openrouter"][["source_id", "canonical_id"]], left_on="model_id", right_on="source_id")
    orp["snapshot_date"] = orp.snapshot_date.dt.to_period("M").dt.to_timestamp()
    rows.append(orp[orp.input_price_per_mtok > 0].assign(src="openrouter"))
    r = pd.concat([x[["snapshot_date", "canonical_id", "src", "input_price_per_mtok", *CACHE_TYPES.values()]] for x in rows])
    out = panel[["month", "canonical_id", "price_source"]]
    for t, col in CACHE_TYPES.items():
        r[t] = r[col] / r.input_price_per_mtok
        g = r.groupby(["snapshot_date", "canonical_id", "src"])[t].agg(**{f"{t}_ratio": "median", "n_listed": "count"}).reset_index()
        g = g.rename(columns={"snapshot_date": "month", "src": "price_source"})
        out = out.merge(g, on=["month", "canonical_id", "price_source"], how="left")
        out[f"{t}_listed"] = out.pop("n_listed").fillna(0) > 0
        out[f"{t}_ratio"] = out[f"{t}_ratio"].where(out[f"{t}_listed"], 1.0)
    out["cache_read_ratio"] = out.cache_read_ratio.clip(upper=1.0)
    return out[["month", "canonical_id", *[f"{t}_{k}" for t in CACHE_TYPES for k in ("ratio", "listed")]]]


def panel_v3() -> pd.DataFrame:
    """v3's model-month panel with hedonic outliers removed, plus cache-read and cache-write prices."""
    panel = bc.build_panel(ref_provider=True)
    p = bc.assign_tiers(panel)
    res, _ = bc.fit_hedonic(p)
    p = p[np.abs(np.asarray(res.get_influence().resid_studentized_internal)) <= 3].copy()
    p = p.merge(cache_ratios(panel), on=["month", "canonical_id"], how="left")
    for t in CACHE_TYPES:
        p[f"{t}_listed"] = p[f"{t}_listed"].fillna(False).astype(bool)
        p[t] = p.input * p[f"{t}_ratio"].fillna(1.0)
    return p


def gap_members(p: pd.DataFrame) -> pd.DataFrame:
    """The models in each v3 gap tier each month (same selection as build_curve.gap_tier_index)."""
    q = p.copy()
    q["gap"] = q.groupby("month").eci.transform("max") - q.eci
    rel = q.epoch_release_date.fillna(q.groupby("canonical_id").month.transform("min"))
    q["age"] = (q.month.dt.year - rel.dt.year) * 12 + q.month.dt.month - rel.dt.month
    return pd.concat([q[(q.gap >= lo) & (q.gap < hi) & (q.age <= bc.GAP_TIER_MAX_AGE)].assign(tier=name)
                      for name, lo, hi in bc.GAP_TIERS], ignore_index=True)


def tier_history(p: pd.DataFrame, idx: pd.DataFrame) -> pd.DataFrame:
    bands = idx[idx.tier.str.startswith("eci>=")].merge(
        p[["month", "canonical_id", "cache_read", "cache_write", "cache_read_listed", "cache_write_listed"]],
        left_on=["month", "cheapest_model"], right_on=["month", "canonical_id"], how="left").drop(columns="canonical_id")
    mem = gap_members(p)
    gap = mem.groupby(["month", "tier"]).agg(cache_read=("cache_read", "median"), cache_write=("cache_write", "median"),
                                             cache_read_listed_share=("cache_read_listed", "mean"),
                                             cache_write_listed_share=("cache_write_listed", "mean")).reset_index()
    gap = idx[~idx.tier.str.startswith("eci>=")].merge(gap, on=["month", "tier"], how="left")
    h = pd.concat([bands, gap], ignore_index=True).rename(columns={"index_input": "input", "index_output": "output",
                                                                   "index_price": "blended_v3"})
    h["blended"] = bc.INPUT_WEIGHT * h.input + (1 - bc.INPUT_WEIGHT) * h.output
    cols = ["month", "tier", *TYPES, "blended", "blended_v3", "cheapest_model", "cheapest_eci", "price_source",
            "cache_read_listed", "cache_write_listed", "cache_read_listed_share", "cache_write_listed_share", "n_models", "used"]
    return h[cols].sort_values(["tier", "month"]).reset_index(drop=True)


def model_base(p: pd.DataFrame, model: str, upto: pd.Timestamp) -> dict:
    """A model's price per token type: median of its last BASE_MEDIAN_MONTHS months (v3's base level rule)."""
    mp = p[(p.canonical_id == model) & (p.month <= upto)].sort_values("month").tail(bc.BASE_MEDIAN_MONTHS)
    return {**{t: float(mp[t].median()) for t in TYPES}, "cache_read_listed": bool(mp.cache_read_listed.iloc[-1]),
            "cache_write_listed": bool(mp.cache_write_listed.iloc[-1])}


def project(base: dict, factors: dict, start: pd.Timestamp, months: int = bc.HORIZON) -> pd.DataFrame:
    ks = np.arange(months + 1)
    rows = []
    for sc in SCENARIOS:
        f = factors[sc] ** (ks / 12)
        rows.append(pd.DataFrame({"month": [start + pd.DateOffset(months=int(k)) for k in ks], "months_ahead": ks,
                                  "scenario": sc, "annual_factor": factors[sc], **{t: base[t] * f for t in TYPES}}))
    d = pd.concat(rows, ignore_index=True)
    d["blended"] = bc.INPUT_WEIGHT * d.input + (1 - bc.INPUT_WEIGHT) * d.output
    return d


def capability(p: pd.DataFrame, v3: dict, last: pd.Timestamp):
    """Cheapest model at or above each integer ECI threshold: monthly history, then a 36-month projection.

    Forward rate: the v3 band priced by the same model if there is one (as in v3's 'same model' rule), else the
    nearest v3 band at or below the threshold."""
    bands = {int(t[5:]): t for t in v3 if t.startswith("eci>=")}
    tmax = int(np.floor(p[p.month == last].eci.max()))
    hist, fwd, today = [], [], []
    for thr in range(CAP_MIN, tmax + 1):
        q = p[p.eci >= thr]
        ch = q.loc[q.groupby("month").blended.idxmin()]
        n = q.groupby("month").canonical_id.nunique()
        hist.append(ch[ch.month < last].assign(eci_min=thr, n_models=lambda d: d.month.map(n))[
            ["month", "eci_min", "canonical_id", "eci", *TYPES, "blended", "cache_read_listed", "cache_write_listed", "n_models"]])
        model = ch[ch.month == last].canonical_id.iloc[0]
        same = [b for b, t in bands.items() if v3[t]["base_case"].get("cheapest_model") == model]
        src = bands[max(same)] if same else bands[max(b for b in bands if b <= thr)]
        b = model_base(p, model, last)
        pr = project(b, v3[src]["base_case"]["scenario_annual_factors"], last)
        fwd.append(pr.assign(eci_min=thr, canonical_id=model, eci=float(q[q.canonical_id == model].eci.iloc[0]),
                             cache_read_listed=b["cache_read_listed"], cache_write_listed=b["cache_write_listed"],
                             n_models=int(n[last]), rate_from=src))
        today.append({"eci_min": thr, "model": model, "eci": float(q[q.canonical_id == model].eci.iloc[0]),
                      **{t: b[t] for t in TYPES}, "rate_from": src, "rate_rule": "same model" if same else "nearest band below",
                      "base_annual_factor": v3[src]["base_case"]["scenario_annual_factors"]["base"]})
    h = pd.concat(hist, ignore_index=True).assign(scenario="history", months_ahead=lambda d: [
        -bc.months_between(m, last) for m in d.month], annual_factor=np.nan, rate_from="")
    cols = ["month", "months_ahead", "eci_min", "scenario", "canonical_id", "eci", *TYPES, "blended",
            "cache_read_listed", "cache_write_listed", "n_models", "rate_from", "annual_factor"]
    cap = pd.concat([h, pd.concat(fwd, ignore_index=True)], ignore_index=True)[cols].rename(columns={"canonical_id": "model", "eci": "model_eci"})
    return cap.sort_values(["eci_min", "scenario", "month"]).reset_index(drop=True), pd.DataFrame(today)


def chart(today: pd.DataFrame, cap: pd.DataFrame, path):
    ink, ink2, grid = bc.INK, bc.INK2, bc.GRID
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    end = cap[cap.months_ahead == bc.HORIZON]
    series = [
        (axes[0], [(today.input, "Input", bc.PALETTE[0], "-"), (today.output, "Output", bc.PALETTE[1], "-"),
                   (today.cache_read, "Cache read", bc.PALETTE[2], "-")],
         "Cheapest price at or above each ECI level today, by token type"),
        (axes[1], [(end[end.scenario == "base"].set_index("eci_min").blended.reindex(today.eci_min).to_numpy(), "+36 months, base", bc.PALETTE[3], "-"),
                   (end[end.scenario == "fast"].set_index("eci_min").blended.reindex(today.eci_min).to_numpy(), "+36 months, fast", bc.PALETTE[4], "-"),
                   (today.input * bc.INPUT_WEIGHT + today.output * (1 - bc.INPUT_WEIGHT), "Today (base is flat where they overlap)", ink, (0, (3, 2)))],
         "Blended (3:1) price: today and in 36 months"),
    ]
    for ax, lines, title in series:
        x = np.append(today.eci_min.to_numpy(), today.eci_min.max() + 1)  # give the last threshold a width
        for y, label, c, ls in lines:
            y = np.asarray(y, dtype=float)
            ax.step(x, np.append(y, y[-1]), where="post", color=c, lw=2, ls=ls, label=label)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
        ax.set_title(title, loc="left", fontsize=11, color=ink, pad=10)
        ax.set_xlabel("Minimum Epoch ECI", color=ink2, fontsize=9)
        ax.tick_params(colors=ink2, labelsize=8.5, length=0)
        ax.grid(True, which="major", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=ink2, loc="upper left")
    axes[0].set_ylabel("USD per 1M tokens (log)", color=ink2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    v3_setup()
    v3 = json.loads((OUT / "curve_v3.json").read_text())["tiers"]
    p = panel_v3()
    last = p.month.max()
    idx = pd.concat([bc.band_index(p), bc.gap_tier_index(p)], ignore_index=True)

    # check: v4 reproduces v3's tier history exactly
    th3 = pd.read_csv(OUT / "tier_history_v3.csv", parse_dates=["month"])
    chk = th3.merge(idx[["month", "tier", "index_price"]], on=["month", "tier"], suffixes=("_v3", "_v4"), how="outer")
    max_diff = float((chk.index_price_v3 - chk.index_price_v4).abs().max())
    assert len(chk) == len(th3) and max_diff < 1e-9, f"v4 does not reproduce v3's tier history (max diff {max_diff})"

    hist = tier_history(p, idx)
    bases, rows = {}, []
    for tier, t in v3.items():
        bcase = t["base_case"]
        if tier.startswith("eci>="):
            b = model_base(p, bcase["cheapest_model"], last)
        else:
            d = hist[(hist.tier == tier) & hist.used].sort_values("month").tail(bc.BASE_MEDIAN_MONTHS)
            b = {**{x: float(d[x].median()) for x in TYPES}, "cache_read_listed_share": float(d.cache_read_listed_share.iloc[-1]),
                 "cache_write_listed_share": float(d.cache_write_listed_share.iloc[-1])}
        blended = bc.INPUT_WEIGHT * b["input"] + (1 - bc.INPUT_WEIGHT) * b["output"]
        bases[tier] = {**b, "blended": blended, "v3_base_price": bcase["base_price"],
                       "blended_vs_v3_pct": 100 * (blended / bcase["base_price"] - 1),
                       "cheapest_model": bcase.get("cheapest_model"), "rule": bcase["rule"], "reason": bcase["reason"],
                       "scenario_annual_factors": bcase["scenario_annual_factors"]}
        rows.append(project(b, bcase["scenario_annual_factors"], last).assign(tier=tier))
    curve = pd.concat(rows, ignore_index=True)[["month", "months_ahead", "tier", "scenario", *TYPES, "blended", "annual_factor"]]
    cap, today = capability(p, v3, last)

    # ---- write
    fmt = lambda d: d.assign(month=d.month.dt.strftime("%Y-%m-%d"))
    fmt(curve).to_csv(OUT / "curve_v4.csv", index=False)
    fmt(hist).to_csv(OUT / "tier_history_v4.csv", index=False)
    fmt(cap).to_csv(OUT / "capability_v4.csv", index=False)
    meta = {"version": "v4", "built_from": "curve_v3 (tiers, base levels, base-case rates)", "generated_from_month": f"{last:%Y-%m}",
            "units": "USD per 1M tokens", "token_types": {
                "input": "uncached input (prompt) tokens", "output": "output tokens; reasoning tokens are billed at this price",
                "cache_read": "input tokens read from a prompt cache; priced as input where the source lists no cache-read price",
                "cache_write": "input tokens written to a prompt cache; priced as input where the source lists no cache-write price"},
            "blended": f"{bc.INPUT_WEIGHT:.0%} input + {1 - bc.INPUT_WEIGHT:.0%} output, as in v3",
            "months": "months_ahead 0 = the base level in the anchor month; 1-36 = forward curve",
            "scenarios": {"base": "v3 base case per tier", "fast": "v3 fast (constant-capability decline unless the base is steeper)",
                          "slow": "no change"},
            "rate_by_type": "every token type moves at its tier's v3 annual factor (v3 fits rates on blended prices)",
            "capability_lookup": "capability_v4.csv: cheapest model with ECI >= eci_min; scenario 'history' for past months; "
                                 "forward rate from the v3 band priced by the same model, else the nearest v3 band below"}
    js = {"meta": meta, "tiers": {t: {"base": bases[t], "forecast": [
        {"month": f"{r.month:%Y-%m}", "scenario": r.scenario, **{x: round(float(getattr(r, x)), 6) for x in TYPES}}
        for r in curve[curve.tier == t].itertuples()]} for t in bases}}
    (OUT / "curve_v4.json").write_text(json.dumps(js, indent=1))
    ratio_hist = hist[hist.used].assign(r=lambda d: d.output / d.input).groupby("tier").apply(
        lambda d: {"first": f"{d.month.min():%Y-%m}", "first_ratio": float(d.r.iloc[0]), "last_ratio": float(d.r.iloc[-1]),
                   "median_ratio": float(d.r.median())}, include_groups=False).to_dict()
    results = {"anchor_month": f"{last:%Y-%m}", "v3_tier_history_max_abs_diff": max_diff, "bases": bases,
               "capability_today": today.to_dict("records"), "output_to_input_ratio": ratio_hist,
               "capability_thresholds": [int(today.eci_min.min()), int(today.eci_min.max())]}
    (REP / "curve_v4_results.json").write_text(json.dumps(results, indent=1, default=str))
    chart(today, cap, REP / "curve_v4_capability.png")

    pd.set_option("display.width", 200)
    print(pd.DataFrame(bases).T[[*TYPES, "blended", "v3_base_price", "blended_vs_v3_pct", "cache_read_listed",
                                 "cache_write_listed", "cache_read_listed_share", "cache_write_listed_share"]].to_string())
    print(today.groupby("model").agg(eci_from=("eci_min", "min"), eci_to=("eci_min", "max"), input=("input", "first"),
                                     output=("output", "first"), cache_read=("cache_read", "first"), cache_write=("cache_write", "first"),
                                     rate_from=("rate_from", lambda s: ",".join(sorted(set(s))))).sort_values("eci_from").to_string())
    print(json.dumps(ratio_hist, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
