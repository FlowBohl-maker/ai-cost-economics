"""Cost every workflow profile in a folder, month by month, scenario by scenario and price view by price view.

  uv run python -m workflows.run_profiles                         # workflows/profiles -> reports/workflows/
  uv run python -m workflows.run_profiles my_profiles/ --out out/ --from 2025-01 --to 2027-12

Writes <out>/profile_costs.csv and <out>/profile_costs.png. Default months: the curve's anchor month (today's
base level) to +36 months.

CSV: one row per profile, month, scenario and price view.
  price_view = median    central price case: the band's tier median model
  price_view = cheapest  low price case: the cheapest model meeting the band
  cost_per_task_{low,central,high}_volume: the profile's low / central / high token volumes at that view's prices.
  band_up_*: the same view one band up.
Chart: one panel per price view, central volumes, shaded = low-high volumes (base scenario).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from .cost import PRICE_TYPES, SCENARIOS, VIEWS, PriceCurves, cost, to_month
from .profile import LEVELS, load_profiles

ROOT = Path(__file__).resolve().parents[1]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]  # fixed order, as in build_curve
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {"base": "-", "fast": (0, (4, 2)), "slow": (0, (1, 2))}
BAND_UP = (0, (6, 2, 1, 2))
VIEW_ORDER = ["median", "cheapest"]  # central price case first
VIEW_TITLE = {v: f"{VIEWS[v]['price_case'].capitalize()} price case: {VIEWS[v]['label']}" for v in VIEWS}


def cost_table(profiles, curves: PriceCurves, months, scenarios=SCENARIOS) -> pd.DataFrame:
    rows = []
    for p in profiles:
        for sc in scenarios:
            for m in months:
                c = cost(p, m, sc, curves)
                for view in VIEW_ORDER:
                    v = c["views"][view]
                    if v is None:
                        continue
                    up = ((c["band_up"] or {}).get("views") or {}).get(view) or {}
                    rows.append({
                        "profile": p.name, "example": p.example, "band": c["band"], "effort": p.effort,
                        "month": c["month"], "months_ahead": (m.year - curves.anchor.year) * 12 + m.month - curves.anchor.month,
                        "scenario": sc, "period": c["period"],
                        "price_view": view, "price_case": v["price_case"], "price_view_label": v["label"], "basis": v["basis"],
                        **{f"price_{t}": v["prices_per_mtok"][t] for t in PRICE_TYPES},
                        **{f"cost_per_task_{lv}_volume": v["cost_per_task"][lv] for lv in LEVELS},
                        **{f"monthly_cost_{lv}_volume": (v["monthly_cost"] or {}).get(lv) for lv in LEVELS},
                        "band_up": (c["band_up"] or {}).get("band") if up else None, "band_up_basis": up.get("basis"),
                        **{f"band_up_cost_per_task_{lv}_volume": (up.get("cost_per_task") or {}).get(lv) for lv in LEVELS},
                        **{f"band_up_monthly_cost_{lv}_volume": (up.get("monthly_cost") or {}).get(lv) for lv in LEVELS},
                        "notes": " | ".join(c["notes"]), "warnings": " | ".join(c["warnings"])})
    return pd.DataFrame(rows)


def chart(df: pd.DataFrame, path: Path):
    d = df.assign(month=pd.to_datetime(df.month))
    names = list(dict.fromkeys(d.profile))
    color = {n: PALETTE[i] if i < len(PALETTE) else INK2 for i, n in enumerate(names)}  # 8th+ profile: grey; split the folder
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    for ax, view in zip(axes, VIEW_ORDER):
        dv = d[d.price_view == view]
        for name in names:
            g = dv[dv.profile == name]
            if g.empty:
                continue
            b = g[g.scenario == "base"]
            if not b.empty:
                ax.fill_between(b.month, b.cost_per_task_low_volume, b.cost_per_task_high_volume, color=color[name], alpha=0.12, lw=0)
                if b.band_up_cost_per_task_central_volume.notna().any():
                    ax.plot(b.month, b.band_up_cost_per_task_central_volume, color=color[name], lw=1, ls=BAND_UP)
            for sc, s in g.groupby("scenario"):
                ax.plot(s.month, s.cost_per_task_central_volume, color=color[name], lw=2, ls=STYLE.get(sc, "-"))
        missing = [n for n in names if n not in set(dv.profile)]
        if missing:
            ax.text(0.01, 0.01, f"no {view} view for: {', '.join(missing)}", transform=ax.transAxes, fontsize=8.5,
                    color=INK2, va="bottom")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.set_title(VIEW_TITLE[view], loc="left", fontsize=11, color=INK, pad=10)
        ax.tick_params(colors=INK2, labelsize=8.5, length=0)
        ax.grid(True, which="major", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("USD per task, central token volumes (log)", color=INK2, fontsize=9)
    ex = d.groupby("profile").example.first()
    band = d.groupby("profile").band.first()
    handles = [Line2D([], [], color=color[n], lw=2, label=f"{n}" + (f" (band {band[n]})" if isinstance(band[n], str) else "")
                      + (" - EXAMPLE, made-up volumes" if ex[n] else "")) for n in names]
    sc_label = {"slow": "slow scenario (no change; under base where base is flat)"}
    handles += [Line2D([], [], color=INK2, lw=2, ls=STYLE[sc], label=sc_label.get(sc, f"{sc} scenario")) for sc in d.scenario.unique()]
    handles += [plt.Rectangle((0, 0), 1, 1, color=INK2, alpha=0.15, label="low-high token volumes (base)")]
    if d.band_up_cost_per_task_central_volume.notna().any():
        handles += [Line2D([], [], color=INK2, lw=1, ls=BAND_UP, label="same price view, one band up (base)")]
    fig.legend(handles=handles, frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", default=str(ROOT / "workflows" / "profiles"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "workflows"))
    ap.add_argument("--from", dest="start", help="first month, YYYY-MM (default: curve anchor month)")
    ap.add_argument("--to", dest="end", help="last month, YYYY-MM (default: anchor + 36 months)")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--curves", default=str(ROOT / "data" / "curves"), help="folder with curve_v4 files")
    a = ap.parse_args(argv)
    curves = PriceCurves.load(a.curves)
    profiles = load_profiles(a.folder)
    if not profiles:
        raise SystemExit(f"no *.yaml profiles in {a.folder}")
    months = pd.date_range(to_month(a.start) if a.start else curves.anchor, to_month(a.end) if a.end else curves.end, freq="MS")
    df = cost_table(profiles, curves, months, a.scenarios.split(","))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "profile_costs.csv", index=False)
    chart(df, out / "profile_costs.png")
    first, last = df.month.min(), df.month.max()
    s = df[df.month.isin([first, last])].pivot_table(index=["profile", "price_view", "scenario"], columns="month",
                                                    values="cost_per_task_central_volume")
    print("cost per task at central token volumes (price_view: median = central price case, cheapest = low):")
    print(s.map(lambda v: f"${v:.4g}").to_string())
    u = df[(df.month == first) & (df.scenario == "base") & df.band_up.notna()]
    for r in u.itertuples():
        print(f"{r.profile}, {r.price_view} view: one band up ({r.band} -> {r.band_up}) in {first}: "
              f"${r.band_up_cost_per_task_central_volume:.4g} per task")
    for col in ("notes", "warnings"):
        for name, w in df[df[col] != ""].groupby("profile")[col].first().items():
            print(f"{col[:-1]}, {name}: {w}")
    if df.example.any():
        print("example profiles (made-up volumes):", ", ".join(sorted(df[df.example].profile.unique())))
    print(f"wrote {out / 'profile_costs.csv'} ({len(df)} rows) and {out / 'profile_costs.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
