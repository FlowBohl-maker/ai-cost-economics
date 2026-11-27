"""Cost every workflow profile in a folder, month by month and scenario by scenario.

  uv run python -m workflows.run_profiles                         # workflows/profiles -> reports/workflows/
  uv run python -m workflows.run_profiles my_profiles/ --out out/ --from 2025-01 --to 2027-12

Writes <out>/profile_costs.csv (one row per profile, month and scenario) and <out>/profile_costs.png
(central cost per task over time; shaded = low-high token volumes under the base scenario).
Default months: the curve's anchor month (today's base level) to +36 months.
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

from .cost import SCENARIOS, PriceCurves, cost, to_month
from .profile import LEVELS, load_profiles

ROOT = Path(__file__).resolve().parents[1]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]  # fixed order, as in build_curve
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {"base": "-", "fast": (0, (4, 2)), "slow": (0, (1, 2))}


def cost_table(profiles, curves: PriceCurves, months, scenarios=SCENARIOS) -> pd.DataFrame:
    rows = []
    for p in profiles:
        for sc in scenarios:
            for m in months:
                c = cost(p, m, sc, curves)
                rows.append({"profile": p.name, "example": p.example, "basis": c["basis"], "effort": p.effort,
                             "month": c["month"], "months_ahead": (m.year - curves.anchor.year) * 12 + m.month - curves.anchor.month,
                             "scenario": sc, "period": c["period"],
                             **{f"price_{t}": v for t, v in c["prices_per_mtok"].items()},
                             **{f"cost_per_task_{lv}": c["cost_per_task"][lv] for lv in LEVELS},
                             **{f"monthly_cost_{lv}": (c["monthly_cost"] or {}).get(lv) for lv in LEVELS}})
    return pd.DataFrame(rows)


def chart(df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    d = df.assign(month=pd.to_datetime(df.month))
    names = list(dict.fromkeys(d.profile))
    for i, name in enumerate(names):
        c = PALETTE[i % len(PALETTE)] if i < len(PALETTE) else INK2  # an 8th+ profile is grey: split the folder instead
        g = d[d.profile == name]
        b = g[g.scenario == "base"]
        if not b.empty:
            ax.fill_between(b.month, b.cost_per_task_low, b.cost_per_task_high, color=c, alpha=0.12, lw=0)
        for sc, s in g.groupby("scenario"):
            ax.plot(s.month, s.cost_per_task_central, color=c, lw=2, ls=STYLE.get(sc, "-"))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_title("Cost per task, central token volumes", loc="left", fontsize=12, color=INK, pad=10)
    ax.set_ylabel("USD per task (log)", color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ex = d.groupby("profile").example.first()
    basis = d.groupby("profile").basis.first().str.split(",").str[0]
    handles = [Line2D([], [], color=PALETTE[i % len(PALETTE)] if i < len(PALETTE) else INK2, lw=2,
                      label=f"{n} ({basis[n]})" + (" - EXAMPLE, made-up volumes" if ex[n] else "")) for i, n in enumerate(names)]
    sc_label = {"slow": "slow scenario (no change; under base where base is flat)"}
    handles += [Line2D([], [], color=INK2, lw=2, ls=STYLE[sc], label=sc_label.get(sc, f"{sc} scenario")) for sc in d.scenario.unique()]
    handles += [plt.Rectangle((0, 0), 1, 1, color=INK2, alpha=0.15, label="low-high volumes (base)")]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper center",
              bbox_to_anchor=(0.5, -0.08), ncol=2)
    fig.tight_layout()
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
    s = df[df.month.isin([first, last])].pivot_table(index=["profile", "scenario"], columns="month",
                                                    values="cost_per_task_central")
    print(s.map(lambda v: f"${v:.4g}").to_string())
    if df.example.any():
        print("note: profiles marked example use made-up volumes:", ", ".join(sorted(df[df.example].profile.unique())))
    print(f"wrote {out / 'profile_costs.csv'} ({len(df)} rows) and {out / 'profile_costs.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
