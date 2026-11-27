"""Cost per task for a workflow profile, from the token price curve (curve_v4).

  from workflows.cost import cost, PriceCurves
  from workflows.profile import load_profile
  cost(load_profile("workflows/profiles/example_placeholder.yaml"), "2027-06", "base")

Prices:
- model.tier: the tier's curve_v4 price per token type. Months from the anchor month (today's base level,
  months_ahead 0) to +36 come from the forward curve; earlier months from the tier's monthly history.
- model.min_eci: the cheapest model with ECI >= ceil(min_eci) (capability_v4). Below the lowest threshold
  in the lookup (120), the ECI >= 120 price is used: an upper bound.
Reasoning tokens are billed at the output price. Past months use history, so the scenario does not matter there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .profile import LEVELS, Profile

ROOT = Path(__file__).resolve().parents[1]
CURVES = ROOT / "data" / "curves"
SCENARIOS = ("base", "fast", "slow")
PRICE_TYPE = {"input": "input", "output": "output", "reasoning": "output", "cache_read": "cache_read"}


class CostError(ValueError):
    pass


def to_month(m) -> pd.Timestamp:
    try:
        return pd.Timestamp(m).to_period("M").to_timestamp()
    except (ValueError, TypeError) as e:
        raise CostError(f"not a month: {m!r}") from e


@dataclass
class PriceCurves:
    """The three curve_v4 tables the cost model reads (USD per 1M tokens)."""
    curve: pd.DataFrame        # month, months_ahead, tier, scenario, input, output, cache_read
    history: pd.DataFrame      # month, tier, input, output, cache_read
    capability: pd.DataFrame   # month, months_ahead, eci_min, scenario ('history' for past months), model, ...

    def __post_init__(self):
        for df in (self.curve, self.history, self.capability):
            df["month"] = pd.to_datetime(df["month"])
        self.anchor = self.curve.loc[self.curve.months_ahead == 0, "month"].min()
        self.end = self.curve.month.max()
        self.tiers = sorted(self.curve.tier.unique())
        self.eci_range = (int(self.capability.eci_min.min()), int(self.capability.eci_min.max()))
        self._curve = self.curve.set_index(["tier", "scenario", "month"]).sort_index()
        self._hist = self.history.set_index(["tier", "month"]).sort_index()
        self._cap = self.capability.set_index(["eci_min", "scenario", "month"]).sort_index()

    @classmethod
    def load(cls, folder: str | Path = CURVES, version: str = "v4") -> "PriceCurves":
        folder = Path(folder)
        return cls(pd.read_csv(folder / f"curve_{version}.csv"), pd.read_csv(folder / f"tier_history_{version}.csv"),
                   pd.read_csv(folder / f"capability_{version}.csv"))

    def prices(self, profile: Profile, month, scenario: str = "base") -> dict:
        """USD per 1M tokens for each token type, plus where the price came from."""
        m = to_month(month)
        if scenario not in SCENARIOS:
            raise CostError(f"scenario must be one of {', '.join(SCENARIOS)}")
        if m > self.end:
            raise CostError(f"{m:%Y-%m} is past the curve's last month {self.end:%Y-%m}")
        past = m < self.anchor
        if profile.tier is not None:
            if profile.tier not in self.tiers:
                raise CostError(f"{profile.name}: unknown tier {profile.tier!r}; available: {', '.join(self.tiers)}")
            key, table, what = ((profile.tier, m), self._hist, "history") if past else ((profile.tier, scenario, m), self._curve, "curve")
            basis = f"tier {profile.tier}"
        else:
            thr = max(math.ceil(profile.min_eci), self.eci_range[0])
            if thr > self.eci_range[1]:
                raise CostError(f"{profile.name}: no priced model with ECI >= {profile.min_eci:g} (highest: {self.eci_range[1]})")
            key, table, what = (thr, "history" if past else scenario, m), self._cap, "capability lookup"
            basis = f"ECI >= {thr}" + (" (lookup starts at 120: upper bound)" if math.ceil(profile.min_eci) < thr else "")
        if key not in table.index:
            raise CostError(f"{profile.name}: no {what} price for {basis} in {m:%Y-%m}")
        row = table.loc[key]
        out = {t: float(row[t]) for t in ("input", "output", "cache_read")}
        out["basis"] = basis + (f", {row['model']}" if "model" in row.index else "")
        out["period"] = "history" if past else "today" if m == self.anchor else "forecast"
        return out


_DEFAULT: PriceCurves | None = None


def default_curves() -> PriceCurves:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = PriceCurves.load()
    return _DEFAULT


def cost(profile: Profile, month, scenario: str = "base", curves: PriceCurves | None = None) -> dict:
    """Cost per task (USD) at low / central / high token volumes, and monthly cost if runs_per_month is set.

    Low / central / high spread the token volumes only; price uncertainty is the scenario argument.
    Monthly cost pairs levels (low volumes x low runs, ...), so its low and high are bounds, not percentiles.
    """
    pr = (curves or default_curves()).prices(profile, month, scenario)
    per_task = {lv: sum(profile.tokens[t][lv] * pr[PRICE_TYPE[t]] for t in profile.tokens) / 1e6 for lv in LEVELS}
    monthly = None if profile.runs_per_month is None else {lv: per_task[lv] * profile.runs_per_month[lv] for lv in LEVELS}
    return {"profile": profile.name, "month": f"{to_month(month):%Y-%m}", "scenario": scenario, "example": profile.example,
            "basis": pr["basis"], "period": pr["period"],
            "prices_per_mtok": {t: pr[t] for t in ("input", "output", "cache_read")},
            "cost_per_task": per_task, "monthly_cost": monthly}
