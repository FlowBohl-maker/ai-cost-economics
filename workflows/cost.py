"""Cost per task for a workflow profile, from the token price curve (curve_v4).

  from workflows.cost import cost
  from workflows.profile import load_profile
  cost(load_profile("workflows/profiles/example_placeholder.yaml"), "2027-06", "base")

Two price views, each at low / central / high token volumes:
- cheapest (low price case): the cheapest model with ECI >= the band's lower edge (config/capability_bands.yaml),
  month by month (capability_v4). For model.min_eci, the same lookup at ceil(min_eci); below the lookup's lowest
  threshold (120) the ECI >= 120 price is used, an upper bound. A threshold near a band edge gets a warning.
- median (central price case): the median model of the band's median tier (small / mid / frontier in curve_v4),
  every token type from the same model(s). The floor band has none.
- model.tier: the tier's own view only (eci>=N tiers are cheapest-model indices; small / mid / frontier are medians).
The output also prices both views one band up, as a sensitivity (not for tier profiles).
Months from the curve's anchor month (today's base level, months_ahead 0) to +36 come from the forward curve;
earlier months from history, where the scenario does not matter. Reasoning tokens are billed at the output price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .bands import Bands, load_bands
from .profile import LEVELS, Profile

ROOT = Path(__file__).resolve().parents[1]
CURVES = ROOT / "data" / "curves"
SCENARIOS = ("base", "fast", "slow")
PRICE_TYPES = ("input", "output", "cache_read", "cache_write")
PRICE_OF = {"input": "input", "output": "output", "reasoning": "output", "cache_read": "cache_read", "cache_write": "cache_write"}


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
    curve: pd.DataFrame        # month, months_ahead, tier, scenario, input, output, cache_read, cache_write
    history: pd.DataFrame      # month, tier, input, output, cache_read, cache_write
    capability: pd.DataFrame   # month, months_ahead, eci_min, scenario ('history' for past months), model, prices

    def __post_init__(self):
        for df in (self.curve, self.history, self.capability):
            df["month"] = pd.to_datetime(df["month"])
            if "cache_write" not in df:  # files built before cache writes were priced: price them as input
                df["cache_write"] = df["input"]
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

    def _check(self, m: pd.Timestamp, scenario: str):
        if scenario not in SCENARIOS:
            raise CostError(f"scenario must be one of {', '.join(SCENARIOS)}")
        if m > self.end:
            raise CostError(f"{m:%Y-%m} is past the curve's last month {self.end:%Y-%m}")

    def _row(self, table, key, what: str) -> pd.Series:
        if key not in table.index:
            raise CostError(f"no {what} price in {key[-1]:%Y-%m}")
        return table.loc[key]

    def period(self, month) -> str:
        m = to_month(month)
        return "history" if m < self.anchor else "today" if m == self.anchor else "forecast"

    def _result(self, row: pd.Series, basis: str, m: pd.Timestamp) -> dict:
        out = {t: float(row[t]) for t in PRICE_TYPES}
        out["basis"] = basis + (f", {row['model']}" if "model" in row.index else "")
        out["period"] = self.period(m)
        return out

    def tier_prices(self, tier: str, month, scenario: str = "base") -> dict:
        m = to_month(month)
        self._check(m, scenario)
        if tier not in self.tiers:
            raise CostError(f"unknown tier {tier!r}; available: {', '.join(self.tiers)}")
        key, table = ((tier, m), self._hist) if m < self.anchor else ((tier, scenario, m), self._curve)
        return self._result(self._row(table, key, f"tier {tier}"), f"tier {tier}", m)

    def capability_prices(self, min_eci: float, month, scenario: str = "base") -> dict:
        """Cheapest model with ECI >= ceil(min_eci)."""
        m = to_month(month)
        self._check(m, scenario)
        thr = max(math.ceil(min_eci), self.eci_range[0])
        if thr > self.eci_range[1]:
            raise CostError(f"no priced model with ECI >= {min_eci:g} (highest: {self.eci_range[1]})")
        basis = f"ECI >= {thr}" + (f" (lookup starts at {thr}: upper bound)" if math.ceil(min_eci) < thr else "")
        key = (thr, "history" if m < self.anchor else scenario, m)
        return self._result(self._row(self._cap, key, basis), basis, m)


_DEFAULT: PriceCurves | None = None


def default_curves() -> PriceCurves:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = PriceCurves.load()
    return _DEFAULT


VIEWS = {
    "cheapest": {"price_case": "low", "label": "cheapest model meeting the band"},
    "median": {"price_case": "central", "label": "tier median model"},
}


def _costs(profile: Profile, pr: dict) -> tuple[dict, dict | None]:
    per_task = {lv: sum(profile.tokens[t][lv] * pr[PRICE_OF[t]] for t in profile.tokens) / 1e6 for lv in LEVELS}
    monthly = None if profile.runs_per_month is None else {lv: per_task[lv] * profile.runs_per_month[lv] for lv in LEVELS}
    return per_task, monthly


def _view(name: str, profile: Profile, pr: dict) -> dict:
    per_task, monthly = _costs(profile, pr)
    return {"view": name, **VIEWS[name], "basis": pr["basis"], "prices_per_mtok": {t: pr[t] for t in PRICE_TYPES},
            "cost_per_task": per_task, "monthly_cost": monthly}


def _band_views(band, profile: Profile, curves: "PriceCurves", month, scenario, eci: float | None = None) -> dict:
    """Both price views for a band: cheapest model at or above `eci` (default: the band's edge), and the band's
    median tier (None where the band has none)."""
    cheap = curves.capability_prices(band.min_eci if eci is None else eci, month, scenario)
    cheap = {**cheap, "basis": f"band {band.name}: {cheap['basis']}" if eci is None else cheap["basis"]}
    views = {"cheapest": _view("cheapest", profile, cheap), "median": None}
    if band.median_tier:
        med = curves.tier_prices(band.median_tier, month, scenario)
        views["median"] = _view("median", profile, {**med, "basis": f"band {band.name}: median of {med['basis']}"})
    return views


def cost(profile: Profile, month, scenario: str = "base", curves: PriceCurves | None = None,
         bands: Bands | None = None) -> dict:
    """Cost per task (USD) and monthly cost under two price views, each at low / central / high token volumes.

    views["cheapest"]: low price case, the cheapest model meeting the band (or the exact min_eci).
    views["median"]:   central price case, the median model of the band's median tier (config median_tier).
    A view is None where it doesn't apply: floor has no median tier; a tier profile has only its own view
    (eci>=N tiers are cheapest-model indices, small / mid / frontier are medians).
    band_up holds the same two views one band up (None for tier profiles and the top band).

    Volume levels spread the token volumes only; scenario covers price changes over time. Monthly cost pairs
    levels (low volumes x low runs, ...), so its low and high are bounds, not percentiles.
    """
    curves = curves or default_curves()
    bands = bands or load_bands()
    m = f"{to_month(month):%Y-%m}"
    notes, band = [], None
    try:
        if profile.tier is not None:
            pr = curves.tier_prices(profile.tier, month, scenario)
            name = "cheapest" if profile.tier.startswith("eci>=") else "median"
            views = {"cheapest": None, "median": None, name: _view(name, profile, pr)}
            notes.append(f"tier profile: only the {name} view ({VIEWS[name]['label']}) applies")
        else:
            if profile.band is not None:
                band = bands.get(profile.band)
                views = _band_views(band, profile, curves, month, scenario)
            else:  # an exact threshold falls in the band of the (rounded-up) threshold it is priced at
                band = bands.containing(math.ceil(profile.min_eci))
                views = _band_views(band, profile, curves, month, scenario, eci=profile.min_eci)
            if views["median"] is None:
                notes.append(f"band {band.name} has no median tier, so there is no central price case")
        up = None
        if band is not None and (nb := bands.up(band.name)) is not None:
            up = {"band": nb.name, "views": _band_views(nb, profile, curves, month, scenario)}
    except CostError as e:
        raise CostError(f"{profile.name}: {e}") from None
    period = next(v for v in views.values() if v is not None)
    return {"profile": profile.name, "month": m, "scenario": scenario, "example": profile.example,
            "band": band.name if band else None, "period": curves.period(month), "views": views, "band_up": up,
            "warnings": bands.edge_warnings(profile.min_eci) if profile.min_eci is not None else [], "notes": notes}
