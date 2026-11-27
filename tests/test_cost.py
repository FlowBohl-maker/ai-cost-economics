"""Tests for the workflow cost function: exact arithmetic on a small synthetic curve, then checks on curve_v4."""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

from workflows.bands import Bands, load_bands
from workflows.cost import CostError, PriceCurves, cost
from workflows.profile import EdgeWarning, ProfileError, load_profile, load_profiles, parse_profile

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = pd.Timestamp("2026-11-01")
# synthetic bands matching the synthetic capability lookup below (thresholds 120-123)
BANDS = Bands.from_dict({"edge_warning_points": 0.5, "bands": [
    {"name": "floor", "min_eci": 120}, {"name": "small", "min_eci": 121}, {"name": "mid", "min_eci": 122},
    {"name": "frontier", "min_eci": 123}]})
CAP_MODELS = {120: ("small-a", 0.1), 121: ("small-b", 0.2), 122: ("mid-a", 1.0), 123: ("top", 5.0)}


def months(n0, n1):
    return [ANCHOR + pd.DateOffset(months=k) for k in range(n0, n1 + 1)]


def tables(with_cache_write: bool = True):
    """Tier 'mid': input $2, output $10, cache read $0.5, cache write $2.5 today; base halves every year, slow is flat.
    Capability lookup: input price per threshold from CAP_MODELS, output 4x, cache read 1/10, cache write 1.25x."""
    rows = []
    for k, m in zip(range(0, 13), months(0, 12)):
        for sc, f in (("base", 0.5), ("fast", 0.25), ("slow", 1.0)):
            g = f ** (k / 12)
            rows.append({"month": m, "months_ahead": k, "tier": "mid", "scenario": sc,
                         "input": 2 * g, "output": 10 * g, "cache_read": 0.5 * g, "cache_write": 2.5 * g})
    hist = pd.DataFrame([{"month": ANCHOR - pd.DateOffset(months=1), "tier": "mid", "input": 3.0, "output": 12.0,
                          "cache_read": 3.0, "cache_write": 3.0}])
    cap = []
    for thr, (model, inp) in CAP_MODELS.items():
        for sc, f in (("base", 0.5), ("fast", 0.25), ("slow", 1.0)):
            for k, m in zip(range(0, 13), months(0, 12)):
                g = f ** (k / 12)
                cap.append({"month": m, "months_ahead": k, "eci_min": thr, "scenario": sc, "model": model,
                            "input": inp * g, "output": 4 * inp * g, "cache_read": inp * g / 10, "cache_write": 1.25 * inp * g})
        cap.append({"month": ANCHOR - pd.DateOffset(months=1), "months_ahead": -1, "eci_min": thr, "scenario": "history",
                    "model": model, "input": 2 * inp, "output": 8 * inp, "cache_read": inp, "cache_write": 2 * inp})
    out = [pd.DataFrame(rows), hist, pd.DataFrame(cap)]
    if not with_cache_write:
        out = [d.drop(columns="cache_write") for d in out]
    return out


@pytest.fixture
def curves() -> PriceCurves:
    return PriceCurves(*tables())


def profile(**kw):
    d = {"name": "t", "description": "test", "tokens_per_task": {
        "input": {"low": 1000, "central": 2000, "high": 4000}, "output": 500,
        "reasoning": {"low": 0, "central": 1000, "high": 3000}, "cache_read": {"low": 0, "central": 10000, "high": 10000}},
        "model": {"tier": "mid"}}
    d.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EdgeWarning)
        return parse_profile(d)


def c(p, month="2026-11", scenario="base", curves=None):
    return cost(p, month, scenario, curves, BANDS)


# ---------------------------------------------------------------- arithmetic


def test_cost_per_task_today(curves):
    r = c(profile(), curves=curves)
    # central: 2000*2 + 500*10 + 1000*10 (reasoning at output price) + 10000*0.5 = 24000 per 1M tokens
    assert r["cost_per_task"]["central"] == pytest.approx(0.024)
    assert r["cost_per_task"]["low"] == pytest.approx((1000 * 2 + 500 * 10) / 1e6)
    assert r["cost_per_task"]["high"] == pytest.approx((4000 * 2 + 500 * 10 + 3000 * 10 + 10000 * 0.5) / 1e6)
    assert r["monthly_cost"] is None
    assert r["period"] == "today"


def test_reasoning_billed_at_output_price(curves):
    base = profile(tokens_per_task={"input": 0, "output": 0})
    more = profile(tokens_per_task={"input": 0, "output": 0, "reasoning": 1_000_000})
    assert c(base, curves=curves)["cost_per_task"]["central"] == 0
    assert c(more, curves=curves)["cost_per_task"]["central"] == pytest.approx(10.0)


def test_cache_write_priced_separately(curves):
    p = profile(tokens_per_task={"input": 0, "output": 0, "cache_write": {"low": 0, "central": 1_000_000, "high": 2_000_000}})
    r = c(p, curves=curves)
    assert r["prices_per_mtok"]["cache_write"] == pytest.approx(2.5)
    assert r["cost_per_task"] == pytest.approx({"low": 0.0, "central": 2.5, "high": 5.0})
    assert c(p, month="2027-11", curves=curves)["cost_per_task"]["central"] == pytest.approx(1.25)  # base halves in a year
    assert c(p, month="2026-10", curves=curves)["cost_per_task"]["central"] == pytest.approx(3.0)   # history


def test_cache_write_defaults_to_input_price_when_curve_has_none():
    old = PriceCurves(*tables(with_cache_write=False))
    p = profile(tokens_per_task={"input": 0, "output": 0, "cache_write": 1_000_000})
    assert c(p, curves=old)["cost_per_task"]["central"] == pytest.approx(2.0)
    assert profile().tokens["cache_write"].central == 0  # optional in the schema


def test_scenarios_and_months(curves):
    p = profile()
    today = c(p, curves=curves)["cost_per_task"]["central"]
    assert c(p, "2027-11", "base", curves)["cost_per_task"]["central"] == pytest.approx(today * 0.5)
    assert c(p, "2027-11", "fast", curves)["cost_per_task"]["central"] == pytest.approx(today * 0.25)
    assert c(p, "2027-11", "slow", curves)["cost_per_task"]["central"] == pytest.approx(today)
    assert c(p, "2027-11-17", "base", curves)["month"] == "2027-11"
    assert c(p, pd.Timestamp("2027-11-30"), "base", curves)["cost_per_task"]["central"] == pytest.approx(today * 0.5)


def test_monthly_cost_pairs_levels(curves):
    p = profile(runs_per_month={"low": 10, "central": 100, "high": 1000})
    r = c(p, curves=curves)
    for lv, n in {"low": 10, "central": 100, "high": 1000}.items():
        assert r["monthly_cost"][lv] == pytest.approx(r["cost_per_task"][lv] * n)
    flat = c(profile(runs_per_month=250), curves=curves)
    assert flat["monthly_cost"]["central"] == pytest.approx(flat["cost_per_task"]["central"] * 250)


def test_history_before_anchor_ignores_scenario(curves):
    p = profile()
    a, b = c(p, "2026-10", "base", curves), c(p, "2026-10", "fast", curves)
    assert a["period"] == "history" and a["cost_per_task"] == b["cost_per_task"]
    assert a["prices_per_mtok"] == {"input": 3.0, "output": 12.0, "cache_read": 3.0, "cache_write": 3.0}


# ---------------------------------------------------------------- bands, band-up sensitivity, exact ECI


def test_band_priced_at_its_lower_edge(curves):
    r = c(profile(model={"band": "small"}), curves=curves)
    assert r["band"] == "small" and "mid" not in r["basis"]
    assert r["prices_per_mtok"]["input"] == pytest.approx(0.2) and "small-b" in r["basis"]


def test_band_up_sensitivity(curves):
    p = profile(model={"band": "small"}, runs_per_month=100)
    r = c(p, "2027-11", "fast", curves)
    up = r["band_up"]
    assert up["band"] == "mid" and "mid-a" in up["basis"]
    # every price in the mid band is 5x the small band's (input 1.0 vs 0.2), in every month and scenario
    for lv in ("low", "central", "high"):
        assert up["cost_per_task"][lv] == pytest.approx(5 * r["cost_per_task"][lv])
        assert up["monthly_cost"][lv] == pytest.approx(5 * r["monthly_cost"][lv])
    assert up["change_central_pct"] == pytest.approx(400)
    assert c(profile(model={"band": "frontier"}), curves=curves)["band_up"] is None  # no band above the top
    assert c(profile(), curves=curves)["band_up"] is None                            # tiers have no band-up


def test_min_eci_lookup_and_band_up(curves):
    r = c(profile(model={"min_eci": 120.4}), curves=curves)  # rounds up to 121, which is in the small band
    assert r["prices_per_mtok"]["input"] == pytest.approx(0.2) and "small-b" in r["basis"]
    assert r["band"] == "small" and r["band_up"]["band"] == "mid"
    low = c(profile(model={"min_eci": 100}), curves=curves)  # below the lookup: ECI >= 120 price, floor band
    assert low["prices_per_mtok"]["input"] == pytest.approx(0.1) and "upper bound" in low["basis"]
    assert low["band"] == "floor" and low["band_up"]["band"] == "small"
    hist = c(profile(model={"min_eci": 122}), "2026-10", "slow", curves)
    assert hist["prices_per_mtok"]["input"] == pytest.approx(2.0) and hist["period"] == "history"


def test_edge_warnings():
    b = load_bands()
    assert b.edge_warning_points == 2
    edges = [x.min_eci for x in b.bands]
    for e in edges:
        for d in (-2, -0.5, 0, 1.5, 2):
            assert b.edge_warnings(e + d), (e, d)
    mid = (edges[1] + edges[2]) / 2  # half-way between two edges that are > 4 points apart
    assert not b.edge_warnings(mid) and not b.edge_warnings(edges[1] + 2.5)
    with pytest.warns(EdgeWarning, match="within 2 ECI points"):
        parse_profile({"name": "x", "description": "x", "tokens_per_task": {"input": 1, "output": 1},
                       "model": {"min_eci": edges[2] - 1}})


def test_edge_warning_in_cost_output(curves):
    r = c(profile(model={"min_eci": 121.2}), curves=curves)  # BANDS warns within 0.5 points
    assert len(r["warnings"]) == 1 and "'small' band edge" in r["warnings"][0]
    assert c(profile(model={"band": "small"}), curves=curves)["warnings"] == []


def test_bands_config_validation():
    with pytest.raises(ValueError, match="increase"):
        Bands.from_dict({"bands": [{"name": "a", "min_eci": 130}, {"name": "b", "min_eci": 120}]})
    with pytest.raises(ValueError, match="unique"):
        Bands.from_dict({"bands": [{"name": "a", "min_eci": 120}, {"name": "a", "min_eci": 130}]})
    assert [x.name for x in load_bands().bands] == ["floor", "small", "mid", "frontier"]


@pytest.mark.parametrize("kw, month, scenario, msg", [
    ({"model": {"tier": "nope"}}, "2026-11", "base", "unknown tier"),
    ({}, "2028-01", "base", "past the curve"),
    ({}, "2026-11", "wild", "scenario"),
    ({}, "2025-01", "base", "no tier mid price"),
    ({"model": {"min_eci": 130}}, "2026-11", "base", "no priced model"),
])
def test_cost_errors(curves, kw, month, scenario, msg):
    with pytest.raises(CostError, match=msg):
        c(profile(**kw), month, scenario, curves)


# ---------------------------------------------------------------- profile validation


@pytest.mark.parametrize("change, msg", [
    ({"model": {"tier": "mid", "min_eci": 140}}, "exactly one"),
    ({"model": {"band": "mid", "tier": "mid"}}, "exactly one"),
    ({"model": {}}, "exactly one"),
    ({"model": {"band": "huge"}}, "unknown band"),
    ({"tokens_per_task": {"input": {"low": 5, "central": 3, "high": 9}, "output": 1}}, "low <= central <= high"),
    ({"tokens_per_task": {"input": -1, "output": 1}}, ">= 0"),
    ({"tokens_per_task": {"input": 1}}, "output is required"),
    ({"tokens_per_task": {"input": 1, "output": 1, "cached": 5}}, "unknown token types"),
    ({"tokens_per_task": {"input": {"low": 1, "central": 2}, "output": 1}}, "missing"),
    ({"effort": "extreme"}, "effort"),
    ({"runs": 5}, "unknown keys"),
    ({"name": ""}, "'name' is required"),
])
def test_profile_validation(change, msg):
    with pytest.raises(ProfileError, match=msg):
        profile(**change)


def test_scalar_volumes_and_defaults():
    p = profile(tokens_per_task={"input": 100, "output": 20})
    assert p.tokens["input"].low == p.tokens["input"].high == 100
    assert all(p.tokens[t].central == 0 for t in ("reasoning", "cache_read", "cache_write"))
    assert p.effort == "none" and not p.example


# ---------------------------------------------------------------- the real curve_v4 files


@pytest.fixture(scope="module")
def v4() -> PriceCurves:
    return PriceCurves.load(ROOT / "data" / "curves")


@pytest.fixture(scope="module")
def example():
    return load_profile(ROOT / "workflows" / "profiles" / "example_placeholder.yaml")


def test_example_profile_against_v4(v4, example):
    assert example.example, "the placeholder profile must stay marked as an example"
    assert example.band == "mid"
    edge = load_bands().get("mid").min_eci
    row = v4.capability[(v4.capability.eci_min == edge) & (v4.capability.months_ahead == 0) & (v4.capability.scenario == "base")].iloc[0]
    r = cost(example, v4.anchor, "base", v4)
    price = {"input": row.input, "output": row.output, "reasoning": row.output, "cache_read": row.cache_read, "cache_write": row.cache_write}
    assert r["cost_per_task"]["central"] == pytest.approx(sum(example.tokens[t].central * price[t] for t in price) / 1e6)
    assert r["band_up"]["band"] == "frontier"


def test_v4_tier_prices_match_json(v4, example):
    base = json.loads((ROOT / "data" / "curves" / "curve_v4.json").read_text())["tiers"]["frontier"]["base"]
    r = cost(dataclasses.replace(example, band=None, tier="frontier"), v4.anchor, "base", v4)
    assert r["prices_per_mtok"] == pytest.approx({t: base[t] for t in ("input", "output", "cache_read", "cache_write")})


def test_v4_scenarios_are_ordered(v4, example):
    for tier in v4.tiers:
        q = dataclasses.replace(example, band=None, tier=tier)
        end = {sc: cost(q, v4.end, sc, v4)["cost_per_task"]["central"] for sc in ("base", "fast", "slow")}
        today = cost(q, v4.anchor, "slow", v4)["cost_per_task"]["central"]
        assert end["fast"] <= end["base"] <= end["slow"] == pytest.approx(today), tier


def test_v4_capability_is_monotone_today(v4):
    """Requiring more capability never makes the cheapest blended price lower."""
    t = v4.capability[(v4.capability.months_ahead == 0) & (v4.capability.scenario == "base")].sort_values("eci_min")
    assert t.blended.is_monotonic_increasing


def test_v4_cache_write_column(v4):
    for df in (v4.curve, v4.history, v4.capability):
        assert df.cache_write.notna().all()


def test_all_shipped_profiles_load():
    assert load_profiles(ROOT / "workflows" / "profiles")
