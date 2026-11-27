"""Tests for the workflow cost function: exact arithmetic on a small synthetic curve, then checks on curve_v4."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
import pytest

from workflows.cost import CostError, PriceCurves, cost
from workflows.profile import ProfileError, load_profile, load_profiles, parse_profile

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = pd.Timestamp("2026-11-01")


def months(n0, n1):
    return [ANCHOR + pd.DateOffset(months=k) for k in range(n0, n1 + 1)]


@pytest.fixture
def curves() -> PriceCurves:
    """Tier 'mid': input $2, output $10, cache read $0.5 today; base halves every year, slow stays flat."""
    rows = []
    for k, m in zip(range(0, 13), months(0, 12)):
        for sc, f in (("base", 0.5), ("fast", 0.25), ("slow", 1.0)):
            g = f ** (k / 12)
            rows.append({"month": m, "months_ahead": k, "tier": "mid", "scenario": sc,
                         "input": 2 * g, "output": 10 * g, "cache_read": 0.5 * g})
    hist = pd.DataFrame([{"month": ANCHOR - pd.DateOffset(months=1), "tier": "mid", "input": 3.0, "output": 12.0, "cache_read": 3.0}])
    cap = []
    for thr, (model, inp) in {120: ("small-a", 0.1), 121: ("small-b", 0.2), 122: ("big", 1.0)}.items():
        for sc in ("base", "fast", "slow"):
            for k, m in zip(range(0, 13), months(0, 12)):
                cap.append({"month": m, "months_ahead": k, "eci_min": thr, "scenario": sc, "model": model,
                            "input": inp, "output": 4 * inp, "cache_read": inp / 10})
        cap.append({"month": ANCHOR - pd.DateOffset(months=1), "months_ahead": -1, "eci_min": thr, "scenario": "history",
                    "model": model, "input": 2 * inp, "output": 8 * inp, "cache_read": inp})
    return PriceCurves(pd.DataFrame(rows), hist, pd.DataFrame(cap))


def profile(**kw):
    d = {"name": "t", "description": "test", "tokens_per_task": {
        "input": {"low": 1000, "central": 2000, "high": 4000}, "output": 500,
        "reasoning": {"low": 0, "central": 1000, "high": 3000}, "cache_read": {"low": 0, "central": 10000, "high": 10000}},
        "model": {"tier": "mid"}}
    d.update(kw)
    return parse_profile(d)


# ---------------------------------------------------------------- arithmetic


def test_cost_per_task_today(curves):
    c = cost(profile(), "2026-11", "base", curves)
    # central: 2000*2 + 500*10 + 1000*10 (reasoning at output price) + 10000*0.5 = 24000 per 1M tokens
    assert c["cost_per_task"]["central"] == pytest.approx(0.024)
    assert c["cost_per_task"]["low"] == pytest.approx((1000 * 2 + 500 * 10) / 1e6)
    assert c["cost_per_task"]["high"] == pytest.approx((4000 * 2 + 500 * 10 + 3000 * 10 + 10000 * 0.5) / 1e6)
    assert c["monthly_cost"] is None
    assert c["period"] == "today"


def test_reasoning_billed_at_output_price(curves):
    base = profile(tokens_per_task={"input": 0, "output": 0})
    more = profile(tokens_per_task={"input": 0, "output": 0, "reasoning": 1_000_000})
    assert cost(base, "2026-11", "base", curves)["cost_per_task"]["central"] == 0
    assert cost(more, "2026-11", "base", curves)["cost_per_task"]["central"] == pytest.approx(10.0)


def test_scenarios_and_months(curves):
    p = profile()
    today = cost(p, "2026-11", "base", curves)["cost_per_task"]["central"]
    assert cost(p, "2027-11", "base", curves)["cost_per_task"]["central"] == pytest.approx(today * 0.5)
    assert cost(p, "2027-11", "fast", curves)["cost_per_task"]["central"] == pytest.approx(today * 0.25)
    assert cost(p, "2027-11", "slow", curves)["cost_per_task"]["central"] == pytest.approx(today)
    # month given as a date inside the month, or as a Timestamp
    assert cost(p, "2027-11-17", "base", curves)["month"] == "2027-11"
    assert cost(p, pd.Timestamp("2027-11-30"), "base", curves)["cost_per_task"]["central"] == pytest.approx(today * 0.5)


def test_monthly_cost_pairs_levels(curves):
    p = profile(runs_per_month={"low": 10, "central": 100, "high": 1000})
    c = cost(p, "2026-11", "base", curves)
    for lv in ("low", "central", "high"):
        assert c["monthly_cost"][lv] == pytest.approx(c["cost_per_task"][lv] * {"low": 10, "central": 100, "high": 1000}[lv])
    flat = cost(profile(runs_per_month=250), "2026-11", "base", curves)
    assert flat["monthly_cost"]["central"] == pytest.approx(flat["cost_per_task"]["central"] * 250)


def test_history_before_anchor_ignores_scenario(curves):
    p = profile()
    a, b = cost(p, "2026-10", "base", curves), cost(p, "2026-10", "fast", curves)
    assert a["period"] == "history" and a["cost_per_task"] == b["cost_per_task"]
    assert a["prices_per_mtok"] == {"input": 3.0, "output": 12.0, "cache_read": 3.0}


def test_min_eci_lookup(curves):
    c = cost(profile(model={"min_eci": 120.4}), "2026-11", "base", curves)  # rounds up to 121
    assert c["prices_per_mtok"]["input"] == pytest.approx(0.2) and "small-b" in c["basis"]
    low = cost(profile(model={"min_eci": 100}), "2026-11", "base", curves)  # below the lookup: ECI >= 120 price
    assert low["prices_per_mtok"]["input"] == pytest.approx(0.1) and "upper bound" in low["basis"]
    hist = cost(profile(model={"min_eci": 122}), "2026-10", "slow", curves)
    assert hist["prices_per_mtok"]["input"] == pytest.approx(2.0) and hist["period"] == "history"


@pytest.mark.parametrize("kw, month, scenario, msg", [
    ({"model": {"tier": "nope"}}, "2026-11", "base", "unknown tier"),
    ({}, "2028-01", "base", "past the curve"),
    ({}, "2026-11", "wild", "scenario"),
    ({}, "2025-01", "base", "no history price"),
    ({"model": {"min_eci": 130}}, "2026-11", "base", "no priced model"),
])
def test_cost_errors(curves, kw, month, scenario, msg):
    with pytest.raises(CostError, match=msg):
        cost(profile(**kw), month, scenario, curves)


# ---------------------------------------------------------------- profile validation


@pytest.mark.parametrize("change, msg", [
    ({"model": {"tier": "mid", "min_eci": 140}}, "exactly one"),
    ({"model": {}}, "exactly one"),
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
    assert p.tokens["reasoning"].central == 0 and p.tokens["cache_read"].central == 0
    assert p.effort == "none" and not p.example


# ---------------------------------------------------------------- the real curve_v4 files


@pytest.fixture(scope="module")
def v4() -> PriceCurves:
    return PriceCurves.load(ROOT / "data" / "curves")


def test_example_profile_against_v4(v4):
    p = load_profile(ROOT / "workflows" / "profiles" / "example_placeholder.yaml")
    assert p.example, "the placeholder profile must stay marked as an example"
    base = json.loads((ROOT / "data" / "curves" / "curve_v4.json").read_text())["tiers"][p.tier]["base"]
    c = cost(p, v4.anchor, "base", v4)
    want = sum(p.tokens[t].central * base[{"reasoning": "output"}.get(t, t)] for t in p.tokens) / 1e6
    assert c["cost_per_task"]["central"] == pytest.approx(want)


def test_v4_scenarios_are_ordered(v4):
    p = load_profile(ROOT / "workflows" / "profiles" / "example_placeholder.yaml")
    for tier in v4.tiers:
        q = dataclasses.replace(p, tier=tier)
        end = {sc: cost(q, v4.end, sc, v4)["cost_per_task"]["central"] for sc in ("base", "fast", "slow")}
        today = cost(q, v4.anchor, "slow", v4)["cost_per_task"]["central"]
        assert end["fast"] <= end["base"] <= end["slow"] == pytest.approx(today), tier


def test_v4_capability_is_monotone_today(v4):
    """Requiring more capability never makes the cheapest blended price lower."""
    t = v4.capability[(v4.capability.months_ahead == 0) & (v4.capability.scenario == "base")].sort_values("eci_min")
    assert t.blended.is_monotonic_increasing


def test_all_shipped_profiles_load():
    assert load_profiles(ROOT / "workflows" / "profiles")
