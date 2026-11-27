"""Tests for the workflow cost function: exact arithmetic on a small synthetic curve, then checks on curve_v4."""

from __future__ import annotations

import dataclasses
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

from workflows.bands import Bands, load_bands
from workflows.cost import CostError, PriceCurves, cost
from workflows.profile import EdgeWarning, ProfileError, load_profile, load_profiles, parse_profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_curve_v4 import median_price_set  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = pd.Timestamp("2026-11-01")
# synthetic bands matching the synthetic capability lookup below (thresholds 120-123)
BANDS = Bands.from_dict({"edge_warning_points": 0.5, "bands": [
    {"name": "floor", "min_eci": 120, "median_tier": None}, {"name": "small", "min_eci": 121, "median_tier": "small"},
    {"name": "mid", "min_eci": 122, "median_tier": "mid"}, {"name": "frontier", "min_eci": 123, "median_tier": "frontier"}]})
TIER_SCALE = {"small": 0.5, "mid": 1.0, "frontier": 2.0, "eci>=120": 0.05}  # times the mid tier's prices
CAP_MODELS = {120: ("small-a", 0.1), 121: ("small-b", 0.2), 122: ("mid-a", 1.0), 123: ("top", 5.0)}


def months(n0, n1):
    return [ANCHOR + pd.DateOffset(months=k) for k in range(n0, n1 + 1)]


def tables(with_cache_write: bool = True):
    """Tier 'mid': input $2, output $10, cache read $0.5, cache write $2.5 today; base halves every year, slow is flat.
    Tiers small, frontier and eci>=120: the same times TIER_SCALE.
    Capability lookup: input price per threshold from CAP_MODELS, output 4x, cache read 1/10, cache write 1.25x."""
    rows = []
    for k, m in zip(range(0, 13), months(0, 12)):
        for sc, f in (("base", 0.5), ("fast", 0.25), ("slow", 1.0)):
            for tier, x in TIER_SCALE.items():
                g = x * f ** (k / 12)
                rows.append({"month": m, "months_ahead": k, "tier": tier, "scenario": sc,
                             "input": 2 * g, "output": 10 * g, "cache_read": 0.5 * g, "cache_write": 2.5 * g})
    hist = pd.DataFrame([{"month": ANCHOR - pd.DateOffset(months=1), "tier": t, "input": 3.0 * x, "output": 12.0 * x,
                          "cache_read": 3.0 * x, "cache_write": 3.0 * x} for t, x in TIER_SCALE.items()])
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


def only(r):
    """The one price view of a tier profile's result."""
    (v,) = [v for v in r["views"].values() if v is not None]
    return v


# ---------------------------------------------------------------- arithmetic (tier profile: one view)


def test_cost_per_task_today(curves):
    r = c(profile(), curves=curves)
    v = only(r)
    assert v["view"] == "median" and r["views"]["cheapest"] is None  # tier mid is a median tier
    # central: 2000*2 + 500*10 + 1000*10 (reasoning at output price) + 10000*0.5 = 24000 per 1M tokens
    assert v["cost_per_task"]["central"] == pytest.approx(0.024)
    assert v["cost_per_task"]["low"] == pytest.approx((1000 * 2 + 500 * 10) / 1e6)
    assert v["cost_per_task"]["high"] == pytest.approx((4000 * 2 + 500 * 10 + 3000 * 10 + 10000 * 0.5) / 1e6)
    assert v["monthly_cost"] is None
    assert r["period"] == "today"


def test_cheapest_index_tier_gives_cheapest_view(curves):
    r = c(profile(model={"tier": "eci>=120"}), curves=curves)
    assert r["views"]["median"] is None and r["views"]["cheapest"]["price_case"] == "low"
    assert r["views"]["cheapest"]["cost_per_task"]["central"] == pytest.approx(0.024 * 0.05)


def test_reasoning_billed_at_output_price(curves):
    base = profile(tokens_per_task={"input": 0, "output": 0})
    more = profile(tokens_per_task={"input": 0, "output": 0, "reasoning": 1_000_000})
    assert only(c(base, curves=curves))["cost_per_task"]["central"] == 0
    assert only(c(more, curves=curves))["cost_per_task"]["central"] == pytest.approx(10.0)


def test_cache_write_priced_separately(curves):
    p = profile(tokens_per_task={"input": 0, "output": 0, "cache_write": {"low": 0, "central": 1_000_000, "high": 2_000_000}})
    v = only(c(p, curves=curves))
    assert v["prices_per_mtok"]["cache_write"] == pytest.approx(2.5)
    assert v["cost_per_task"] == pytest.approx({"low": 0.0, "central": 2.5, "high": 5.0})
    assert only(c(p, month="2027-11", curves=curves))["cost_per_task"]["central"] == pytest.approx(1.25)  # base halves
    assert only(c(p, month="2026-10", curves=curves))["cost_per_task"]["central"] == pytest.approx(3.0)   # history


def test_cache_write_defaults_to_input_price_when_curve_has_none():
    old = PriceCurves(*tables(with_cache_write=False))
    p = profile(tokens_per_task={"input": 0, "output": 0, "cache_write": 1_000_000})
    assert only(c(p, curves=old))["cost_per_task"]["central"] == pytest.approx(2.0)
    assert profile().tokens["cache_write"].central == 0  # optional in the schema


def test_scenarios_and_months(curves):
    p = profile()
    central = lambda *a: only(c(p, *a, curves=curves))["cost_per_task"]["central"]
    today = central("2026-11", "base")
    assert central("2027-11", "base") == pytest.approx(today * 0.5)
    assert central("2027-11", "fast") == pytest.approx(today * 0.25)
    assert central("2027-11", "slow") == pytest.approx(today)
    assert c(p, "2027-11-17", "base", curves)["month"] == "2027-11"
    assert central(pd.Timestamp("2027-11-30"), "base") == pytest.approx(today * 0.5)


def test_monthly_cost_pairs_levels(curves):
    v = only(c(profile(runs_per_month={"low": 10, "central": 100, "high": 1000}), curves=curves))
    for lv, n in {"low": 10, "central": 100, "high": 1000}.items():
        assert v["monthly_cost"][lv] == pytest.approx(v["cost_per_task"][lv] * n)
    flat = only(c(profile(runs_per_month=250), curves=curves))
    assert flat["monthly_cost"]["central"] == pytest.approx(flat["cost_per_task"]["central"] * 250)


def test_history_before_anchor_ignores_scenario(curves):
    p = profile()
    a, b = c(p, "2026-10", "base", curves), c(p, "2026-10", "fast", curves)
    assert a["period"] == "history" and only(a)["cost_per_task"] == only(b)["cost_per_task"]
    assert only(a)["prices_per_mtok"] == {"input": 3.0, "output": 12.0, "cache_read": 3.0, "cache_write": 3.0}


# ---------------------------------------------------------------- two price views, bands, band-up, exact ECI


def test_band_has_both_price_views(curves):
    r = c(profile(model={"band": "small"}), curves=curves)
    lo, mid = r["views"]["cheapest"], r["views"]["median"]
    assert r["band"] == "small"
    assert (lo["price_case"], lo["label"]) == ("low", "cheapest model meeting the band")
    assert (mid["price_case"], mid["label"]) == ("central", "tier median model")
    assert lo["prices_per_mtok"]["input"] == pytest.approx(0.2) and "small-b" in lo["basis"]  # capability lookup at 121
    assert mid["prices_per_mtok"]["input"] == pytest.approx(1.0) and "tier small" in mid["basis"]  # 0.5 x mid tier
    # both views apply the same token volumes to their own prices
    p = profile(model={"band": "small"})
    of = {"input": "input", "output": "output", "reasoning": "output", "cache_read": "cache_read", "cache_write": "cache_write"}
    for v in (lo, mid):
        for lv in ("low", "central", "high"):
            want = sum(p.tokens[t][lv] * v["prices_per_mtok"][of[t]] for t in of) / 1e6
            assert v["cost_per_task"][lv] == pytest.approx(want)


def test_floor_band_has_no_median_view(curves):
    r = c(profile(model={"band": "floor"}), curves=curves)
    assert r["views"]["median"] is None and r["views"]["cheapest"] is not None
    assert "no median tier" in r["notes"][0]


def test_band_up_sensitivity(curves):
    p = profile(model={"band": "small"}, runs_per_month=100)
    r = c(p, "2027-11", "fast", curves)
    up = r["band_up"]
    assert up["band"] == "mid"
    lo, lo_up = r["views"]["cheapest"], up["views"]["cheapest"]
    assert "mid-a" in lo_up["basis"]
    for lv in ("low", "central", "high"):  # cheapest: input 1.0 vs 0.2; median: mid tier vs 0.5 x mid tier
        assert lo_up["cost_per_task"][lv] == pytest.approx(5 * lo["cost_per_task"][lv])
        assert lo_up["monthly_cost"][lv] == pytest.approx(5 * lo["monthly_cost"][lv])
        assert up["views"]["median"]["cost_per_task"][lv] == pytest.approx(2 * r["views"]["median"]["cost_per_task"][lv])
    assert c(profile(model={"band": "frontier"}), curves=curves)["band_up"] is None  # no band above the top
    assert c(profile(), curves=curves)["band_up"] is None                            # tiers have no band-up
    floor_up = c(profile(model={"band": "floor"}), curves=curves)["band_up"]
    assert floor_up["band"] == "small" and floor_up["views"]["median"] is not None


def test_min_eci_lookup_and_views(curves):
    r = c(profile(model={"min_eci": 120.4}), curves=curves)  # rounds up to 121, which is in the small band
    assert r["views"]["cheapest"]["prices_per_mtok"]["input"] == pytest.approx(0.2) and "small-b" in r["views"]["cheapest"]["basis"]
    assert r["band"] == "small" and r["band_up"]["band"] == "mid"
    assert "tier small" in r["views"]["median"]["basis"]  # the median view comes from the band the threshold is in
    low = c(profile(model={"min_eci": 100}), curves=curves)  # below the lookup: ECI >= 120 price, floor band
    assert low["views"]["cheapest"]["prices_per_mtok"]["input"] == pytest.approx(0.1)
    assert "upper bound" in low["views"]["cheapest"]["basis"]
    assert low["band"] == "floor" and low["views"]["median"] is None and low["band_up"]["band"] == "small"
    hist = c(profile(model={"min_eci": 122}), "2026-10", "slow", curves)
    assert hist["views"]["cheapest"]["prices_per_mtok"]["input"] == pytest.approx(2.0) and hist["period"] == "history"
    assert hist["views"]["median"]["prices_per_mtok"]["input"] == pytest.approx(3.0)


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


def test_bands_config():
    with pytest.raises(ValueError, match="increase"):
        Bands.from_dict({"bands": [{"name": "a", "min_eci": 130}, {"name": "b", "min_eci": 120}]})
    with pytest.raises(ValueError, match="unique"):
        Bands.from_dict({"bands": [{"name": "a", "min_eci": 120}, {"name": "a", "min_eci": 130}]})
    b = load_bands()
    assert [x.name for x in b.bands] == ["floor", "small", "mid", "frontier"]
    assert [x.median_tier for x in b.bands] == [None, "small", "mid", "frontier"]


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


# ---------------------------------------------------------------- median-model price set (curve_v4 tiers)


def _models(*rows):
    return pd.DataFrame([{"canonical_id": n, "blended": 0.75 * i + 0.25 * o, "input": i, "output": o,
                          "cache_read": cr, "cache_write": cw} for n, i, o, cr, cw in rows])


def test_median_price_set_odd_count_is_one_model():
    g = _models(("a", 1, 4, 0.1, 1), ("b", 2, 10, 0.2, 2.5), ("c", 5, 25, 0.5, 6))
    r = median_price_set(g)
    assert (r["input"], r["output"], r["cache_read"], r["cache_write"]) == (2, 10, 0.2, 2.5) and r["median_models"] == "b"


def test_median_price_set_even_count_averages_middle_two():
    g = _models(("a", 1, 4, 0.1, 1), ("b", 2, 10, 0.2, 2.5), ("c", 4, 20, 0.4, 5), ("d", 9, 40, 1, 9))
    r = median_price_set(g)
    assert r["input"] == pytest.approx(3) and r["cache_read"] == pytest.approx(0.3) and r["median_models"] == "b; c"
    assert 0.75 * r["input"] + 0.25 * r["output"] == pytest.approx(g.blended.median())


def test_median_price_set_averages_ties():
    # b and c have the same blended price but different cache prices; the result must not depend on row order
    g = _models(("a", 1, 4, 0.1, 1), ("b", 2, 10, 0.2, 2.5), ("c", 2, 10, 0.6, 2.0), ("d", 9, 40, 1, 9), ("e", 9, 41, 1, 9))
    r1, r2 = median_price_set(g), median_price_set(g.iloc[::-1])
    assert r1 == r2 and r1["cache_read"] == pytest.approx(0.4) and r1["median_models"] == "b; c"


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
    cap = v4.capability
    row = cap[(cap.eci_min == edge) & (cap.months_ahead == 0) & (cap.scenario == "base")].iloc[0]
    base = json.loads((ROOT / "data" / "curves" / "curve_v4.json").read_text())["tiers"]["mid"]["base"]
    r = cost(example, v4.anchor, "base", v4)
    for view, pr in (("cheapest", row), ("median", base)):
        price = {"input": pr["input"], "output": pr["output"], "reasoning": pr["output"],
                 "cache_read": pr["cache_read"], "cache_write": pr["cache_write"]}
        want = sum(example.tokens[t].central * price[t] for t in price) / 1e6
        assert r["views"][view]["cost_per_task"]["central"] == pytest.approx(want), view
    assert r["band_up"]["band"] == "frontier"


def test_v4_median_tiers_use_one_price_set(v4):
    """small / mid / frontier: every token type from the median model(s), so the blended price is v3's tier index."""
    h = pd.read_csv(ROOT / "data" / "curves" / "tier_history_v4.csv")
    g = h[h.tier.isin(["small", "mid", "frontier"]) & h.used]
    assert set(g.tier) == {"small", "mid", "frontier"} and g.median_models.notna().all()
    assert (g.blended - g.blended_v3).abs().max() < 1e-9
    js = json.loads((ROOT / "data" / "curves" / "curve_v4.json").read_text())["tiers"]
    for t in ("small", "mid", "frontier"):
        b = js[t]["base"]
        row = g[(g.tier == t) & (g.month.str.startswith(b["base_month"]))].iloc[0]
        assert all(b[x] == pytest.approx(row[x]) for x in ("input", "output", "cache_read", "cache_write")), t
        assert b["blended"] == pytest.approx(b["v3_base_price"]), t


def test_v4_median_view_not_below_cheapest(v4):
    for band in load_bands().bands:
        if not band.median_tier:
            continue
        p = parse_profile({"name": "x", "description": "x", "tokens_per_task": {"input": 1000, "output": 1000},
                           "model": {"band": band.name}})
        for m in (v4.anchor, v4.end):
            for sc in ("base", "fast", "slow"):
                r = cost(p, m, sc, v4)
                assert (r["views"]["median"]["cost_per_task"]["central"]
                        >= r["views"]["cheapest"]["cost_per_task"]["central"]), (band.name, m, sc)


def test_v4_scenarios_are_ordered(v4, example):
    for tier in v4.tiers:
        q = dataclasses.replace(example, band=None, tier=tier)
        end = {sc: only(cost(q, v4.end, sc, v4))["cost_per_task"]["central"] for sc in ("base", "fast", "slow")}
        today = only(cost(q, v4.anchor, "slow", v4))["cost_per_task"]["central"]
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
