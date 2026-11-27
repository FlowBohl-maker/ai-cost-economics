"""Deterministic inputs, prompts and quality checks for the tokens-per-task workflows.

Everything is synthetic (fixed seed) except the earnings-call transcript, which is a public SEC filing
read from .cache/transcripts/ (git-ignored, not redistributed). Each workflow has a known answer or a
fact list, so every run gets a simple automated quality score in [0, 1].
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPT = ROOT / ".cache" / "transcripts" / "clir_q3_2026.txt"
TRANSCRIPT_SOURCE = ("ClearSign Technologies Corp, Q3 2026 earnings call transcript, SEC Form 8-K Exhibit 99.2, filed "
                     "2026-11-20: https://www.sec.gov/Archives/edgar/data/1434524/000110465926130612/tm2630382d1_ex99-2.htm")
SEED = 20261126

# ---------------------------------------------------------------- 1. data Q&A

QA_COMPANIES = ["Aldermoor Foods", "Brightwater Logistics", "Castellan Software", "Dunmore Energy",
                "Eastvale Medical", "Fernhill Retail", "Glenrock Materials", "Hartwell Media"]
QA_QUARTERS = ["2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2", "2026Q3"]


def qa_table() -> list[dict]:
    rng = np.random.default_rng(SEED)
    rows = []
    for c in QA_COMPANIES:
        rev = rng.uniform(300, 2500)
        growth = rng.uniform(-0.02, 0.06)
        gm = rng.uniform(0.25, 0.65)
        om = gm - rng.uniform(0.08, 0.25)
        for q in QA_QUARTERS:
            rev *= 1 + growth + rng.normal(0, 0.03)
            g = gm + rng.normal(0, 0.01)
            o = om + rng.normal(0, 0.015)
            rows.append({"company": c, "quarter": q, "revenue_musd": round(rev, 1), "gross_margin_pct": round(100 * g, 1),
                         "operating_income_musd": round(rev * o, 1), "net_income_musd": round(rev * o * 0.76, 1),
                         "free_cash_flow_musd": round(rev * o * rng.uniform(0.5, 1.1), 1),
                         "capex_musd": round(rev * rng.uniform(0.03, 0.12), 1),
                         "net_debt_musd": round(rng.uniform(-200, 1500), 1)})
    return rows


def qa_answer() -> dict:
    rows = qa_table()
    out = {}
    for c in QA_COMPANIES:
        r = {x["quarter"]: x for x in rows if x["company"] == c}
        g = 100 * (r["2026Q3"]["revenue_musd"] / r["2025Q3"]["revenue_musd"] - 1)
        out[c] = {"growth": g, "op_margin": 100 * r["2026Q3"]["operating_income_musd"] / r["2026Q3"]["revenue_musd"]}
    best = max(out, key=lambda c: out[c]["growth"])
    return {"company": best, **out[best]}


def qa_prompt() -> list[dict]:
    rows = qa_table()
    cols = list(rows[0])
    table = "\n".join([",".join(cols)] + [",".join(str(r[c]) for c in cols) for r in rows])
    q = ("Using only the table below (quarterly results, USD millions), which company had the highest year-over-year "
         "revenue growth in 2026Q3 (versus 2025Q3), and what was its operating margin in 2026Q3? Explain briefly, then "
         "end with one line in exactly this format:\nANSWER: <company>; growth=<x.x>%; op_margin=<y.y>%")
    return [{"role": "system", "content": "You are a careful financial data analyst."},
            {"role": "user", "content": f"{q}\n\n```csv\n{table}\n```"}]


def qa_score(text: str) -> float:
    a = qa_answer()
    m = re.search(r"ANSWER:\s*(.+?);\s*growth\s*=\s*(-?[\d.]+)%?;\s*op_margin\s*=\s*(-?[\d.]+)", text or "", re.I)
    if not m:
        return 0.0
    s = 0.5 * (a["company"].lower() in m.group(1).lower())
    s += 0.25 * (abs(float(m.group(2)) - a["growth"]) <= 0.15)
    s += 0.25 * (abs(float(m.group(3)) - a["op_margin"]) <= 0.15)
    return s


# ---------------------------------------------------------------- 2. screening

SECTORS = ["Technology", "Healthcare", "Financials", "Industrials", "Consumer Discretionary", "Consumer Staples",
           "Energy", "Materials", "Utilities", "Real Estate", "Communication Services"]
REGIONS = ["US", "UK", "Germany", "France", "Japan", "Canada", "Switzerland", "Netherlands", "Australia"]
SCREEN_CRITERIA = ("1. Market capitalisation of at least USD 5 bn.\n"
                   "2. Revenue growth (last fiscal year) of at least 10%.\n"
                   "3. Net debt / EBITDA of at most 2.0x (net cash counts as negative).\n"
                   "4. Return on equity of at least 15%.")


def screen_table() -> list[dict]:
    rng = np.random.default_rng(SEED + 1)
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    used, rows = set(), []
    while len(rows) < 200:
        t = "".join(rng.choice(letters, 4))
        if t in used:
            continue
        used.add(t)
        rows.append({"ticker": t, "sector": str(rng.choice(SECTORS)), "country": str(rng.choice(REGIONS)),
                     "mkt_cap_busd": round(float(np.exp(rng.normal(1.6, 1.2))), 1),
                     "pe": round(float(rng.uniform(6, 45)), 1),
                     "rev_growth_pct": round(float(rng.normal(7, 9)), 1),
                     "net_debt_ebitda": round(float(rng.normal(1.6, 1.4)), 1),
                     "roe_pct": round(float(rng.normal(13, 7)), 1),
                     "div_yield_pct": round(float(max(0, rng.normal(2.2, 1.5))), 1)})
    return rows


def screen_answer() -> list[str]:
    return sorted(r["ticker"] for r in screen_table() if r["mkt_cap_busd"] >= 5 and r["rev_growth_pct"] >= 10
                  and r["net_debt_ebitda"] <= 2.0 and r["roe_pct"] >= 15)


def screen_prompt() -> list[dict]:
    rows = screen_table()
    cols = list(rows[0])
    table = "\n".join([",".join(cols)] + [",".join(str(r[c]) for c in cols) for r in rows])
    q = (f"Screen the 200 companies below. A company passes only if it meets ALL of these criteria:\n{SCREEN_CRITERIA}\n\n"
         "List every company that passes with a one-line explanation per company, then add two or three sentences on "
         "what the shortlist has in common. End with one line in exactly this format:\n"
         "SHORTLIST: <TICKER>, <TICKER>, ...")
    return [{"role": "system", "content": "You are an equity research analyst. Be exact when applying numeric thresholds."},
            {"role": "user", "content": f"{q}\n\n```csv\n{table}\n```"}]


def screen_score(text: str) -> float:
    m = re.search(r"SHORTLIST:\s*(.+)", text or "")
    if not m:
        return 0.0
    got = set(re.findall(r"\b[A-Z]{4}\b", m.group(1)))
    want = set(screen_answer())
    if not got:
        return 0.0
    p, r = len(got & want) / len(got), len(got & want) / len(want)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)  # F1


# ---------------------------------------------------------------- 3. earnings summary

# facts a good summary should carry (regex, description); checked case-insensitively
EARNINGS_FACTS = [
    (r"712", "Q3 revenue of about $712k"),
    (r"55\.6", "gross margin 55.6%"),
    (r"10\.3", "cash of about $10.3m"),
    (r"\b26[- ]burner|26 burners", "26-burner Gulf Coast order starting up"),
    (r"\b32[- ]burner|32 burners", "32-burner California order"),
    (r"licens", "licensing arrangement with an OEM heater manufacturer"),
]


def transcript_text() -> str:
    """The transcript as plain text; converts the raw SEC exhibit (.htm, saved by the user) on first use."""
    if not TRANSCRIPT.exists():
        htm = TRANSCRIPT.with_suffix(".htm")
        if not htm.exists():
            raise SystemExit(f"transcript missing: save the exhibit as {htm} (see README)")
        from html.parser import HTMLParser

        class P(HTMLParser):
            def __init__(self):
                super().__init__()
                self.out, self.buf = [], []

            def handle_starttag(self, tag, attrs):
                if tag in ("p", "div", "br", "tr"):
                    self.flush()

            def handle_endtag(self, tag):
                if tag in ("p", "div", "tr"):
                    self.flush()

            def handle_data(self, d):
                self.buf.append(d)

            def flush(self):
                s = re.sub(r"\s+", " ", "".join(self.buf)).strip()
                if s:
                    self.out.append(s)
                self.buf = []

        p = P()
        p.feed(htm.read_text(errors="replace"))
        p.flush()
        TRANSCRIPT.write_text("\n\n".join(p.out))
    return TRANSCRIPT.read_text()


def earnings_prompt() -> list[dict]:
    text = transcript_text()
    q = ("Summarise the earnings call transcript below for a portfolio manager. Use three sections with these "
         "headings: 'Key points' (5-8 bullets with the important numbers), 'Guidance and outlook', and 'Risks'. "
         "Only use information from the transcript.")
    return [{"role": "system", "content": "You are a sell-side analyst writing concise, factual notes."},
            {"role": "user", "content": f"{q}\n\n<transcript>\n{text}\n</transcript>"}]


def earnings_score(text: str) -> float:
    t = text or ""
    heads = all(re.search(h, t, re.I) for h in (r"key points", r"guidance", r"risk"))
    facts = sum(bool(re.search(rx, t, re.I)) for rx, _ in EARNINGS_FACTS) / len(EARNINGS_FACTS)
    return round(0.8 * facts + 0.2 * heads, 3)


# ---------------------------------------------------------------- 4. portfolio review agent

NEWS_TEMPLATES = {
    "neg": ["{n} cut full-year revenue guidance by 8%, citing weaker orders in its largest segment.",
            "{n} disclosed a regulatory investigation into its sales practices; the company says it is cooperating.",
            "Two brokers downgraded {n} to underweight after a profit warning; shares fell 14% on the day.",
            "{n} delayed its debt refinancing; its bonds trade at a widening spread."],
    "pos": ["{n} raised its dividend by 10% and announced a share buyback.",
            "{n} won a multi-year contract with a large public-sector customer.",
            "{n} reported quarterly earnings 6% above consensus."],
    "neutral": ["{n} appointed a new chief financial officer from a peer company.",
                "{n} will hold its capital markets day next month."],
}


def portfolio() -> dict:
    rng = np.random.default_rng(SEED + 2)
    names = ["Arcadia Systems", "Beacon Health", "Cobalt Mining", "Delphi Bank", "Ember Utilities", "Falcon Aero",
             "Granite Homes", "Harbor Shipping", "Ion Semiconductors", "Juniper Foods", "Keystone Insurance",
             "Lumen Telecom", "Meridian Pharma", "Northstar Oil", "Orchid Retail", "Pinnacle Software", "Quarry Cement",
             "Ridge Biotech", "Summit Chemicals", "Tidal Renewables", "Umber Media", "Vertex Robotics", "Willow REIT",
             "Xenon Cloud", "Yardley Autos", "Zephyr Airlines", "Atlas Payments", "Boreal Timber", "Crescent Gaming",
             "Dune Logistics"]
    sectors = ["Technology", "Healthcare", "Materials", "Financials", "Utilities", "Industrials", "Real Estate",
               "Industrials", "Technology", "Consumer Staples", "Financials", "Communication Services", "Healthcare",
               "Energy", "Consumer Discretionary", "Technology", "Materials", "Healthcare", "Materials", "Utilities",
               "Communication Services", "Technology", "Real Estate", "Technology", "Consumer Discretionary",
               "Industrials", "Technology", "Materials", "Consumer Discretionary", "Industrials"]
    w = rng.dirichlet(np.full(30, 2.0))
    holdings, risk = [], {}
    for i, (n, s) in enumerate(zip(names, sectors)):
        t = "".join(x[0] for x in n.split()) + n.split()[0][1:3].upper()
        reg = str(rng.choice(["US", "US", "US", "Europe", "Europe", "Japan", "Emerging"]))
        holdings.append({"ticker": t, "name": n, "sector": s, "region": reg, "weight_pct": round(100 * float(w[i]), 2)})
        risk[t] = {"beta": round(float(rng.uniform(0.5, 1.8)), 2), "vol_1y_pct": round(float(rng.uniform(14, 55)), 1),
                   "max_drawdown_1y_pct": round(-float(rng.uniform(8, 48)), 1),
                   "return_1m_pct": round(float(rng.normal(0.5, 6)), 1),
                   "liquidity_days_to_exit": round(float(rng.uniform(0.2, 6)), 1)}
    tickers = [h["ticker"] for h in holdings]
    news = {}
    for t, h in zip(tickers, holdings):
        kind = rng.choice(["neg", "pos", "neutral", "none", "none"], p=[0.2, 0.2, 0.15, 0.25, 0.2])
        if kind != "none":
            news[t] = [{"date": f"2026-11-{int(rng.integers(2, 25)):02d}",
                        "headline": str(rng.choice(NEWS_TEMPLATES[kind])).format(n=h["name"]), "tone": str(kind)}]
    return {"holdings": holdings, "risk": risk, "news": news}


def exposures(dimension: str) -> dict:
    p = portfolio()
    out: dict[str, float] = {}
    for h in p["holdings"]:
        k = h.get(dimension)
        if k is None:
            return {"error": f"unknown dimension '{dimension}'; use 'sector' or 'region'"}
        out[k] = round(out.get(k, 0) + h["weight_pct"], 2)
    return {"dimension": dimension, "weights_pct": dict(sorted(out.items(), key=lambda kv: -kv[1])),
            "benchmark_note": "Benchmark sector weights: no sector above 25%."}


AGENT_TOOLS = [
    {"type": "function", "function": {"name": "get_portfolio", "description": "Return the current holdings with weights.",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "get_risk_metrics", "description": "Risk metrics for the given tickers.",
                                      "parameters": {"type": "object", "properties": {"tickers": {"type": "array", "items": {"type": "string"}}},
                                                     "required": ["tickers"]}}},
    {"type": "function", "function": {"name": "get_news", "description": "Recent news headlines for the given tickers.",
                                      "parameters": {"type": "object", "properties": {"tickers": {"type": "array", "items": {"type": "string"}}},
                                                     "required": ["tickers"]}}},
    {"type": "function", "function": {"name": "get_exposures", "description": "Portfolio weights aggregated by sector or region.",
                                      "parameters": {"type": "object", "properties": {"dimension": {"type": "string", "enum": ["sector", "region"]}},
                                                     "required": ["dimension"]}}},
]


def call_tool(name: str, args_json: str) -> str:
    """Fixed, repeatable tool responses."""
    p = portfolio()
    try:
        args = json.loads(args_json or "{}")
    except json.JSONDecodeError:
        return json.dumps({"error": "arguments were not valid JSON"})
    if name == "get_portfolio":
        return json.dumps({"as_of": "2026-11-25", "base_currency": "USD", "holdings": p["holdings"]})
    if name == "get_risk_metrics":
        ts = args.get("tickers") or []
        return json.dumps({t: p["risk"].get(t, {"error": "unknown ticker"}) for t in ts})
    if name == "get_news":
        ts = args.get("tickers") or []
        return json.dumps({t: p["news"].get(t, []) for t in ts})
    if name == "get_exposures":
        return json.dumps(exposures(args.get("dimension", "sector")))
    return json.dumps({"error": f"unknown tool {name}"})


def agent_prompt() -> list[dict]:
    sys_ = ("You are a portfolio risk analyst with tools. Work step by step and wait for each tool result before "
            "deciding the next step: (1) load the portfolio; (2) get risk metrics for all holdings; (3) get news for "
            "the eight holdings with the highest 1-year volatility; (4) get exposures by sector and by region. "
            "Then write the final risk commentary.")
    user = ("Please review our portfolio and write a risk commentary for the investment committee (300-500 words): "
            "the main concentration risks, the riskiest holdings and why, any news-driven concerns, and two or three "
            "concrete actions to consider.")
    return [{"role": "system", "content": sys_}, {"role": "user", "content": user}]


def agent_truth() -> dict:
    p = portfolio()
    top_vol = sorted(p["risk"], key=lambda t: -p["risk"][t]["vol_1y_pct"])[:8]
    neg = [t for t in top_vol if any(n["tone"] == "neg" for n in p["news"].get(t, []))]
    sec = exposures("sector")["weights_pct"]
    return {"top_vol": top_vol, "neg_news_in_top_vol": neg, "top_sector": next(iter(sec)), "top_sector_pct": next(iter(sec.values()))}


def agent_score(text: str, n_turns: int, tools_used: set[str]) -> float:
    a, t = agent_truth(), text or ""
    names = {h["ticker"]: h["name"] for h in portfolio()["holdings"]}
    mentioned = lambda k: k in t or names[k] in t
    s_vol = sum(mentioned(k) for k in a["top_vol"][:5]) / 5
    s_neg = (sum(mentioned(k) for k in a["neg_news_in_top_vol"]) / len(a["neg_news_in_top_vol"])) if a["neg_news_in_top_vol"] else 1
    s_sec = float(a["top_sector"].lower() in t.lower())
    s_proc = float(len(tools_used) == 4 and n_turns >= 4)
    return round(0.3 * s_vol + 0.3 * s_neg + 0.2 * s_sec + 0.2 * s_proc, 3)


WORKFLOWS = ["data_qa", "screening", "earnings_summary", "portfolio_agent"]


if __name__ == "__main__":
    print("QA answer:", qa_answer())
    print("screen answer:", screen_answer(), len(screen_answer()))
    print("agent truth:", agent_truth())
    for name, msgs in (("qa", qa_prompt()), ("screen", screen_prompt()), ("agent", agent_prompt())):
        print(name, "chars", sum(len(m["content"]) for m in msgs))
    print("tool output chars:", {n: len(call_tool(n, a)) for n, a in (("get_portfolio", "{}"),
          ("get_risk_metrics", json.dumps({"tickers": [h["ticker"] for h in portfolio()["holdings"]]})),
          ("get_news", json.dumps({"tickers": agent_truth()["top_vol"]})), ("get_exposures", '{"dimension":"sector"}'))})
