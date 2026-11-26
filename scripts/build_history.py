"""Build the historical price dataset from LiteLLM (+ optionally llm-prices.com), map model IDs, attach Epoch ECI.

llm-prices.com declares no licence, so it is left out by default and the committed data is built
without it. Pass --with-llm-prices for a local build that includes it (also writes the conflict report).

Inputs (see README for licences):
  .cache/litellm_monthly.parquet   from scripts/fetch_litellm_history.py
  .cache/llm-prices/historical-v1.json  (only with --with-llm-prices)
  .cache/epoch/                    unzipped https://epoch.ai/data/benchmark_data.zip
  data/prices.parquet              OpenRouter daily snapshots (only its model IDs are used)

Outputs:
  data/model_map.csv, data/price_history.parquet, data/models.csv,
  data/price_conflicts.csv, data/price_conflicts.md  (only with --with-llm-prices)
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache"
DATA = ROOT / "data"
OVERRIDES = DATA / "model_map_overrides.csv"
CONFLICT_THRESHOLD = 0.05
PRICE_COLS = ["input_price_per_mtok", "output_price_per_mtok", "cache_read_price_per_mtok"]

# ---------------------------------------------------------------- normalisation

VENDOR_ALIASES = {
    "meta": "meta-llama", "meta-ai": "meta-llama", "meta-llama": "meta-llama",
    "mistral": "mistralai", "mistral-ai": "mistralai", "mistralai": "mistralai",
    "xai": "x-ai", "x-ai": "x-ai",
    "moonshot": "moonshotai", "moonshot-ai": "moonshotai", "moonshotai": "moonshotai",
    "zai": "z-ai", "z-ai": "z-ai", "zai-org": "z-ai", "z-ai-(zhipu-ai)": "z-ai", "zhipu": "z-ai",
    "alibaba": "qwen", "qwen": "qwen",
    "deepseek-ai": "deepseek", "deepseek": "deepseek",
    "google-deepmind": "google", "google": "google",
    "openai": "openai", "anthropic": "anthropic", "amazon": "amazon", "cohere": "cohere",
    "nvidia": "nvidia", "microsoft": "microsoft", "minimax": "minimax", "perplexity": "perplexity",
    "thinking-machines": "thinkingmachines", "thinkingmachines": "thinkingmachines", "tencent": "tencent",
    "stepfun": "stepfun", "stepfun-ai": "stepfun", "gryphe": "gryphe", "01-ai": "01-ai", "bigcode": "bigcode",
}
GENERIC_PATH_PARTS = {"models", "invoke", "converse", "v1", "accounts"}

# (regex anchored at slug start, canonical vendor); first hit wins
NAME_RULES = [
    (r"claude", "anthropic"),
    (r"deepseek", "deepseek"),
    (r"nvidia|nemotron", "nvidia"),
    (r"gpt|o\d(-|$)|chatgpt|text-davinci|davinci|babbage|text-curie|codex|computer-use-preview|omni-moderation", "openai"),
    (r"gemini|gemma|palm|bison|gecko|learnlm|imagen|veo", "google"),
    (r"llama|codellama", "meta-llama"),
    (r"mistral|mixtral|codestral|ministral|pixtral|devstral|magistral|mathstral|voxtral", "mistralai"),
    (r"qwen|qwq", "qwen"),
    (r"command|aya", "cohere"),
    (r"grok", "x-ai"),
    (r"nova|titan", "amazon"),
    (r"jamba|j2-", "ai21"),
    (r"kimi|moonshot", "moonshotai"),
    (r"glm|chatglm", "z-ai"),
    (r"minimax|abab", "minimax"),
    (r"phi-", "microsoft"),
    (r"sonar|r1-1776", "perplexity"),
    (r"hunyuan", "tencent"),
    (r"ernie", "baidu"),
    (r"doubao|seed-", "bytedance"),
    (r"mimo", "xiaomi"),
    (r"falcon", "tii"),
    (r"muse-", "meta-llama"),
    (r"step-\d", "stepfun"),
    (r"hy\d", "tencent"),
    (r"mythomax", "gryphe"),
    (r"inkling", "thinkingmachines"),
    (r"yi-", "01-ai"),
    (r"starcoder", "bigcode"),
    (r"lyria", "google"),
]
REGION_RE = re.compile(r"^(us|eu|apac|au|jp|ca|global|us-gov)\.")
VENDOR_DOT_RE = re.compile(r"^(anthropic|meta|amazon|mistral|cohere|ai21|deepseek|xai|openai|qwen|moonshotai|"
                           r"writer|twelvelabs|google|stability|minimax|zai)\.")
DATE_RES = [
    re.compile(r"-20\d{2}-?\d{2}-?\d{2}$"),                        # -20250219 / -2025-02-19
    re.compile(r"-2[3-6](0[1-9]|1[0-2])$"),                        # -2407 (YYMM)
    re.compile(r"-(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])$"),         # -0613 (MMDD)
    re.compile(r"-00\d$"),                                         # -001 version
]


def alias_vendor(v: str | None) -> str | None:
    if not isinstance(v, str) or not v:
        return None
    k = re.sub(r"[\s_]+", "-", v.strip().lower().lstrip("~"))
    return VENDOR_ALIASES.get(k)


def vendor_from_name(slug: str) -> str | None:
    for pat, v in NAME_RULES:
        if re.match(pat, slug):
            return v
    return None


def slugify(raw: str) -> tuple[str, str | None, str | None]:
    """Return (slug, date_stamp, vendor_hint_from_path) for a model identifier."""
    s = raw.strip().lower().lstrip("~")
    s = re.sub(r"\([^)]*\)", "", s).strip()          # Epoch-style "(May 2024)"
    s = re.sub(r"@.*$", "", s)                        # vertex "@default" / "@20250219"
    s = s.replace(":thinking", "-thinking")
    s = re.sub(r":(free|nitro|floor|extended|beta|online|exacto|batch)$", "", s)
    parts = [p for p in s.split("/") if p]
    s = parts[-1] if parts else s
    hint = parts[-2] if len(parts) >= 2 and parts[-2] not in GENERIC_PATH_PARTS else None
    s = REGION_RE.sub("", s)
    s = VENDOR_DOT_RE.sub("", s)
    s = re.sub(r"-v\d+(:\d+)?$", "", s)
    s = re.sub(r":\d+$", "", s)
    s = re.sub(r"(?<=\d)\.(?=\d)", "-", s)
    s = re.sub(r"[\s_.:]+", "-", s)
    s = re.sub(r"^(llama|gemma|qwen|mistral)(\d)", r"\1-\2", s)  # bedrock "llama3-1"
    s = re.sub(r"-{2,}", "-", s).strip("-")
    s = re.sub(r"-preview-\d{2}-\d{2}$", "-preview", s)
    s = re.sub(r"-latest$", "", s)
    stamp = None
    for rx in DATE_RES:
        m = rx.search(s)
        if m:
            stamp = m.group(0).lstrip("-")
            s = s[: m.start()]
            break
    return s, stamp, hint


def canonical(source: str, raw: str, vendor_hint: str | None) -> tuple[str, str | None, bool]:
    """Return (canonical_id, date_stamp, vendor_inferred_by_fallback_or_unknown)."""
    slug, stamp, path_hint = slugify(raw)
    explicit = alias_vendor(vendor_hint)
    if source == "openrouter":
        explicit = alias_vendor(raw.split("/")[0]) or alias_vendor(path_hint)
    by_name = vendor_from_name(slug)
    if source in ("openrouter", "llm_prices"):
        vendor = explicit or by_name or alias_vendor(path_hint)
    else:
        vendor = by_name or explicit or alias_vendor(path_hint)
    fallback = vendor is None
    if vendor is None:  # long tail (hosting-provider keys etc.): use the org in the path
        org = path_hint or (raw.split("/")[0] if "/" in raw else None)
        vendor = re.sub(r"[^a-z0-9-]+", "-", org.lower()).strip("-") if org else "unknown"
    return f"{vendor}/{slug}", stamp, fallback


# ---------------------------------------------------------------- loading


LLM_PRICES_COLS = ["id", "vendor", "name", "input", "output", "input_cached", "from_date", "to_date"]


def load_sources(with_llm_prices: bool):
    lit = pd.read_parquet(CACHE / "litellm_monthly.parquet")
    if with_llm_prices:
        llm = pd.DataFrame(json.loads((CACHE / "llm-prices" / "historical-v1.json").read_text())["prices"])
    else:
        llm = pd.DataFrame(columns=LLM_PRICES_COLS)
    llm = llm.drop_duplicates(["id", "input", "output", "input_cached", "from_date", "to_date"])  # source has exact repeats
    orp = pd.read_parquet(DATA / "prices.parquet")
    eci = pd.read_csv(CACHE / "epoch" / "epoch_capabilities_index" / "eci_scores.csv")
    return lit, llm, orp, eci


def build_map(lit, llm, orp, eci) -> pd.DataFrame:
    rows = []
    for k, prov in lit.groupby("model_id")["provider"].first().items():
        rows.append(("litellm", k, None, k))
    for k in orp.model_id.unique():
        rows.append(("openrouter", k, None, k))
    for r in llm.drop_duplicates("id").itertuples():
        rows.append(("llm_prices", r.id, r.vendor, r.name))
    for r in eci.itertuples():
        rows.append(("epoch", r.Model, r.Organization, r.Model))
    out = []
    for source, sid, vhint, name in rows:
        cid, stamp, fb = canonical(source, sid, vhint)
        out.append({"source": source, "source_id": sid, "source_name": name, "canonical_id": cid,
                    "date_stamp": stamp, "vendor_unresolved": fb})
    m = pd.DataFrame(out)
    m["match_method"] = ""
    m["candidate_canonical_id"] = ""
    m["needs_review"] = False
    m["note"] = ""
    m = split_dated(m, lit)
    m = apply_overrides(m)

    # llm-prices context-tier rows ("claude-sonnet-4-200k"): same model, different price tier
    lp_ids = set(llm.id)
    tier = m.source.eq("llm_prices") & (m.match_method == "") & m.source_id.map(
        lambda i: bool(re.search(r"-\d+k$", i)) and re.sub(r"-\d+k$", "", i) in lp_ids)
    base_of = m[tier].source_id.map(lambda i: re.sub(r"-\d+k$", "", i))
    for idx, base in base_of.items():
        c = m[(m.source == "llm_prices") & (m.source_id == base)].canonical_id.iloc[0]
        m.loc[idx, "canonical_id"] = c
        m.loc[idx, "match_method"] = "tier_variant"
        m.loc[idx, "note"] = f"context-length price tier of {base}; excluded from price_history"

    # Epoch: several rows collapsing into one canonical id (dated snapshots) are ambiguous
    ep = m.source.eq("epoch")
    dup = ep & (m.match_method == "") & m.duplicated(["source", "canonical_id"], keep=False)
    m.loc[dup, ["match_method", "needs_review"]] = ["ambiguous_epoch_variants", True]
    m.loc[dup, "note"] = "several Epoch models normalise to this id (dated snapshots?); ECI not attached"

    n_src = m[m.match_method != "tier_variant"].groupby("canonical_id").source.nunique()
    m["n_sources"] = m.canonical_id.map(n_src).fillna(1).astype(int)
    multi = (m.match_method == "") & (m.n_sources >= 2)
    m.loc[multi, "match_method"] = "exact_normalised"

    # fuzzy candidates for single-source ids from the small sources (no merging, review only)
    all_canon = pd.Series(m[m.match_method != "ambiguous_epoch_variants"].canonical_id.unique())
    by_slug = {c.split("/", 1)[1]: c for c in all_canon}
    for idx in m[(m.match_method == "") & m.source.isin(["openrouter", "llm_prices", "epoch"])].index:
        cid = m.at[idx, "canonical_id"]
        slug = cid.split("/", 1)[1]
        srcs_of_cid = set(m[m.canonical_id == cid].source)
        cands = [s for s in difflib.get_close_matches(slug, by_slug, n=3, cutoff=0.88) if by_slug[s] != cid]
        cands = [s for s in cands if set(m[m.canonical_id == by_slug[s]].source) - srcs_of_cid]
        if cands:
            m.at[idx, "match_method"] = "fuzzy_candidate"
            m.at[idx, "candidate_canonical_id"] = by_slug[cands[0]]
            m.at[idx, "needs_review"] = True
            m.at[idx, "note"] = "no exact match; closest: " + ", ".join(by_slug[c] for c in cands)
        else:
            m.at[idx, "match_method"] = "single_source"
    m.loc[m.match_method == "", "match_method"] = "single_source"

    unk = m.vendor_unresolved & (m.note == "") & ~m.match_method.isin(["single_source", "manual_override"])
    m.loc[unk, "note"] = "vendor taken from path/provider, not recognised"
    # vendor mismatch for same slug across sources is also a review item
    slug_col = m.canonical_id.str.split("/", n=1).str[1]
    vend_n = m[m.match_method != "tier_variant"].assign(slug=slug_col).groupby("slug").canonical_id.nunique()
    split = slug_col.map(vend_n).fillna(1) > 1
    sel = split & ~m.needs_review & (m.source != "litellm") & (m.match_method != "manual_override")
    m.loc[sel, "needs_review"] = True
    m.loc[sel, "note"] = "same model name resolved to different vendors across sources"
    return m[["source", "source_id", "source_name", "canonical_id", "match_method", "needs_review",
              "candidate_canonical_id", "n_sources", "date_stamp", "note"]]


def split_dated(m: pd.DataFrame, lit: pd.DataFrame) -> pd.DataFrame:
    """Keep the date stamp in the canonical id when dated snapshots of one model are priced differently.

    Prices are compared per stamp at that stamp's latest month, using first-party LiteLLM rows when
    those cover at least two stamps, otherwise all providers. Undated aliases
    (`gpt-4o`, `-latest`) stay on the undated id.
    """
    li = m.loc[(m.source == "litellm") & m.date_stamp.notna(), ["source_id", "canonical_id", "date_stamp"]]
    p = lit.merge(li, left_on="model_id", right_on="source_id")
    p["stamp"] = p.date_stamp.str.replace("-", "", regex=False)
    vendor = p.canonical_id.str.split("/").str[0]
    p["first_party"] = [prov in FIRST_PARTY.get(v, set()) for prov, v in zip(p.provider, vendor)]
    split = set()
    for cid, g in p.groupby("canonical_id"):
        if g[g.first_party].stamp.nunique() >= 2:  # prefer the model maker's own prices
            g = g[g.first_party]
        if g.stamp.nunique() < 2:
            continue
        last = g[g.snapshot_date == g.groupby("stamp").snapshot_date.transform("max")]
        for col in ("input_price_per_mtok", "output_price_per_mtok"):
            per = last[last[col] > 0].groupby("stamp")[col].median()
            if len(per) >= 2 and (per.max() - per.min()) / per.max() > CONFLICT_THRESHOLD:
                split.add(cid)
                break
    sel = m.canonical_id.isin(split) & m.date_stamp.notna()
    m.loc[sel, "canonical_id"] = m.loc[sel, "canonical_id"] + "-" + m.loc[sel, "date_stamp"].str.replace("-", "", regex=False)
    m.loc[sel, "note"] = "dated snapshot kept separate: snapshots of this model are priced differently"
    print(f"date-split: {len(split)} model groups, {int(sel.sum())} ids now carry their date stamp")
    return m


def apply_overrides(m: pd.DataFrame) -> pd.DataFrame:
    """Apply hand-reviewed mappings from data/model_map_overrides.csv (source, source_id, canonical_id, note)."""
    if not OVERRIDES.exists():
        return m
    ov = pd.read_csv(OVERRIDES, comment="#").fillna("")
    for r in ov.itertuples():
        hit = (m.source == r.source) & (m.source_id == r.source_id)
        if not hit.any():
            print(f"override not applied, id not found: {r.source} {r.source_id}", file=sys.stderr)
            continue
        m.loc[hit, ["canonical_id", "match_method", "needs_review", "candidate_canonical_id"]] = \
            [r.canonical_id, "manual_override", False, ""]
        m.loc[hit, "note"] = r.note or "manual override"
    return m


# ---------------------------------------------------------------- history


def month_starts(first: str, last: str):
    return pd.date_range(first, last, freq="MS")


def expand_llm_prices(llm, mp, lit, eci_exact, months) -> pd.DataFrame:
    cmap = mp[mp.source == "llm_prices"].set_index("source_id")
    first_lit = lit.assign(cid=lit.model_id.map(mp[mp.source == "litellm"].set_index("source_id").canonical_id)) \
        .groupby("cid").snapshot_date.min()
    rows = []
    for r in llm.itertuples():
        info = cmap.loc[r.id]
        if info.match_method == "tier_variant":
            continue
        cid = info.canonical_id
        frm = pd.Timestamp(r.from_date) if pd.notna(r.from_date) else None
        to = pd.Timestamp(r.to_date) if pd.notna(r.to_date) else None
        inferred = False
        if frm is None:
            inferred = True
            frm = eci_exact.get(cid) or first_lit.get(cid) or months[-1]
            frm = max(pd.Timestamp(frm).to_period("M").to_timestamp(), months[0])
        for mth in months:
            if mth >= frm and (to is None or mth < to):
                rows.append({
                    "snapshot_date": mth, "model_id": r.id, "provider": r.vendor,
                    "input_price_per_mtok": r.input, "output_price_per_mtok": r.output,
                    "cache_read_price_per_mtok": None if pd.isna(r.input_cached) else r.input_cached, "context_length": pd.NA,
                    "inferred_start": inferred,
                })
    return pd.DataFrame(rows)


# first-party LiteLLM providers per canonical vendor (used as the reference for cross-source checks)
FIRST_PARTY = {
    "openai": {"openai", "text-completion-openai"}, "anthropic": {"anthropic"},
    "google": {"gemini", "vertex_ai-language-models"}, "mistralai": {"mistral", "text-completion-codestral"},
    "deepseek": {"deepseek"}, "x-ai": {"xai"}, "moonshotai": {"moonshot"}, "amazon": {"bedrock", "bedrock_converse"},
    "qwen": {"dashscope"}, "minimax": {"minimax"}, "perplexity": {"perplexity"},
}


def find_conflicts(hist: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    h = hist[hist.canonical_id.notna()]
    lp = h[h.source == "llm_prices"]
    li = h[h.source == "litellm"].copy()
    li["vendor"] = li.canonical_id.str.split("/").str[0]
    li = li[[p in FIRST_PARTY.get(v, set()) for p, v in zip(li.provider, li.vendor)]]
    out = []
    n_cmp = 0
    for col in PRICE_COLS:
        a = lp[["snapshot_date", "canonical_id", "model_id", col]].dropna(subset=[col])
        b = li[["snapshot_date", "canonical_id", "model_id", col]].dropna(subset=[col])
        j = a.merge(b, on=["snapshot_date", "canonical_id"], suffixes=("_llm_prices", "_litellm"))
        j["rel_diff"] = (j[f"{col}_llm_prices"] - j[f"{col}_litellm"]).abs() / j[[f"{col}_llm_prices", f"{col}_litellm"]].max(axis=1)
        # conflict only if NO first-party LiteLLM row for that model/month agrees within threshold
        best = j.sort_values("rel_diff").groupby(["snapshot_date", "canonical_id", "model_id_llm_prices"], as_index=False).first()
        n_cmp += len(best)
        best = best[best.rel_diff > CONFLICT_THRESHOLD]
        for r in best.itertuples():
            out.append({"snapshot_date": r.snapshot_date, "canonical_id": r.canonical_id, "field": col,
                        "llm_prices_id": r.model_id_llm_prices, "litellm_id": r.model_id_litellm,
                        "llm_prices_value": getattr(r, f"{col}_llm_prices"),
                        "litellm_value": getattr(r, f"{col}_litellm"), "rel_diff": r.rel_diff})
    df = pd.DataFrame(out).sort_values(["canonical_id", "field", "snapshot_date"]).reset_index(drop=True)
    return df, n_cmp


def write_conflict_md(conf: pd.DataFrame, n_compared: int, path: Path):
    lines = ["# Price conflicts between LiteLLM and llm-prices.com", "",
             f"Both sources are kept in `price_history.parquet` (rows flagged `price_conflict = true`). "
             f"A conflict means the llm-prices value differs by more than {CONFLICT_THRESHOLD:.0%} from every "
             "first-party LiteLLM row for the same canonical model and month "
             "(relative difference = |a-b| / max(a,b)).", "",
             f"- model/month/field comparisons made: {n_compared}",
             f"- conflicts (month level): {len(conf)}",
             f"- distinct models with a conflict: {conf.canonical_id.nunique() if len(conf) else 0}", ""]
    if len(conf):
        lines += ["## Conflict periods", "",
                  "| canonical_id | field | months | llm-prices | LiteLLM | max diff | llm-prices id / LiteLLM key |",
                  "|---|---|---|---|---|---|---|"]
        for (cid, f), g in conf.groupby(["canonical_id", "field"]):
            g = g.sort_values("snapshot_date")
            lines.append(
                f"| {cid} | {f.replace('_price_per_mtok', '')} | {g.snapshot_date.min():%Y-%m} → {g.snapshot_date.max():%Y-%m} "
                f"({len(g)}) | {g.llm_prices_value.iloc[-1]:.4g} | {g.litellm_value.iloc[-1]:.4g} | "
                f"{g.rel_diff.max():.0%} | {g.llm_prices_id.iloc[0]} / {g.litellm_id.iloc[0]} |")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-llm-prices", action="store_true",
                    help="include llm-prices.com (no licence declared: don't publish the result)")
    with_llm = ap.parse_args().with_llm_prices
    lit, llm, orp, eci = load_sources(with_llm)
    months = month_starts("2023-10-01", lit.snapshot_date.max())

    mp = build_map(lit, llm, orp, eci)

    # ECI: attach only exact, unambiguous Epoch matches
    ep = mp[(mp.source == "epoch") & (mp.match_method != "ambiguous_epoch_variants")]
    eci_idx = eci.set_index("Model")
    ep = ep.assign(eci=ep.source_id.map(eci_idx.eci), eci_ci_low=ep.source_id.map(eci_idx.eci_ci_low),
                   eci_ci_high=ep.source_id.map(eci_idx.eci_ci_high),
                   epoch_release=pd.to_datetime(ep.source_id.map(eci_idx.date)))
    # an Epoch row is usable if its canonical id also exists in a price source
    price_canon = set(mp[mp.source.isin(["litellm", "llm_prices", "openrouter"])].canonical_id)
    ep_exact = ep[ep.canonical_id.isin(price_canon)].drop_duplicates("canonical_id")
    epoch_release = ep_exact.set_index("canonical_id").epoch_release.to_dict()

    lit_rows = lit.merge(mp[mp.source == "litellm"][["source_id", "canonical_id"]],
                         left_on="model_id", right_on="source_id", how="left").drop(columns=["source_id", "litellm_commit"])
    lit_rows["source"] = "litellm"
    lit_rows["inferred_start"] = False
    if with_llm:
        llm_rows = expand_llm_prices(llm, mp, lit, epoch_release, months)
        llm_rows = llm_rows.merge(mp[mp.source == "llm_prices"][["source_id", "canonical_id"]],
                                  left_on="model_id", right_on="source_id", how="left").drop(columns="source_id")
        llm_rows["source"] = "llm_prices"
        hist = pd.concat([lit_rows, llm_rows], ignore_index=True)
        conf, n_cmp = find_conflicts(hist)
    else:
        hist, conf, n_cmp = lit_rows, pd.DataFrame(), 0
    conf_lp = set(zip(conf.snapshot_date, conf.llm_prices_id)) if len(conf) else set()
    conf_li = set(zip(conf.snapshot_date, conf.litellm_id)) if len(conf) else set()
    hist["price_conflict"] = [
        (d, m) in (conf_lp if s == "llm_prices" else conf_li) for d, m, s in zip(hist.snapshot_date, hist.model_id, hist.source)]

    eci_map = ep_exact.set_index("canonical_id")
    hist["eci"] = hist.canonical_id.map(eci_map.eci)
    hist = hist.astype({"model_id": "string", "provider": "string", "canonical_id": "string", "source": "string",
                        "context_length": "Int64", "snapshot_date": "datetime64[ns]"})
    cols = ["snapshot_date", "model_id", "provider", "input_price_per_mtok", "output_price_per_mtok",
            "cache_read_price_per_mtok", "context_length", "source", "canonical_id", "eci", "inferred_start", "price_conflict"]
    hist = hist[cols].sort_values(["snapshot_date", "source", "model_id"]).reset_index(drop=True)
    dupes = hist.duplicated(["snapshot_date", "source", "model_id"]).sum()
    if dupes:
        raise SystemExit(f"{dupes} duplicate (snapshot_date, source, model_id) rows; refusing to write")

    ov_bad = set(mp[mp.match_method == "manual_override"].canonical_id) - set(hist.canonical_id)
    for c in sorted(ov_bad):
        print(f"warning: override target has no price history: {c}", file=sys.stderr)

    # per-canonical table
    g = hist.groupby("canonical_id").agg(first_month=("snapshot_date", "min"), last_month=("snapshot_date", "max"),
                                         sources=("source", lambda s: ",".join(sorted(set(s)))))
    g["vendor"] = g.index.str.split("/").str[0]
    g["eci"] = g.index.map(eci_map.eci)
    g["eci_ci_low"] = g.index.map(eci_map.eci_ci_low)
    g["eci_ci_high"] = g.index.map(eci_map.eci_ci_high)
    g["epoch_model"] = g.index.map(eci_map.source_id)
    g["epoch_release_date"] = g.index.map(eci_map.epoch_release)

    DATA.mkdir(exist_ok=True)
    mp.sort_values(["needs_review", "canonical_id", "source"], ascending=[False, True, True]).to_csv(DATA / "model_map.csv", index=False)
    hist.to_parquet(DATA / "price_history.parquet", index=False)
    g.reset_index().to_csv(DATA / "models.csv", index=False)
    if with_llm:
        if len(conf):
            conf.to_csv(DATA / "price_conflicts.csv", index=False)
        write_conflict_md(conf, n_cmp, DATA / "price_conflicts.md")
    else:  # conflicts only exist between the two price sources
        for p in ("price_conflicts.csv", "price_conflicts.md"):
            (DATA / p).unlink(missing_ok=True)

    print(f"rows={len(hist)} canonical={hist.canonical_id.nunique()} range={hist.snapshot_date.min():%Y-%m}..{hist.snapshot_date.max():%Y-%m}")
    print(f"with ECI: {g.eci.notna().sum()}  conflicts: {len(conf)} ({conf.canonical_id.nunique() if len(conf) else 0} models)")
    print(mp.match_method.value_counts().to_dict(), "review:", int(mp.needs_review.sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
