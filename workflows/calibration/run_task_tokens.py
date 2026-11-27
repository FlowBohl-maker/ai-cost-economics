"""Optional calibration: run the four synthetic finance workflows against a model and log tokens per task.

The cost model does not need this. Profiles in workflows/profiles/ can be written by hand; these runs only
check or calibrate their token volumes. Model access goes through an adapter (workflows/calibration/adapters.py),
so the same runner works with any provider that has one.

  uv run python -m workflows.calibration.run_task_tokens --adapter mock --models demo --estimate
  uv run python -m workflows.calibration.run_task_tokens --adapter ollama --models gpt-oss:20b --run --reps 1
  uv run python -m workflows.calibration.run_task_tokens summarise --runs data/tasks/token_runs.parquet --out drafts/

--run stops before the next run could take total spend over --max-cost (default $20), using the adapter's list
prices; it refuses to start if the estimate (x1.5 margin) is over the cap or if the adapter has no prices.
Raw runs go to .cache/tasks/runs_<adapter>.jsonl (resume log, git-ignored); results to data/tasks/token_runs.parquet
(mock runs stay in .cache/tasks/).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from . import adapters
from . import task_inputs as ti

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache" / "tasks"
OUT = ROOT / "data" / "tasks" / "token_runs.parquet"

MAX_TOKENS = 32000
MAX_AGENT_TURNS = 10
SAFETY = 1.5
# planning assumptions for the estimate: output tokens (incl. reasoning) per model call, and calls per agent run
EST_OUTPUT = {"data_qa": {"low": 600, "high": 4000, None: 600}, "screening": {"low": 2500, "high": 12000, None: 2500},
              "earnings_summary": {"low": 1500, "high": 6000, None: 1500}, "portfolio_agent": {"low": 600, "high": 3000, None: 600}}
EST_AGENT_TURNS = 5


def messages(workflow: str) -> list[dict]:
    return {"data_qa": ti.qa_prompt, "screening": ti.screen_prompt, "earnings_summary": ti.earnings_prompt,
            "portfolio_agent": ti.agent_prompt}[workflow]()


def input_tokens_est(workflow: str) -> int:
    if workflow == "earnings_summary" and not (ti.TRANSCRIPT.exists() or ti.TRANSCRIPT.with_suffix(".htm").exists()):
        return 16000  # ~57k characters; the transcript is not in the repo (see task_inputs.TRANSCRIPT_SOURCE)
    return int(sum(len(m["content"]) for m in messages(workflow)) / adapters.CHARS_PER_TOKEN)


def estimate(ad, models: list[str], workflows: list[str], efforts: list, reps: int) -> pd.DataFrame:
    tool_chars = sum(len(ti.call_tool(n, a)) for n, a in (
        ("get_portfolio", "{}"), ("get_risk_metrics", json.dumps({"tickers": [h["ticker"] for h in ti.portfolio()["holdings"]]})),
        ("get_news", json.dumps({"tickers": ti.agent_truth()["top_vol"]})), ("get_exposures", '{"dimension":"sector"}'),
        ("get_exposures", '{"dimension":"region"}')))
    rows = []
    for model in models:
        pr = ad.price(model)
        for wf in workflows:
            for eff in efforts:
                out = EST_OUTPUT[wf][eff]
                if wf == "portfolio_agent":
                    base = input_tokens_est(wf) + 800  # tool schemas
                    inp = sum(base + (tool_chars / adapters.CHARS_PER_TOKEN) * min(k, 4) / 4 + out * k for k in range(EST_AGENT_TURNS))
                    out *= EST_AGENT_TURNS
                else:
                    inp = input_tokens_est(wf)
                cost = None if pr is None else (inp * pr["input"] + out * pr["output"]) / 1e6
                rows.append({"adapter": ad.name, "model": model, "workflow": wf, "effort": eff, "input_tokens": inp,
                             "output_tokens": out, "cost_per_run": cost, "runs": reps,
                             "cost": None if cost is None else cost * reps})
    return pd.DataFrame(rows)


def run_one(ad, model: str, wf: str, effort, rep: int, budget_left: float) -> dict:
    msgs = messages(wf)
    calls, final, tools_used = [], "", set()
    if wf != "portfolio_agent":
        msg, rec = ad.chat(model, msgs, effort=effort, tools=None, max_tokens=MAX_TOKENS)
        calls.append(rec)
        final = msg.get("content") or ""
    else:
        for _ in range(MAX_AGENT_TURNS):
            msg, rec = ad.chat(model, msgs, effort=effort, tools=ti.AGENT_TOOLS, max_tokens=MAX_TOKENS)
            calls.append(rec)
            tcs = msg.get("tool_calls") or []
            msgs.append({k: msg[k] for k in ("role", "content", "tool_calls", "reasoning_details") if msg.get(k) is not None})
            if not tcs:
                final = msg.get("content") or ""
                break
            for tc in tcs:
                tools_used.add(tc["function"]["name"])
                msgs.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": ti.call_tool(tc["function"]["name"], tc["function"].get("arguments"))})
            if sum(c["cost_usd"] for c in calls) > budget_left:
                raise RuntimeError("budget exhausted mid-run")
    n_turns = len(calls)
    score = {"data_qa": lambda: ti.qa_score(final), "screening": lambda: ti.screen_score(final),
             "earnings_summary": lambda: ti.earnings_score(final),
             "portfolio_agent": lambda: ti.agent_score(final, n_turns, tools_used)}[wf]()
    tot = lambda k: sum(c[k] for c in calls)
    return {"adapter": ad.name, "model": model, "workflow": wf, "effort": effort, "rep": rep, "n_calls": n_turns,
            "tools_used": sorted(tools_used),
            **{k: tot(k) for k in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens", "cache_write_tokens",
                                   "cost_usd", "latency_s")},
            "reasoning_tokens_estimated": any(c["reasoning_tokens_estimated"] for c in calls),
            "providers": sorted({str(c["provider"]) for c in calls}), "finish_reasons": [c["finish_reason"] for c in calls],
            "quality": score, "final_text": final, "calls": calls, "ts": pd.Timestamp.now(tz="UTC").isoformat()}


def write_parquet(log: Path, out: Path) -> pd.DataFrame:
    df = pd.DataFrame([json.loads(x) for x in log.read_text().splitlines() if x.strip()])
    if "calls" in df:
        df["calls"] = df["calls"].map(json.dumps)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def run(a) -> int:
    ad = adapters.get(a.adapter)
    models = a.models.split(",")
    wfs = a.only.split(",") if a.only else ti.WORKFLOWS
    efforts = [None if e == "none" else e for e in a.efforts.split(",")]
    est = estimate(ad, models, wfs, efforts, a.reps)
    print(est.groupby(["model", "workflow"])[["input_tokens", "output_tokens"]].mean().round(0).to_string())
    if est.cost.isna().any():
        print(f"no prices for: {', '.join(est[est.cost.isna()].model.unique())}; the {ad.name} adapter must return prices to run")
        return 0 if not a.run else 1
    total = est.cost.sum()
    print(f"runs: {int(est.runs.sum())}  estimated total: ${total:.2f}  with x{SAFETY} margin: ${total * SAFETY:.2f}")
    if not a.run:
        return 0
    if total * SAFETY > a.max_cost:
        raise SystemExit(f"estimate with margin exceeds --max-cost {a.max_cost}; not running")
    log = Path(a.log or CACHE / f"runs_{ad.name}.jsonl")
    out = Path(a.out or (CACHE / "token_runs_mock.parquet" if ad.name == "mock" else OUT))
    log.parent.mkdir(parents=True, exist_ok=True)
    prev = [json.loads(x) for x in log.read_text().splitlines() if x.strip()] if log.exists() else []
    done = {(r["model"], r["workflow"], r["effort"], r["rep"]) for r in prev if "error" not in r}
    spent = sum(r["cost_usd"] for r in prev)
    est_run = {(r.model, r.workflow, r.effort): r.cost_per_run for r in est.itertuples()}
    errors_in_a_row = 0
    for rep in range(1, a.reps + 1):                                        # every combination once before repeats
        for model in sorted(models, key=lambda m: est[est.model == m].cost.sum()):  # cheapest model first
            for wf in wfs:
                for eff in efforts:
                    if (model, wf, eff, rep) in done:
                        continue
                    worst = est_run[(model, wf, eff)] * 3
                    if spent + worst > a.max_cost:
                        print(f"stop: spent ${spent:.2f}; next run could cost up to ${worst:.2f} (limit ${a.max_cost})")
                        write_parquet(log, out)
                        return 1
                    try:
                        r = run_one(ad, model, wf, eff, rep, a.max_cost - spent)
                    except RuntimeError as e:
                        r = {"adapter": ad.name, "model": model, "workflow": wf, "effort": eff, "rep": rep,
                             "error": str(e)[:500], "cost_usd": 0.0, "ts": pd.Timestamp.now(tz="UTC").isoformat()}
                    with log.open("a") as fh:
                        fh.write(json.dumps(r) + "\n")
                    spent += r["cost_usd"]
                    errors_in_a_row = errors_in_a_row + 1 if "error" in r else 0
                    print(f"rep{rep} {model:24} {wf:17} {str(eff):4} in={r.get('input_tokens')} out={r.get('output_tokens')} "
                          f"q={r.get('quality')} ${r['cost_usd']:.4f} total ${spent:.2f} {r.get('error', '')[:80]}", flush=True)
                    if errors_in_a_row >= 3:
                        print("stop: 3 errors in a row (is the model server reachable?); rerun to resume")
                        write_parquet(log, out)
                        return 1
    df = write_parquet(log, out)
    print(f"done; {len(df)} runs in {out}; spent ${spent:.2f}")
    return 0


def draft_profiles(runs: pd.DataFrame) -> list[dict]:
    """One draft profile per (model, workflow, effort): low / central / high = min / median / max over repetitions.

    Usage fields follow the OpenAI convention: input_tokens includes cached reads and output_tokens includes
    reasoning, so they are split into the profile's four billed token types here.
    """
    r = runs[runs.get("error").isna()] if "error" in runs else runs
    r = r.assign(input=r.input_tokens - r.cached_tokens, output=r.output_tokens - r.reasoning_tokens,
                 reasoning=r.reasoning_tokens, cache_read=r.cached_tokens)
    out = []
    for (model, wf, eff), g in r.groupby(["model", "workflow", r.effort.fillna("none")]):
        tok = {k: {"low": int(g[k].min()), "central": int(g[k].median()), "high": int(g[k].max())}
               for k in ("input", "output", "reasoning", "cache_read")}
        out.append({"name": f"{wf}-{model.replace('/', '-').replace(':', '-')}-{eff}",
                    "description": f"Draft calibrated from {len(g)} run(s) of the synthetic '{wf}' workflow on {model}. "
                                   "Set model.tier or model.min_eci before use.",
                    "tokens_per_task": tok, "model": {"tier": "CHOOSE"}, "effort": eff,
                    "calibration": {"source": "workflows/calibration/run_task_tokens.py", "adapter": str(g.adapter.iloc[0]),
                                    "model": model, "runs": int(len(g)), "median_quality": float(g.quality.median())}})
    return out


def summarise(a) -> int:
    runs = pd.read_parquet(a.runs)
    drafts = draft_profiles(runs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for d in drafts:
        (out / f"{d['name']}.yaml").write_text(yaml.safe_dump(d, sort_keys=False))
    print(f"wrote {len(drafts)} draft profiles to {out}")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["summarise"]:
        ap = argparse.ArgumentParser(prog="run_task_tokens summarise", description="Draft profiles from logged runs")
        ap.add_argument("--runs", default=str(OUT))
        ap.add_argument("--out", required=True, help="folder for draft profile YAML files (review before moving to profiles/)")
        return summarise(ap.parse_args(argv[1:]))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, choices=sorted(adapters.ADAPTERS))
    ap.add_argument("--models", required=True, help="comma-separated model names as the adapter knows them")
    ap.add_argument("--efforts", default="low,high", help="comma-separated reasoning efforts; 'none' sends no effort setting")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--only", help="comma-separated workflows (default: all four)")
    ap.add_argument("--estimate", action="store_true", help="print the estimate only (the default without --run)")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--max-cost", type=float, default=20.0)
    ap.add_argument("--log", help="resume log (default .cache/tasks/runs_<adapter>.jsonl)")
    ap.add_argument("--out", help="results parquet (default data/tasks/token_runs.parquet)")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
