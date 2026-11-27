"""Model-access adapters for the calibration runner. One small class per provider.

An adapter turns one chat request into (assistant message, usage record). The runner never talks to a
provider directly, so adding a provider means adding one class here and registering it in ADAPTERS.

Included (none needs an account):
  mock    offline and deterministic; answers from the synthetic ground truth. For tests and dry runs.
  ollama  a local Ollama server (OpenAI-compatible endpoint, default http://localhost:11434). Cost is 0.

To add a provider that needs a key (e.g. Gemini's free tier, or a single vendor's API), add a class with
the same three members, read the key from the environment or the git-ignored .env inside that class, and
return list prices from `price()` so the runner's cost estimate and --max-cost cap work.
"""

from __future__ import annotations

import json
import os
import time
from typing import Protocol

import requests

from . import task_inputs as ti

CHARS_PER_TOKEN = 3.6


class Adapter(Protocol):
    name: str

    def price(self, model: str) -> dict | None:
        """USD per 1M tokens: {"input", "output", "cache_read"}; None if unknown (then the runner won't run)."""

    def chat(self, model: str, messages: list[dict], *, effort: str | None, tools: list | None,
             max_tokens: int) -> tuple[dict, dict]:
        """Return (assistant message in OpenAI chat format, usage record with the keys in USAGE_KEYS)."""


USAGE_KEYS = ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens", "cache_write_tokens",
              "cost_usd", "latency_s", "provider", "finish_reason", "reasoning_tokens_estimated")


def usage_record(**kw) -> dict:
    rec = dict.fromkeys(USAGE_KEYS, 0)
    rec.update(provider=None, finish_reason=None, reasoning_tokens_estimated=False)
    rec.update(kw)
    return rec


def post_json(url: str, body: dict, headers: dict | None = None, timeout: int = 600, tries: int = 4) -> tuple[dict, float]:
    """POST with retries on network errors and 429/5xx; client errors (4xx) fail at once."""
    err = ""
    for attempt in range(tries):
        t0 = time.monotonic()
        try:
            r = requests.post(url, headers={"Content-Type": "application/json", **(headers or {})}, json=body, timeout=timeout)
        except requests.RequestException as e:
            err = str(e)
        else:
            dt = time.monotonic() - t0
            if r.status_code == 200:
                j = r.json()
                if "error" not in j:
                    return j, dt
                err = json.dumps(j["error"])[:500]
            else:
                err = f"HTTP {r.status_code}: {r.text[:500]}"
            if 400 <= r.status_code < 500 and r.status_code != 429:
                raise RuntimeError(err)
        time.sleep(2 * 2 ** attempt)
    raise RuntimeError(err)


class OllamaAdapter:
    """Local Ollama via its OpenAI-compatible /v1/chat/completions. No account, no key, no cost.

    Ollama's usage block has no reasoning split: when the reply carries a separate `reasoning` text, its
    length / CHARS_PER_TOKEN is logged as reasoning_tokens with reasoning_tokens_estimated = True.
    """

    name = "ollama"

    def __init__(self, host: str | None = None):
        self.url = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/") + "/v1/chat/completions"

    def price(self, model: str) -> dict:
        return {"input": 0.0, "output": 0.0, "cache_read": 0.0}

    def chat(self, model, messages, *, effort, tools, max_tokens):
        body = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": False}
        if effort:
            body["reasoning_effort"] = effort
        if tools:
            body["tools"] = tools
        j, dt = post_json(self.url, body)
        msg = j["choices"][0]["message"]
        u = j.get("usage") or {}
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        return msg, usage_record(input_tokens=u.get("prompt_tokens", 0), output_tokens=u.get("completion_tokens", 0),
                                 reasoning_tokens=int(len(reasoning) / CHARS_PER_TOKEN), reasoning_tokens_estimated=bool(reasoning),
                                 latency_s=dt, provider="ollama", finish_reason=j["choices"][0].get("finish_reason"))


class MockAdapter:
    """Offline stand-in: returns the correct answer for each workflow and walks the agent through all four tools.
    Token counts are character-based estimates. Use it to test the runner and the calibration summary."""

    name = "mock"
    AGENT_PLAN = [("get_portfolio", "{}"), ("get_risk_metrics", None), ("get_news", None), ("get_exposures", '{"dimension": "sector"}')]

    def price(self, model: str) -> dict:
        return {"input": 0.0, "output": 0.0, "cache_read": 0.0}

    def chat(self, model, messages, *, effort, tools, max_tokens):
        n_in = int(sum(len(str(m.get("content") or "")) for m in messages) / CHARS_PER_TOKEN)
        if tools:
            step = sum(1 for m in messages if m["role"] == "assistant")
            if step < len(self.AGENT_PLAN):
                name, args = self.AGENT_PLAN[step]
                tickers = [h["ticker"] for h in ti.portfolio()["holdings"]]
                args = args or json.dumps({"tickers": tickers if name == "get_risk_metrics" else ti.agent_truth()["top_vol"]})
                msg = {"role": "assistant", "content": None,
                       "tool_calls": [{"id": f"call_{step}", "type": "function", "function": {"name": name, "arguments": args}}]}
            else:
                t = ti.agent_truth()
                msg = {"role": "assistant", "content": (f"Highest volatility: {', '.join(t['top_vol'])}. Negative news: "
                                                         f"{', '.join(t['neg_news_in_top_vol'])}. Largest sector: {t['top_sector']} "
                                                         f"({t['top_sector_pct']:.2f}%).")}
        else:
            msg = {"role": "assistant", "content": self._answer(messages)}
        n_out = int(len(json.dumps(msg)) / CHARS_PER_TOKEN)
        n_reason = {"low": 200, "high": 2000}.get(effort or "", 0)
        return msg, usage_record(input_tokens=n_in, output_tokens=n_out + n_reason, reasoning_tokens=n_reason,
                                 provider="mock", finish_reason="stop")

    @staticmethod
    def _answer(messages) -> str:
        text = messages[-1]["content"]
        if "SHORTLIST" in text:
            return "SHORTLIST: " + ", ".join(ti.screen_answer())
        if "ANSWER:" in text:
            a = ti.qa_answer()
            return f"ANSWER: {a['company']}; growth={a['growth']:.2f}%; op_margin={a['op_margin']:.2f}%"
        return "Key points\n...\nGuidance\n...\nRisks\n..."


ADAPTERS = {"mock": MockAdapter, "ollama": OllamaAdapter}


def get(name: str, **kw) -> Adapter:
    if name not in ADAPTERS:
        raise SystemExit(f"unknown adapter {name!r}; available: {', '.join(ADAPTERS)}")
    return ADAPTERS[name](**kw)
