"""Shared prompt construction and response parsing for LLM agents.

All three providers see an identical task specification and an identical rendering of the
observation, so differences on the leaderboard reflect the model, not prompt engineering.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..schema import Action, Observation

SYSTEM_PROMPT = """\
You are a portfolio manager competing in a trading benchmark run on a fully synthetic market.
Each period you are given a strictly point-in-time snapshot of the market (prices, fundamentals,
recent news and filings, all dated on or before the decision date) together with your current
book. You must output a TARGET PORTFOLIO for the next holding period.

Rules:
- Output the COMPLETE desired book as target weights per ticker. Any ticker you omit is treated
  as 0% (i.e. fully sold). The remainder (1 - sum of weights) is held as cash.
- Respect the constraints given in the observation (long_only, max_position_weight, gross_leverage).
- You are scored on risk-adjusted return AND on genuine stock-selection skill (your return is
  decomposed into market, sector and idiosyncratic components — only the idiosyncratic part
  reflects skill). Diversify or concentrate as you see fit, but avoid unintended factor bets.
- Trades execute at the NEXT session's open and pay transaction costs on turnover, so do not churn.

Respond with ONLY a JSON object, no prose outside it:
{"rationale": "<=3 sentences", "target_weights": {"TICKER": 0.05, ...}}"""


def _fmt_pct(x: Any) -> str:
    return f"{x*100:.1f}%" if isinstance(x, (int, float)) else "n/a"


def render_observation(obs: Observation, max_universe: int = 200) -> str:
    lines: list[str] = []
    lines.append(f"# Decision {obs.step}/{obs.total_steps} — {obs.scenario_name}")
    lines.append(f"Date: {obs.decision_date}  |  Holding period: ~{obs.horizon_days} trading days")
    lines.append(
        f"NAV: ${obs.nav:,.0f}  |  Cash: {_fmt_pct(obs.cash_weight)}  |  "
        f"Constraints: {json.dumps(obs.constraints)}"
    )
    if obs.positions:
        held = ", ".join(f"{t} {_fmt_pct(w)}" for t, w in sorted(obs.positions.items()))
        lines.append(f"Current holdings: {held}")
    else:
        lines.append("Current holdings: (all cash)")

    if obs.macro:
        macro = {k: v for k, v in obs.macro.items() if k not in ("time", "date", "scenario_id")}
        lines.append(f"\nMacro (as of date): {json.dumps(macro)[:500]}")

    lines.append("\n## Universe snapshot (adj price, trailing returns, valuation)")
    lines.append("ticker | sector | price | 1m | 3m | 12m | PE_ttm | held")
    for s in obs.universe[:max_universe]:
        lines.append(
            f"{s.ticker} | {s.sector} | {s.last_price} | {_fmt_pct(s.ret_1m)} | "
            f"{_fmt_pct(s.ret_3m)} | {_fmt_pct(s.ret_12m)} | "
            f"{s.pe_ttm if s.pe_ttm is not None else 'n/a'} | {_fmt_pct(s.held_weight)}"
        )
    if len(obs.universe) > max_universe:
        lines.append(f"... ({len(obs.universe) - max_universe} more tickers omitted)")

    if obs.holdings_detail:
        lines.append("\n## Fundamentals, financials, estimates & ownership (holdings + candidates)")
        for tk, f in obs.holdings_detail.items():
            e = f.latest_earnings.get("quarterly", {}) if isinstance(f.latest_earnings, dict) else {}
            surp = e.get("eps_surprise_pct")
            parts = [f"ttm={json.dumps(f.ttm)[:180]}"]
            if f.statements:
                parts.append(f"10Q={json.dumps(f.statements)[:220]}")
            if isinstance(surp, (int, float)):
                parts.append(f"eps_surprise={_fmt_pct(surp)}")
            if f.analyst:
                parts.append(f"analysts={json.dumps(f.analyst)[:200]}")
            if f.institutional:
                parts.append(f"13F={json.dumps(f.institutional)[:160]}")
            if f.insider:
                parts.append(f"insider={json.dumps(f.insider)[:120]}")
            if f.guidance:
                parts.append(f"guidance={json.dumps(f.guidance)[:120]}")
            lines.append(f"{tk}: " + " | ".join(parts))

    if obs.recent_docs:
        lines.append("\n## Recent news & filings (within window)")
        for d in obs.recent_docs:
            snippet = d["text"].replace("\n", " ")[:280]
            lines.append(f"[{d['date']}] {d['ticker']} ({d['doc_type']}): {d['title']} — {snippet}")

    lines.append('\nReturn ONLY the JSON object: {"rationale": "...", "target_weights": {...}}')
    return "\n".join(lines)


def parse_action(text: str) -> Action:
    """Extract the target-weight JSON from a model response, tolerant of code fences/prose."""
    raw = text
    # strip code fences
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = fenced + _brace_spans(text)
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "target_weights" in obj:
            tw = obj.get("target_weights") or {}
            if isinstance(tw, dict):
                weights = {}
                for k, v in tw.items():
                    try:
                        weights[str(k).upper()] = float(v)
                    except (TypeError, ValueError):
                        continue
                return Action(
                    target_weights=weights,
                    rationale=str(obj.get("rationale", ""))[:2000],
                    raw=raw[:8000],
                )
    # nothing parseable → hold cash
    return Action(target_weights={}, rationale="unparseable response; held cash", raw=raw[:8000])


def _brace_spans(text: str) -> list[str]:
    """Return top-level {...} substrings, largest first (handles nested braces)."""
    spans, stack, start = [], 0, -1
    for i, ch in enumerate(text):
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}":
            if stack > 0:
                stack -= 1
                if stack == 0 and start >= 0:
                    spans.append(text[start : i + 1])
    return sorted(spans, key=len, reverse=True)
