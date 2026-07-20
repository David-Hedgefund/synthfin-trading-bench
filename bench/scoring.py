"""Scoring: turn a trajectory into performance and *skill* metrics.

Three families of metrics:

1. Net performance from the realized NAV series — total/annualized return, volatility, Sharpe,
   Sortino, max drawdown, Calmar, turnover and cost drag. These are what a real book earns and
   are computed only from executed, cost-charged trades in split-adjusted price space.

2. Skill vs the market (the headline). We regress the book's realized daily returns on the
   market proxy's realized returns (CAPM): this yields the market beta, the annualized Jensen's
   ``alpha_ann``, and the ``appraisal_ratio`` = alpha / idiosyncratic-vol (annualized) — the
   standard measure of risk-adjusted stock-selection skill net of market exposure. It uses only
   realized, tradeable prices, so it is fully defensible.

3. Generator attribution (diagnostic, approximate). The synthetic generator labels each bar's
   return with latent drivers (market/sector/idiosyncratic/event/…). Weighting by holdings gives
   an *intended-driver* breakdown of the book's return. This is illustrative, not an exact
   variance decomposition — the labeled drivers leave an unexplained ``residual`` — so it
   supplements, and never replaces, the regression-based skill metrics above.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from .corpus import ScenarioData

ANN = 252.0

# How the generator's 13 latent return drivers roll up into interpretable buckets.
# `selection` is name-specific — idiosyncratic drift plus event/jump reactions a model can earn
# by reading the (synthetic) news and fundamentals correctly. `market`/`sector` are systematic
# exposures you get from beta, not skill. Anything the components miss vs the realized price move
# lands in `residual` (dominated by 2-decimal price rounding).
FACTOR_GROUPS = {
    "market": ["market", "stress_beta", "drift", "scenario_tape", "stress_packet", "tail_cluster"],
    "sector": ["sector"],
    "selection": [
        "idiosyncratic", "mean_reversion", "jump", "aftershock", "event",
        "self_exciting_jump_cluster",
    ],
}
_ALL_FACTORS = [c for cs in FACTOR_GROUPS.values() for c in cs]


def bar_buckets(bar, realized: float) -> dict[str, float]:
    """Return {market, sector, selection, residual} for one bar's realized return.

    Uses the full 13-way ``components`` when present; otherwise falls back to the four coarse
    top-level fields. ``residual`` reconciles the buckets to the realized price return exactly.
    """
    if bar.components:
        b = {g: sum(bar.components.get(c, 0.0) for c in cs) for g, cs in FACTOR_GROUPS.items()}
        explained = sum(bar.components.get(c, 0.0) for c in _ALL_FACTORS)
    else:
        b = {
            "market": bar.market_return,
            "sector": bar.sector_return,
            "selection": bar.idiosyncratic_return + bar.event_return,
        }
        explained = b["market"] + b["sector"] + b["selection"]
    b["residual"] = realized - explained
    return b


def _nav_returns(nav_series: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    dates = [r["date"] for r in nav_series]
    navs = np.array([r["nav"] for r in nav_series], dtype=float)
    rets = np.zeros(len(navs))
    rets[1:] = navs[1:] / navs[:-1] - 1.0
    return dates, rets


def _series_returns(scn: ScenarioData, ticker: str, dates: list[str]) -> Optional[np.ndarray]:
    rets = np.zeros(len(dates) - 1)
    for k in range(1, len(dates)):
        b0 = scn.last_bar_on_or_before(ticker, dates[k - 1])
        b1 = scn.last_bar_on_or_before(ticker, dates[k])
        if not b0 or not b1 or not b0.adj_close:
            return None
        rets[k - 1] = b1.adj_close / b0.adj_close - 1.0
    return rets


def _ew_universe_returns(scn: ScenarioData, dates: list[str]) -> np.ndarray:
    """Equal-weight daily return of the whole priced universe — a self-contained market factor
    used when no index ETF is in the corpus. This is a legitimate broad-market benchmark."""
    acc = np.zeros(len(dates) - 1)
    cnt = np.zeros(len(dates) - 1)
    for tk in scn.priced_tickers:
        r = _series_returns(scn, tk, dates)
        if r is not None:
            acc += r
            cnt += 1
    return np.divide(acc, cnt, out=np.zeros_like(acc), where=cnt > 0)


def _proxy_daily_returns(scn: ScenarioData, dates: list[str]) -> tuple[Optional[str], np.ndarray]:
    """Realized daily returns of the market proxy, aligned to ``dates[1:]``.

    Prefers a real index ETF (SPY→QQQ→DIA); if the corpus has none (e.g. an operating-only
    1000-name universe), falls back to the equal-weight universe index so the CAPM regression is
    always well-defined and comparable across corpora.
    """
    if len(dates) < 2:
        return None, np.zeros(0)
    for proxy in ("SPY", "QQQ", "DIA"):
        if proxy in scn.priced_tickers:
            rets = _series_returns(scn, proxy, dates)
            if rets is not None:
                return proxy, rets
    return "EW_UNIVERSE", _ew_universe_returns(scn, dates)


def _capm(r_p: np.ndarray, r_m: np.ndarray) -> dict[str, float]:
    """Regress portfolio returns on market returns → beta, annualized alpha, appraisal ratio."""
    if r_p.size < 3 or r_m.size != r_p.size or r_m.var() == 0:
        return {"beta_market": 0.0, "alpha_ann": 0.0, "appraisal_ratio": 0.0}
    beta = float(np.cov(r_p, r_m, ddof=1)[0, 1] / np.var(r_m, ddof=1))
    alpha_daily = float(r_p.mean() - beta * r_m.mean())
    resid = r_p - (alpha_daily + beta * r_m)
    resid_std = float(resid.std(ddof=1))
    appraisal = float(alpha_daily / resid_std * math.sqrt(ANN)) if resid_std > 0 else 0.0
    return {
        "beta_market": beta,
        "alpha_ann": alpha_daily * ANN,
        "appraisal_ratio": appraisal,
    }


def _max_drawdown(navs: np.ndarray) -> float:
    if navs.size == 0:
        return 0.0
    peak = np.maximum.accumulate(navs)
    dd = navs / peak - 1.0
    return float(dd.min())


def _units_timeline(rebalances: list[dict[str, Any]]) -> list[tuple[str, dict[str, float]]]:
    tl = []
    for rb in rebalances:
        ed = rb.get("exec_date")
        if ed:
            tl.append((ed, rb.get("units_after", {})))
    tl.sort(key=lambda x: x[0])
    return tl


def _units_in_effect(timeline, date: str) -> dict[str, float]:
    cur: dict[str, float] = {}
    for ed, units in timeline:
        if ed <= date:
            cur = units
        else:
            break
    return cur


def score_trajectory(traj: dict[str, Any], scn: ScenarioData) -> dict[str, Any]:
    nav_series = traj.get("nav_series", [])
    if len(nav_series) < 3:
        return {"error": "nav series too short", "n_days": len(nav_series)}

    dates, rets = _nav_returns(nav_series)
    navs = np.array([r["nav"] for r in nav_series], dtype=float)
    active = rets[1:]

    total_return = float(navs[-1] / navs[0] - 1.0)
    n_days = len(active)
    years = n_days / ANN if n_days else 0.0
    cagr = float((navs[-1] / navs[0]) ** (1 / years) - 1.0) if years > 0 else 0.0
    vol = float(active.std(ddof=1) * math.sqrt(ANN)) if n_days > 1 else 0.0
    mean_daily = float(active.mean()) if n_days else 0.0
    sharpe = float(mean_daily / active.std(ddof=1) * math.sqrt(ANN)) if active.std(ddof=1) > 0 else 0.0
    downside = active[active < 0]
    dstd = float(downside.std(ddof=1)) if downside.size > 1 else 0.0
    sortino = float(mean_daily / dstd * math.sqrt(ANN)) if dstd > 0 else 0.0
    mdd = _max_drawdown(navs)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else 0.0

    turnovers = [rb.get("turnover", 0.0) for rb in traj.get("rebalances", [])]
    costs = [rb.get("cost", 0.0) for rb in traj.get("rebalances", [])]
    avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0
    cost_drag = float(sum(costs) / navs[0]) if navs[0] else 0.0

    # --- factor attribution from the generator's latent return decomposition -------
    # Portfolio daily return in each bucket = sum_i w_i * bucket_return_i. Buckets sum (with
    # residual) to the realized gross book return each day, so the split is exact per day.
    timeline = _units_timeline(traj.get("rebalances", []))
    comp = {k: [] for k in ("market", "sector", "selection", "residual", "gross")}
    for k in range(1, len(dates)):
        d_prev, d = dates[k - 1], dates[k]
        units = _units_in_effect(timeline, d)
        nav_prev = navs[k - 1]
        if not units or nav_prev <= 0:
            for key in comp:
                comp[key].append(0.0)
            continue
        acc = {"market": 0.0, "sector": 0.0, "selection": 0.0, "residual": 0.0, "gross": 0.0}
        for tk, u in units.items():
            bar = scn.bar(tk, d)
            prev_bar = scn.last_bar_on_or_before(tk, d_prev)
            if bar is None or prev_bar is None or not prev_bar.adj_close:
                continue
            w = (u * prev_bar.adj_close) / nav_prev
            realized = bar.adj_close / prev_bar.adj_close - 1.0
            buckets = bar_buckets(bar, realized)
            for key in ("market", "sector", "selection", "residual"):
                acc[key] += w * buckets[key]
            acc["gross"] += w * realized
        for key in comp:
            comp[key].append(acc[key])

    arr = {k: np.array(v, dtype=float) for k, v in comp.items()}
    selection = arr["selection"]
    sel_mean = float(selection.mean()) if selection.size else 0.0
    sel_std = float(selection.std(ddof=1)) if selection.size > 1 else 0.0
    selection_ir = float(sel_mean / sel_std * math.sqrt(ANN)) if sel_std > 0 else 0.0

    def cum(x: np.ndarray) -> float:
        return float(np.prod(1.0 + x) - 1.0) if x.size else 0.0

    # Diagnostic generator attribution (approximate — see module docstring).
    gen_attribution = {
        "gen_market_contribution": float(arr["market"].sum()),
        "gen_sector_contribution": float(arr["sector"].sum()),
        "gen_selection_contribution": float(selection.sum()),
        "gen_residual_contribution": float(arr["residual"].sum()),
        "gen_selection_ir": selection_ir,
    }

    # --- headline skill: CAPM regression vs the market proxy (realized returns) ------
    proxy, r_m = _proxy_daily_returns(scn, dates)
    capm = _capm(active, r_m) if r_m.size == active.size else {
        "beta_market": 0.0, "alpha_ann": 0.0, "appraisal_ratio": 0.0
    }
    # benchmark total return over the window = cumulative proxy return (consistent with CAPM)
    bench = float(np.prod(1.0 + r_m) - 1.0) if proxy and r_m.size else None

    return {
        "scenario_id": traj.get("scenario_id"),
        "scenario_slug": traj.get("scenario_slug"),
        "agent_name": traj.get("agent_name"),
        "n_days": n_days,
        "n_decisions": traj.get("n_decisions"),
        "n_errors": len(traj.get("errors", [])),
        # net performance
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "avg_turnover": avg_turnover,
        "cost_drag": cost_drag,
        "gross_book_return": cum(arr["gross"]),
        # skill vs market (headline)
        "market_proxy": proxy,
        "beta_market": capm["beta_market"],
        "alpha_ann": capm["alpha_ann"],
        "appraisal_ratio": capm["appraisal_ratio"],
        # benchmark-relative
        "benchmark_return": bench,
        "excess_return": (total_return - bench) if bench is not None else None,
        # generator attribution (diagnostic, approximate)
        **gen_attribution,
    }


def aggregate(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Average a set of per-scenario scores for one agent into a leaderboard row."""
    if not scores:
        return {}
    keys = [
        "total_return", "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown",
        "calmar", "avg_turnover", "cost_drag", "excess_return",
        "beta_market", "alpha_ann", "appraisal_ratio",
        "gen_selection_contribution",
    ]
    out: dict[str, Any] = {
        "agent_name": scores[0].get("agent_name"),
        "n_scenarios": len(scores),
        "n_errors": int(sum(s.get("n_errors", 0) for s in scores)),
    }
    for k in keys:
        vals = [s[k] for s in scores if isinstance(s.get(k), (int, float))]
        out[f"mean_{k}"] = float(np.mean(vals)) if vals else None
        if k in ("sharpe", "appraisal_ratio", "total_return", "alpha_ann"):
            out[f"std_{k}"] = float(np.std(vals)) if len(vals) > 1 else 0.0
    # win rate vs benchmark
    wins = [1 for s in scores if isinstance(s.get("excess_return"), (int, float)) and s["excess_return"] > 0]
    out["benchmark_win_rate"] = len(wins) / len(scores)
    return out
