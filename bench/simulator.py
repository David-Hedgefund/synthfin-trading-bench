"""Sequential portfolio simulator with strict no-lookahead execution.

Protocol per decision day ``t`` (t is a trading day at index i in the scenario calendar):
  1. Value the current book at t's close and build the point-in-time observation.
  2. Ask the agent for a target portfolio (weights).
  3. Validate/clip the target against the portfolio constraints.
  4. Execute the resulting trades at the *next* session's adjusted open (t+1), charging
     ``cost_bps`` on turnover. This one-bar delay is what prevents same-bar lookahead.
Between decisions the book is held and marked to market daily at adjusted close.

Trading happens in split-adjusted price space, so corporate actions never distort share
counts or P&L. The simulator emits a compact trajectory; all metrics are computed later by
scoring.py from that trajectory plus the scenario bars.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import RunConfig
from .corpus import ScenarioData
from .observation import build_observation
from .schema import Action, asdict


@dataclass
class PortfolioState:
    cash: float
    units: dict[str, float] = field(default_factory=dict)  # ticker -> adjusted-space units

    def market_value(self, price_of) -> float:
        mv = self.cash
        for tk, u in self.units.items():
            p = price_of(tk)
            if p is not None:
                mv += u * p
        return mv

    def weights(self, price_of, nav: float) -> dict[str, float]:
        if nav <= 0:
            return {}
        out = {}
        for tk, u in self.units.items():
            p = price_of(tk)
            if p is not None and abs(u) > 1e-12:
                out[tk] = (u * p) / nav
        return out


def validate_target(
    target: dict[str, float], priced: set[str], cfg: RunConfig
) -> dict[str, float]:
    """Clip an agent's target book to the constraints. Never raises on bad model output."""
    clean: dict[str, float] = {}
    for tk, w in target.items():
        if tk not in priced:
            continue
        try:
            w = float(w)
        except (TypeError, ValueError):
            continue
        if w != w:  # NaN
            continue
        if cfg.long_only and w < 0:
            w = 0.0
        # cap absolute size of any single position
        cap = cfg.max_position_weight
        w = max(-cap, min(cap, w))
        if abs(w) > 1e-9:
            clean[tk] = w
    # enforce gross leverage
    gross = sum(abs(w) for w in clean.values())
    if gross > cfg.gross_leverage and gross > 0:
        scale = cfg.gross_leverage / gross
        clean = {tk: w * scale for tk, w in clean.items()}
    return clean


class Simulator:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg

    def _decision_indices(self, n_days: int) -> list[int]:
        cfg = self.cfg
        idxs = list(range(cfg.warmup_days, n_days - 1, cfg.rebalance_every_days))
        if cfg.max_decisions and len(idxs) > cfg.max_decisions:
            idxs = idxs[: cfg.max_decisions]
        return idxs

    def run(self, agent, scn: ScenarioData, *, agent_meta: dict[str, Any]) -> dict[str, Any]:
        cfg = self.cfg
        days = scn.trading_days
        priced = set(scn.priced_tickers)
        if len(days) < cfg.warmup_days + 2:
            raise ValueError(f"scenario {scn.meta.slug} too short for warmup={cfg.warmup_days}")

        decision_idx = set(self._decision_indices(len(days)))
        state = PortfolioState(cash=cfg.initial_nav)
        pending: Optional[dict[str, float]] = None  # target to execute at today's open
        pending_meta: dict[str, Any] = {}

        rebalances: list[dict[str, Any]] = []
        nav_series: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        first_i = min(decision_idx)
        for i in range(first_i, len(days)):
            date = days[i]

            def adj_close_of(tk: str, d: str = date):
                b = scn.bar(tk, d) or scn.last_bar_on_or_before(tk, d)
                return b.adj_close if b else None

            # (a) execute any target scheduled for today's open
            if pending is not None:
                nav_exec = state.market_value(lambda tk: _adj_open_of(scn, tk, date))
                turnover, cost = _execute(state, pending, scn, date, nav_exec, cfg)
                rebalances.append(
                    {
                        **pending_meta,
                        "exec_date": date,
                        "nav_at_exec": round(nav_exec, 2),
                        "target_weights": {k: round(v, 4) for k, v in pending.items()},
                        "units_after": {k: round(v, 6) for k, v in state.units.items()},
                        "cash_after": round(state.cash, 2),
                        "turnover": round(turnover, 4),
                        "cost": round(cost, 2),
                    }
                )
                pending = None
                pending_meta = {}

            # (b) mark to market at today's close
            nav = state.market_value(adj_close_of)
            nav_series.append({"date": date, "nav": round(nav, 2)})

            # (c) if today is a decision day, ask the agent (executes tomorrow)
            if i in decision_idx and i < len(days) - 1:
                weights_now = state.weights(adj_close_of, nav)
                cash_w = 1.0 - sum(weights_now.values())
                horizon = (
                    days[min(i + cfg.rebalance_every_days, len(days) - 1)]
                )
                obs = build_observation(
                    scn,
                    decision_date=date,
                    day_index=i,
                    step=len(rebalances) + 1,
                    total_steps=len(decision_idx),
                    horizon_days=cfg.rebalance_every_days,
                    nav=nav,
                    cash_weight=cash_w,
                    positions=weights_now,
                    config=cfg,
                )
                t0 = time.time()
                try:
                    action: Action = agent.decide(obs)
                except Exception as exc:  # noqa: BLE001 — a bad agent must not kill the run
                    errors.append({"date": date, "error": f"{type(exc).__name__}: {exc}"})
                    action = Action(target_weights=weights_now, rationale=f"agent error: {exc}")
                latency = time.time() - t0
                target = validate_target(action.target_weights, priced, cfg)
                pending = target
                pending_meta = {
                    "decision_date": date,
                    "rationale": (action.rationale or "")[:2000],
                    "latency_s": round(latency, 3),
                    "usage": getattr(action, "usage", None) or agent_meta.get("_last_usage"),
                    "raw_target": {k: round(float(v), 4) for k, v in action.target_weights.items()
                                   if _is_num(v)},
                }

        return {
            "scenario_id": scn.meta.id,
            "scenario_slug": scn.meta.slug,
            "scenario_name": scn.meta.name,
            "seed": scn.meta.seed,
            **agent_meta,
            "config": cfg.to_dict(),
            "start_date": nav_series[0]["date"] if nav_series else None,
            "end_date": nav_series[-1]["date"] if nav_series else None,
            "n_decisions": len(rebalances),
            "rebalances": rebalances,
            "nav_series": nav_series,
            "errors": errors,
        }


def _adj_open_of(scn: ScenarioData, tk: str, date: str) -> Optional[float]:
    b = scn.bar(tk, date)
    if b:
        return b.adj_open
    b = scn.last_bar_on_or_before(tk, date)
    return b.adj_close if b else None


def _execute(
    state: PortfolioState,
    target: dict[str, float],
    scn: ScenarioData,
    exec_date: str,
    nav: float,
    cfg: RunConfig,
) -> tuple[float, float]:
    """Move the book toward ``target`` at ``exec_date`` open. Returns (turnover_frac, cost)."""
    turnover_dollars = 0.0
    new_units = dict(state.units)
    for tk in set(list(target.keys()) + list(state.units.keys())):
        price = _adj_open_of(scn, tk, exec_date)
        if not price:
            continue  # cannot fill without a price; leave position unchanged
        desired_dollars = target.get(tk, 0.0) * nav
        desired_units = desired_dollars / price
        cur_units = state.units.get(tk, 0.0)
        trade_units = desired_units - cur_units
        if abs(trade_units) < 1e-12:
            continue
        turnover_dollars += abs(trade_units) * price
        state.cash -= trade_units * price
        new_units[tk] = desired_units
    state.units = {tk: u for tk, u in new_units.items() if abs(u) > 1e-9}
    cost = turnover_dollars * cfg.cost_bps / 1e4
    state.cash -= cost
    return (turnover_dollars / nav if nav else 0.0), cost


def _is_num(x: Any) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False
