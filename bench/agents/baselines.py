"""Non-LLM reference strategies. A leaderboard is meaningless without them: they tell you
whether an LLM is adding skill or just riding beta. All are deterministic given the seed."""

from __future__ import annotations

import random

from ..schema import Action, Observation
from .base import Agent


def _cap(weights: dict[str, float], cap: float, gross: float) -> dict[str, float]:
    weights = {t: min(w, cap) for t, w in weights.items() if w > 0}
    s = sum(weights.values())
    if s > gross and s > 0:
        weights = {t: w * gross / s for t, w in weights.items()}
    return weights


class BuyHoldEqual(Agent):
    """Equal-weight the whole universe once, then hold (no rebalancing turnover)."""

    name = "baseline_buyhold"
    provider = "baseline"
    model = "buyhold_equal"

    def decide(self, obs: Observation) -> Action:
        if obs.step == 1:
            n = len(obs.universe) or 1
            cap = obs.constraints["max_position_weight"]
            gross = obs.constraints["gross_leverage"]
            w = min(1.0 / n, cap)
            weights = _cap({s.ticker: w for s in obs.universe}, cap, gross)
            return Action(target_weights=weights, rationale="equal-weight, buy and hold")
        # hold: re-issue current weights so the simulator does (almost) nothing
        return Action(target_weights=dict(obs.positions), rationale="hold")


class EqualWeightRebalance(Agent):
    """Equal-weight the universe and rebalance back to equal every period."""

    name = "baseline_ew_rebal"
    provider = "baseline"
    model = "equal_weight_rebalance"

    def decide(self, obs: Observation) -> Action:
        n = len(obs.universe) or 1
        cap = obs.constraints["max_position_weight"]
        gross = obs.constraints["gross_leverage"]
        w = min(1.0 / n, cap)
        return Action(
            target_weights=_cap({s.ticker: w for s in obs.universe}, cap, gross),
            rationale="equal-weight rebalance",
        )


class Momentum(Agent):
    """Long the top-``k`` names by trailing 3m return, equal-weighted."""

    name = "baseline_momentum"
    provider = "baseline"
    model = "momentum_top_decile"

    def __init__(self, name: str = "", top_k: int = 0, top_frac: float = 0.1, **kw):
        super().__init__(name, top_k=top_k, top_frac=top_frac, **kw)
        self.top_k = top_k
        self.top_frac = top_frac

    def decide(self, obs: Observation) -> Action:
        ranked = sorted(obs.universe, key=lambda s: (s.ret_3m or -9.9), reverse=True)
        k = self.top_k or max(1, int(len(ranked) * self.top_frac))
        picks = ranked[:k]
        cap = obs.constraints["max_position_weight"]
        gross = obs.constraints["gross_leverage"]
        w = min(1.0 / max(1, len(picks)), cap)
        return Action(
            target_weights=_cap({s.ticker: w for s in picks}, cap, gross),
            rationale=f"top-{k} 3m momentum",
        )


class RandomAgent(Agent):
    """Random equal-weight basket. Seeded per (scenario, step) for reproducibility."""

    name = "baseline_random"
    provider = "baseline"
    model = "random_basket"

    def __init__(self, name: str = "", seed: int = 12345, basket: int = 10, **kw):
        super().__init__(name, seed=seed, basket=basket, **kw)
        self.seed = seed
        self.basket = basket

    def decide(self, obs: Observation) -> Action:
        rng = random.Random(f"{self.seed}:{obs.scenario_id}:{obs.step}")
        tickers = [s.ticker for s in obs.universe]
        k = min(self.basket, len(tickers))
        picks = rng.sample(tickers, k) if k else []
        cap = obs.constraints["max_position_weight"]
        gross = obs.constraints["gross_leverage"]
        w = min(1.0 / max(1, k), cap)
        return Action(
            target_weights=_cap({t: w for t in picks}, cap, gross),
            rationale="random basket",
        )


BASELINES = {
    "buyhold": BuyHoldEqual,
    "ew_rebalance": EqualWeightRebalance,
    "momentum": Momentum,
    "random": RandomAgent,
}
