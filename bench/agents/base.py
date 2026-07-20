"""Agent interface. An agent maps a point-in-time Observation to a target-weight Action."""

from __future__ import annotations

from typing import Any, Optional

from ..schema import Action, Observation


class Agent:
    """Base class. Subclasses implement :meth:`decide`.

    ``meta`` describes the agent for the trajectory/leaderboard (provider, model, params).
    """

    name: str = "agent"
    provider: str = "none"
    model: str = ""

    def __init__(self, name: str = "", **kwargs: Any):
        if name:
            self.name = name
        self.params = kwargs
        self.last_usage: Optional[dict[str, Any]] = None

    def decide(self, obs: Observation) -> Action:  # pragma: no cover - interface
        raise NotImplementedError

    def meta(self) -> dict[str, Any]:
        return {
            "agent_name": self.name,
            "provider": self.provider,
            "model": self.model,
            "params": self.params,
        }
