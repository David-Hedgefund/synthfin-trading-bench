"""Agent registry + factory. Config lists agents by ``provider`` and the factory builds them."""

from __future__ import annotations

from typing import Any

from .base import Agent
from .baselines import BASELINES, BuyHoldEqual, EqualWeightRebalance, Momentum, RandomAgent
from .llm import AnthropicAgent, GeminiAgent, LLMAgent, MockLLM, OpenAIAgent

_PROVIDERS = {
    "anthropic": AnthropicAgent,
    "openai": OpenAIAgent,
    "gemini": GeminiAgent,
    "mock": MockLLM,
}


def build_agent(spec: dict[str, Any]) -> Agent:
    """Build an agent from a config dict, e.g.::

        {"provider": "anthropic", "name": "claude-opus", "model": "claude-opus-4-8"}
        {"provider": "baseline", "strategy": "momentum", "top_frac": 0.1}
    """
    spec = dict(spec)
    provider = spec.pop("provider")
    if provider == "baseline":
        strategy = spec.pop("strategy")
        cls = BASELINES[strategy]
        return cls(**spec)
    if provider in _PROVIDERS:
        return _PROVIDERS[provider](**spec)
    raise ValueError(f"unknown provider: {provider!r}")


__all__ = [
    "Agent",
    "LLMAgent",
    "AnthropicAgent",
    "OpenAIAgent",
    "GeminiAgent",
    "MockLLM",
    "BuyHoldEqual",
    "EqualWeightRebalance",
    "Momentum",
    "RandomAgent",
    "build_agent",
]
