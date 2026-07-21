"""LLM-backed agents for Anthropic, OpenAI and Google Gemini.

Each provider SDK is imported lazily so you only need the package for the providers you run.
Every agent uses the identical shared prompt (see prompt.py); the only per-provider code is
the API call and usage accounting. Decoding is pinned (temperature 0 by default) for
reproducibility, and model ids are recorded verbatim in the trajectory.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from ..schema import Action, Observation
from .base import Agent
from .prompt import SYSTEM_PROMPT, parse_action, render_observation


class LLMAgent(Agent):
    provider = "llm"

    def __init__(
        self,
        name: str = "",
        model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_universe: int = 200,
        max_retries: int = 3,
        api_key_env: Optional[str] = None,
        **kw: Any,
    ):
        super().__init__(name or model, model=model, temperature=temperature, **kw)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_universe = max_universe
        self.max_retries = max_retries
        self.api_key_env = api_key_env
        self._client = None

    # subclasses implement these two ---------------------------------------------------
    def _client_init(self):  # pragma: no cover - provider specific
        raise NotImplementedError

    def _complete(self, system: str, user: str) -> tuple[str, dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError

    def client(self):
        if self._client is None:
            self._client = self._client_init()
        return self._client

    def decide(self, obs: Observation) -> Action:
        user = render_observation(obs, max_universe=self.max_universe)
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                text, usage = self._complete(SYSTEM_PROMPT, user)
                self.last_usage = usage
                action = parse_action(text)
                action.usage = usage  # type: ignore[attr-defined]
                if action.target_weights or "cash" in action.rationale.lower():
                    return action
                # empty & not explicitly cash → retry once more
                last_err = ValueError("empty target_weights")
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"{self.name}: all {self.max_retries} attempts failed: {last_err}")


class AnthropicAgent(LLMAgent):
    provider = "anthropic"

    def _client_init(self):
        import anthropic  # lazy

        key = os.environ.get(self.api_key_env or "ANTHROPIC_API_KEY")
        return anthropic.Anthropic(api_key=key)

    def _complete(self, system: str, user: str) -> tuple[str, dict[str, Any]]:
        resp = self.client().messages.create(
            model=self.model,
            system=system,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        usage = {
            "input_tokens": getattr(resp.usage, "input_tokens", None),
            "output_tokens": getattr(resp.usage, "output_tokens", None),
        }
        return text, usage


class OpenAIAgent(LLMAgent):
    provider = "openai"

    def _client_init(self):
        import openai  # lazy

        key = os.environ.get(self.api_key_env or "OPENAI_API_KEY")
        return openai.OpenAI(api_key=key)

    def _is_reasoning(self) -> bool:
        m = (self.model or "").lower()
        return m.startswith("o1") or m.startswith("o3") or m.startswith("o4") or m.startswith("gpt-5")

    def _complete(self, system: str, user: str) -> tuple[str, dict[str, Any]]:
        # Chat Completions is the most broadly compatible surface across GPT-5.x snapshots.
        params: dict[str, Any] = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self._is_reasoning():
            # GPT-5 / o-series reasoning models: temperature is fixed at 1 (omit it), and the
            # token budget is passed as max_completion_tokens (must also cover reasoning tokens).
            params["max_completion_tokens"] = max(self.max_tokens, 4096)
            eff = self.params.get("reasoning_effort")
            if eff:
                params["reasoning_effort"] = eff
        else:
            params["temperature"] = self.temperature
            params["max_tokens"] = self.max_tokens
        resp = self.client().chat.completions.create(**params)
        text = resp.choices[0].message.content or ""
        u = resp.usage
        usage = {
            "input_tokens": getattr(u, "prompt_tokens", None),
            "output_tokens": getattr(u, "completion_tokens", None),
            "reasoning_tokens": getattr(
                getattr(u, "completion_tokens_details", None), "reasoning_tokens", None
            ),
        }
        return text, usage


class XAIAgent(OpenAIAgent):
    """xAI Grok. The xAI API is OpenAI-compatible, so we reuse OpenAIAgent's completion path
    and only swap the client (base_url + XAI_API_KEY). Grok accepts standard chat params
    (temperature + max_tokens + json_object), so it takes the non-reasoning branch."""

    provider = "xai"

    def _client_init(self):
        import openai  # lazy

        key = os.environ.get(self.api_key_env or "XAI_API_KEY")
        return openai.OpenAI(api_key=key, base_url="https://api.x.ai/v1")

    def _is_reasoning(self) -> bool:
        return False


class GeminiAgent(LLMAgent):
    provider = "gemini"

    def _client_init(self):
        from google import genai  # lazy (google-genai)

        key = os.environ.get(self.api_key_env or "GEMINI_API_KEY")
        return genai.Client(api_key=key)

    def _complete(self, system: str, user: str) -> tuple[str, dict[str, Any]]:
        from google.genai import types  # lazy

        resp = self.client().models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                response_mime_type="application/json",
            ),
        )
        text = resp.text or ""
        um = getattr(resp, "usage_metadata", None)
        usage = {
            "input_tokens": getattr(um, "prompt_token_count", None),
            "output_tokens": getattr(um, "candidates_token_count", None),
        }
        return text, usage


class FileAgent(LLMAgent):
    """External-in-the-loop agent: writes each decision's prompt to a queue directory and
    blocks until an answer file appears, then parses it with the shared parser.

    This is how we benchmark a model that has no API binding available in this process - e.g.
    Claude driven by Claude Code subagents with no API key. An external orchestrator polls
    ``<queue_dir>/pending/<scenario>__<step>.txt`` (the exact rendered observation the API
    agents see) and writes the model's JSON to ``<queue_dir>/decisions/<scenario>__<step>.txt``.
    Each decision is an independent file with no shared state, so the run is stateless per
    decision and uses the identical prompt + parser as the API-backed agents - comparable by
    construction. The fixed system prompt the responder must use is ``prompt.SYSTEM_PROMPT``.
    """

    provider = "file"

    def _client_init(self):
        return object()

    def decide(self, obs: Observation) -> Action:
        import os

        qd = self.params.get("queue_dir", "claude_queue")
        poll = float(self.params.get("poll_secs", 3.0))
        timeout = float(self.params.get("timeout_secs", 21600))
        pend_dir = os.path.join(qd, "pending")
        dec_dir = os.path.join(qd, "decisions")
        os.makedirs(pend_dir, exist_ok=True)
        os.makedirs(dec_dir, exist_ok=True)
        key = f"{obs.scenario_id}__{obs.step:03d}"
        pend = os.path.join(pend_dir, key + ".txt")
        ans = os.path.join(dec_dir, key + ".txt")

        prompt = render_observation(obs, max_universe=self.max_universe)
        tmp = pend + ".tmp"
        with open(tmp, "w") as f:
            f.write(prompt)
        os.replace(tmp, pend)  # atomic publish so the orchestrator never reads a partial file

        waited = 0.0
        while not os.path.exists(ans):
            time.sleep(poll)
            waited += poll
            if waited > timeout:
                raise RuntimeError(f"{self.name}: no decision for {key} after {timeout:.0f}s")
        text = open(ans).read()
        action = parse_action(text)
        action.usage = {"input_tokens": None, "output_tokens": None}  # type: ignore[attr-defined]
        try:
            os.remove(pend)
        except OSError:
            pass
        return action


class MockLLM(LLMAgent):
    """Offline agent that exercises the full prompt+parse path without any API call.

    It deterministically tilts toward positive-3m-momentum names, emitting exactly the JSON an
    LLM is asked for. Used by tests and for dry-runs so the pipeline can be validated for free.
    """

    provider = "mock"

    def __init__(self, name: str = "mock_llm", top_k: int = 8, **kw):
        super().__init__(name=name, model="mock-llm-v1", **kw)
        self.top_k = top_k

    def _client_init(self):
        return object()

    def _complete(self, system: str, user: str) -> tuple[str, dict[str, Any]]:
        import json
        import re

        rows = re.findall(r"^([A-Z][A-Z0-9.\-]{0,6}) \| .* \| (-?\d+\.\d)% \|", user, re.M)
        ranked = sorted(rows, key=lambda r: float(r[1]), reverse=True)[: self.top_k]
        w = round(min(0.1, 1.0 / max(1, len(ranked))), 4)
        weights = {t: w for t, _ in ranked}
        payload = {"rationale": "mock momentum tilt", "target_weights": weights}
        return json.dumps(payload), {"input_tokens": len(user) // 4, "output_tokens": 20}
