"""
LLM Adapter — Provider-agnostic interface for the Creativity Engine.

Swap between OpenAI, Anthropic, or any other provider by changing
the config. The rest of the engine only talks to LLMAdapter.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: dict | None = None


class LLMAdapter(ABC):
    """Base class — every provider implements this."""

    @abstractmethod
    async def generate(self, prompt: str, *, system: str | None = None, temperature: float = 0.7) -> LLMResponse:
        ...

    async def generate_json(self, prompt: str, *, system: str | None = None, temperature: float = 0.7) -> list | dict:
        """Generate and parse a JSON response. Retries once on parse failure."""
        resp = await self.generate(prompt, system=system, temperature=temperature)
        try:
            return _extract_json(resp.text)
        except (json.JSONDecodeError, ValueError):
            retry_prompt = (
                f"{prompt}\n\nIMPORTANT: Your previous response was not valid JSON. "
                "Return ONLY valid JSON, no markdown fences or extra text."
            )
            resp = await self.generate(retry_prompt, system=system, temperature=temperature)
            return _extract_json(resp.text)

    async def generate_float(self, prompt: str, *, system: str | None = None, temperature: float = 0.3) -> float:
        """Generate a single float value. Lower temperature for determinism."""
        resp = await self.generate(prompt, system=system, temperature=temperature)
        return _extract_float(resp.text)


class OpenAIAdapter(LLMAdapter):
    """Adapter for OpenAI-compatible APIs (OpenAI, Azure, local servers)."""

    def __init__(self, model: str = "gpt-4o-mini", api_key_env: str = "OPENAI_API_KEY", base_url: str | None = None):
        self.model = model
        self.api_key = os.environ.get(api_key_env, "")
        self.base_url = base_url
        self._client = None

    async def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def generate(self, prompt: str, *, system: str | None = None, temperature: float = 0.7) -> LLMResponse:
        client = await self._get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        choice = resp.choices[0]
        return LLMResponse(
            text=choice.message.content or "",
            model=resp.model,
            usage={
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            },
        )


class AnthropicAdapter(LLMAdapter):
    """Adapter for Anthropic Claude models."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key_env: str = "ANTHROPIC_API_KEY"):
        self.model = model
        self.api_key = os.environ.get(api_key_env, "")
        self._client = None

    async def _get_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def generate(self, prompt: str, *, system: str | None = None, temperature: float = 0.7) -> LLMResponse:
        client = await self._get_client()
        kwargs = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        resp = await client.messages.create(**kwargs)
        text = resp.content[0].text if resp.content else ""
        return LLMResponse(
            text=text,
            model=resp.model,
            usage={
                "input_tokens": resp.usage.input_tokens if resp.usage else 0,
                "output_tokens": resp.usage.output_tokens if resp.usage else 0,
            },
        )


def create_llm_adapter(provider: str, **kwargs) -> LLMAdapter:
    """Factory — create the right adapter from a provider string."""
    providers = {
        "openai": OpenAIAdapter,
        "anthropic": AnthropicAdapter,
    }
    cls = providers.get(provider.lower())
    if cls is None:
        raise ValueError(f"Unknown LLM provider '{provider}'. Available: {list(providers.keys())}")
    return cls(**kwargs)


# ── Parsing helpers ──────────────────────────────────────────────

def _extract_json(text: str) -> list | dict:
    """Extract JSON from LLM output that may contain markdown fences or preamble."""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    return json.loads(text)


def _extract_float(text: str) -> float:
    """Pull the first float-like number from LLM output."""
    match = re.search(r"(\d+\.?\d*)", text.strip())
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    raise ValueError(f"Could not extract float from: {text}")
