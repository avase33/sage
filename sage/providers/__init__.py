"""LLM providers with a streaming interface."""

from __future__ import annotations

from ..config import Settings
from .base import Provider
from .mock import MockProvider


def get_provider(name: str | None = None, **kwargs) -> Provider:
    name = (name or Settings.from_env().provider or "mock").lower()
    if name in ("mock", "offline"):
        return MockProvider()
    if name == "anthropic":
        from .anthropic import AnthropicProvider

        return AnthropicProvider(**kwargs)
    if name == "openai":
        from .openai import OpenAIProvider

        return OpenAIProvider(**kwargs)
    raise ValueError(f"Unknown provider: {name!r}")


__all__ = ["Provider", "MockProvider", "get_provider"]
