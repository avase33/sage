"""Runtime configuration, resolved from the environment.

Sage is offline-first: with nothing configured it uses the deterministic mock
model and the dependency-free hashing embedder, so the whole app runs and tests
pass without network access or API keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    provider: str = "mock"          # mock | anthropic | openai
    model: str = ""
    embedder: str = "hashing"       # hashing | openai
    embedding_dim: int = 512
    db_path: str = "sage.db"
    max_tool_steps: int = 6
    rag_top_k: int = 4
    temperature: float = 0.7
    request_timeout: int = 60

    @classmethod
    def from_env(cls) -> "Settings":
        provider = os.environ.get("SAGE_PROVIDER")
        if not provider:
            if os.environ.get("ANTHROPIC_API_KEY"):
                provider = "anthropic"
            elif os.environ.get("OPENAI_API_KEY"):
                provider = "openai"
            else:
                provider = "mock"
        embedder = os.environ.get("SAGE_EMBEDDER") or (
            "openai" if os.environ.get("OPENAI_API_KEY") and provider == "openai" else "hashing"
        )
        return cls(
            provider=provider,
            model=os.environ.get("SAGE_MODEL", ""),
            embedder=embedder,
            embedding_dim=int(os.environ.get("SAGE_EMBED_DIM", "512")),
            db_path=os.environ.get("SAGE_DB", "sage.db"),
            max_tool_steps=int(os.environ.get("SAGE_MAX_STEPS", "6")),
            rag_top_k=int(os.environ.get("SAGE_RAG_TOPK", "4")),
            temperature=float(os.environ.get("SAGE_TEMPERATURE", "0.7")),
        )
