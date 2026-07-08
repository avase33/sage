"""Text embeddings for semantic search / RAG.

The default :class:`HashingEmbedder` is dependency-free and deterministic: it maps
text to a fixed-dimensional vector via the hashing trick (sublinear term
frequency + L2 normalisation), so cosine similarity gives sensible relevance
ranking with no model download and no network. Swap in :class:`OpenAIEmbedder`
for real neural embeddings when an API key is available.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request
from abc import ABC, abstractmethod
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class HashingEmbedder(Embedder):
    """Hashing-trick vectorizer with sublinear TF weighting."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _bucket(self, token: str) -> int:
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        return h % self.dim

    def embed(self, text: str) -> list[float]:
        counts = Counter(tokenize(text))
        vec = [0.0] * self.dim
        for token, tf in counts.items():
            # Sublinear TF dampens very frequent tokens.
            vec[self._bucket(token)] += 1.0 + math.log(tf)
        norm = math.sqrt(sum(v * v for v in vec))
        if norm:
            vec = [v / norm for v in vec]
        return vec


class OpenAIEmbedder(Embedder):
    """Neural embeddings via the OpenAI embeddings API (stdlib only)."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.dim = 1536
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAIEmbedder")

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - network
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=json.dumps({"model": self.model, "input": texts}).encode(),
            headers={"content-type": "application/json", "authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read())
        vectors = [item["embedding"] for item in body["data"]]
        return [self._normalize(v) for v in vectors]

    @staticmethod
    def _normalize(v: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v] if norm else v


def get_embedder(kind: str = "hashing", dim: int = 512) -> Embedder:
    if kind == "openai":
        return OpenAIEmbedder()
    return HashingEmbedder(dim=dim)
