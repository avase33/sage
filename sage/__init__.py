"""Sage — a self-hostable personal AI agent.

Chat like Claude, but yours: a FastAPI backend with a streaming tool-using agent,
long-term memory, and document RAG (retrieval-augmented generation) over an
embeddings index. Offline-first — it runs on a deterministic mock model with no
API key, and drops in Anthropic or OpenAI when you have one.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
