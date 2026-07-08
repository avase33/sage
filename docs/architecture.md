# Sage architecture

Sage is a full-stack personal AI agent. The **ML/agent core is pure Python
standard library** (so it runs and tests offline with no API keys); the web layer
adds FastAPI + a single-file browser UI.

```
                    Browser chat UI  (web/index.html, SSE streaming)
                              │  HTTP / Server-Sent Events
                    ┌─────────▼──────────┐
                    │  FastAPI server    │  server.py
                    │  /api/chat (stream)│
                    └─────────┬──────────┘
                    ┌─────────▼──────────┐
                    │       Agent        │  agent.py
                    │  persona + memory  │
                    │  + RAG + tool loop │
                    └───┬─────┬─────┬────┘
              ┌─────────┘     │     └──────────┐
              ▼               ▼                ▼
        Providers          Tools            RAG            Memory
     (mock/anthropic/   (calculator,   (chunk→embed→    (summarize
      openai, stream)    time, …)       cosine search)   old turns)
                                          │
                                          ▼
                                     Embeddings
                                 (hashing / OpenAI)
                                          │
                                          ▼
                                   SQLite store (db.py)
                          conversations · messages · docs · chunks
```

## Request lifecycle (a chat turn)

1. The browser POSTs the message to `/api/chat` and reads the **SSE stream**.
2. The server loads conversation history and the RAG index from SQLite, then
   builds an `Agent`.
3. `Agent.build_messages` assembles the prompt: persona → memory summary of old
   turns → **retrieved document context** (top-k by embedding cosine similarity)
   → recent history → the new message.
4. `Agent.stream` calls the provider and forwards **text deltas** as they arrive.
   If the model requests a **tool**, the agent runs it, streams the result, and
   re-prompts — a bounded tool-use loop.
5. The final answer and cited sources are streamed out and persisted.

## Why offline-first

`get_provider()` returns a real model when an API key is present and otherwise a
deterministic `MockProvider` that still streams, calls tools, and grounds on
retrieved context. Embeddings default to a dependency-free `HashingEmbedder`
(hashing trick + sublinear TF). Result: `pytest` and the CLI work with **zero
setup**, and CI is hermetic — while a one-line env change swaps in Anthropic or
OpenAI for production-quality answers.

## Extending Sage

- **New tool:** decorate a typed function with `@tool` and register it.
- **New provider:** implement `Provider.stream` (see `providers/base.py`).
- **Real embeddings:** set `SAGE_EMBEDDER=openai` (or subclass `Embedder`).
- **New transport/UI:** the API is plain JSON + SSE; build any client on it.
