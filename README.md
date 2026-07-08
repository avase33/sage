<div align="center">

# 🧠 Sage

### Your own personal AI agent — chat like Claude, self-hosted.

Streaming chat · tool use · long-term memory · document RAG · pluggable LLMs · runs offline.

[![CI](https://github.com/akhil/sage/actions/workflows/ci.yml/badge.svg)](https://github.com/akhil/sage/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/api-FastAPI-009485)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![core deps](https://img.shields.io/badge/core%20runtime%20deps-0-brightgreen)](pyproject.toml)

</div>

---

**Sage** is a full-stack personal AI assistant you can run yourself. It's a
FastAPI backend with a streaming, tool-using agent, conversation memory, and
**retrieval-augmented generation (RAG)** over your own documents — served to a
clean, single-file web chat UI.

The whole thing is **offline-first**: with no API key it runs on a deterministic
mock model and a dependency-free embeddings index, so it works the moment you
clone it. Add `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` and it upgrades to a real
model with one env var. The ML/agent core has **zero runtime dependencies**.

```bash
pip install -e ".[server]"
python -m sage serve          # open http://127.0.0.1:8000
```

Or try it in the terminal, no server needed:

```bash
python -m sage chat "What is 21 * 2?"
# [sage:mock]   ↳ calling calculator({'expression': '21 * 2'})... = 42
#               The result is 42.
```

## ✨ Features

- 💬 **Streaming chat** — token-by-token responses over Server-Sent Events, like a real assistant.
- 🛠️ **Tool use** — a bounded agent loop: the model calls tools (calculator, clock, …), sees the result, and continues. Add your own with a `@tool` decorator.
- 📚 **Document RAG** — upload text; Sage chunks it, embeds it, retrieves the most relevant passages by cosine similarity, and **cites its sources**.
- 🧵 **Memory** — full conversation history in SQLite, with automatic summarization of older turns to keep prompts bounded.
- 🔌 **Pluggable models** — `mock` (offline), `anthropic`, `openai` behind one streaming interface.
- 🧮 **Pluggable embeddings** — dependency-free hashing vectorizer by default, or OpenAI embeddings.
- 🖥️ **Self-hostable web UI** — one HTML file, no build step; sidebar, conversations, document upload, live streaming.

## 🖼️ The app

```
┌─ Sage ───────────┬─────────────────────────────────────────────┐
│  + New chat      │  You:  How tall is the Eiffel Tower?         │
│                  │  Sage: Based on your documents: The Eiffel   │
│  Conversations   │        Tower is 330 metres tall. [1]         │
│   • Eiffel tower │        ⚙ calculator(...) = …                 │
│   • Trip plan    │        Sources                               │
│                  │        [1] Facts — The Eiffel Tower is 330…  │
│  📄 Add document │  ┌──────────────────────────────────────┐    │
│                  │  │ Message Sage…                     ➤ │    │
└──────────────────┴──└──────────────────────────────────────┘────┘
```

## 🚀 Quick start

```bash
git clone https://github.com/akhil/sage.git
cd sage
pip install -e ".[server]"     # core is dependency-free; this adds FastAPI/uvicorn

python -m sage serve           # web app at http://127.0.0.1:8000
```

Ground answers in your own knowledge:

```bash
python -m sage add-doc notes.md --title "My notes"
python -m sage chat "summarize my notes"
```

Go live with a real model:

```bash
export ANTHROPIC_API_KEY=sk-...      # or OPENAI_API_KEY=sk-...
python -m sage serve                 # now answers come from Claude / GPT
```

## 🧩 How it works

A chat turn flows through the agent as: **persona → memory summary → retrieved
document context → recent history → your message**. The provider streams a reply;
if it asks for a tool, the agent runs it and re-prompts. See
[`docs/architecture.md`](docs/architecture.md) for the full diagram and the
request lifecycle.

```python
from sage.agent import Agent
from sage.providers.mock import MockProvider
from sage.rag import RagIndex
from sage.tools import default_tools

rag = RagIndex()
rag.add_document("d1", "Company", "Sage was founded in 2026 to build AI agents.")

agent = Agent(MockProvider(), tools=default_tools(), rag=rag)
print(agent.respond("When was Sage founded?"))   # grounded in the document, with sources
```

## 🔬 Tested & offline

The ML/agent core is covered by tests that need **no network and no keys** —
embeddings similarity, RAG retrieval, the tool loop, grounded answers, memory
summarization, and SQLite persistence. The HTTP layer is tested with FastAPI's
`TestClient`, including the streaming endpoint and RAG-cited answers.

```bash
pip install -e ".[dev]"
pytest -q
```

CI runs the suite on Python 3.10–3.12 and smoke-tests the CLI and server import.

## 🗺️ Roadmap

- [ ] True token streaming from Anthropic/OpenAI (SSE passthrough)
- [ ] File upload (PDF/markdown) parsing for RAG
- [ ] Vector store backends (sqlite-vss / pgvector)
- [ ] Auth + multi-user workspaces
- [ ] Tool plugins (web search, code execution sandbox)

## 📄 License

 Akhil — see [LICENSE](LICENSE).
