"""Command-line interface for Sage.

    sage serve                     # run the web app
    sage chat "your question"      # one-shot chat in the terminal (offline mock ok)
    sage add-doc notes.txt         # index a document for RAG
    sage docs                      # list indexed documents
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .agent import Agent
from .config import Settings
from .db import Store
from .embeddings import get_embedder
from .providers import get_provider
from .tools import default_tools
from .types import TextDelta, ToolCallEvent, ToolResultEvent


def _store(settings: Settings) -> Store:
    return Store(settings.db_path, embedder=get_embedder(settings.embedder, settings.embedding_dim))


def cmd_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required to serve. Install with: pip install 'sage-agent[server]'", file=sys.stderr)
        return 1
    uvicorn.run("sage.server:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_chat(args) -> int:
    settings = Settings.from_env()
    store = _store(settings)
    provider = get_provider(settings.provider)
    agent = Agent(provider, tools=default_tools(), rag=store.load_index(),
                  max_steps=settings.max_tool_steps, rag_top_k=settings.rag_top_k)
    print(f"[sage:{settings.provider}] ", end="", flush=True)
    for ev in agent.stream(args.message, history=[]):
        if isinstance(ev, TextDelta):
            print(ev.text, end="", flush=True)
        elif isinstance(ev, ToolCallEvent):
            print(f"\n  -> calling {ev.tool_call.name}({ev.tool_call.arguments})... ", end="", flush=True)
        elif isinstance(ev, ToolResultEvent):
            print(f"= {ev.result.content}\n", end="", flush=True)
    print()
    if agent.last_sources:
        print("\nSources:")
        for i, s in enumerate(agent.last_sources, 1):
            print(f"  [{i}] {s.chunk.title} (score {s.score:.3f})")
    store.close()
    return 0


def cmd_add_doc(args) -> int:
    settings = Settings.from_env()
    store = _store(settings)
    text = Path(args.path).read_text(encoding="utf-8", errors="replace")
    title = args.title or Path(args.path).name
    info = store.add_document(title, text)
    print(f"Indexed '{info['title']}' -> {info['chunks']} chunks.")
    store.close()
    return 0


def cmd_docs(args) -> int:
    store = _store(Settings.from_env())
    docs = store.list_documents()
    if not docs:
        print("No documents indexed yet. Add one with: sage add-doc <file>")
    for d in docs:
        print(f"  - {d['title']} ({d['id']})")
    store.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sage", description="Sage — your personal AI agent.")
    p.add_argument("--version", action="version", version=f"Sage {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="run the web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=cmd_serve)

    c = sub.add_parser("chat", help="one-shot chat in the terminal")
    c.add_argument("message")
    c.set_defaults(func=cmd_chat)

    a = sub.add_parser("add-doc", help="index a document for RAG")
    a.add_argument("path")
    a.add_argument("--title")
    a.set_defaults(func=cmd_add_doc)

    d = sub.add_parser("docs", help="list indexed documents")
    d.set_defaults(func=cmd_docs)

    return p


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # cross-platform console output
    except Exception:
        pass
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
