"""FastAPI backend for Sage.

Endpoints:
  GET  /                       -> the chat web UI
  GET  /api/health             -> health + active provider
  GET  /api/conversations      -> list conversations
  POST /api/conversations      -> create a conversation
  GET  /api/conversations/{id}/messages
  DELETE /api/conversations/{id}
  POST /api/chat               -> streaming chat (Server-Sent Events)
  GET  /api/documents          -> list indexed documents
  POST /api/documents          -> add + index a document for RAG

Run:  uvicorn sage.server:app --reload   (or `python -m sage serve`)
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .agent import Agent
from .config import Settings
from .db import Store
from .embeddings import get_embedder
from .providers import get_provider
from .tools import default_tools
from .types import DoneEvent, TextDelta, ToolCallEvent, ToolResultEvent

settings = Settings.from_env()
_embedder = get_embedder(settings.embedder, settings.embedding_dim)
store = Store(settings.db_path, embedder=_embedder)

app = FastAPI(title="Sage", version="0.1.0")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "provider": settings.provider, "embedder": settings.embedder,
            "documents": len(store.list_documents())}


@app.get("/api/conversations")
def list_conversations():
    return store.list_conversations()


class NewConversation(BaseModel):
    title: str = "New chat"


@app.post("/api/conversations")
def create_conversation(body: NewConversation):
    return {"id": store.create_conversation(body.title)}


@app.get("/api/conversations/{cid}/messages")
def get_messages(cid: str):
    return [m.to_dict() for m in store.get_messages(cid)]


@app.delete("/api/conversations/{cid}")
def delete_conversation(cid: str):
    store.delete_conversation(cid)
    return {"ok": True}


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


@app.post("/api/chat")
def chat(req: ChatRequest):
    cid = req.conversation_id or store.create_conversation(req.message[:40] or "New chat")
    history = store.get_messages(cid)
    store.add_message(cid, "user", req.message)

    provider = get_provider(settings.provider)
    rag = store.load_index()
    agent = Agent(provider, tools=default_tools(), rag=rag,
                  max_steps=settings.max_tool_steps, rag_top_k=settings.rag_top_k)

    def gen():
        yield _sse({"type": "conversation", "id": cid})
        parts: list[str] = []
        for ev in agent.stream(req.message, history):
            if isinstance(ev, TextDelta):
                parts.append(ev.text)
                yield _sse({"type": "text", "text": ev.text})
            elif isinstance(ev, ToolCallEvent):
                yield _sse({"type": "tool_call", "name": ev.tool_call.name,
                            "arguments": ev.tool_call.arguments})
            elif isinstance(ev, ToolResultEvent):
                yield _sse({"type": "tool_result", "name": ev.result.name,
                            "content": ev.result.content, "is_error": ev.result.is_error})
            elif isinstance(ev, DoneEvent):
                pass
        store.add_message(cid, "assistant", "".join(parts))
        if agent.last_sources:
            yield _sse({"type": "sources", "sources": [
                {"title": s.chunk.title, "text": s.chunk.text[:220], "score": round(s.score, 3)}
                for s in agent.last_sources]})
        yield _sse({"type": "done"})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/documents")
def list_documents():
    return store.list_documents()


class NewDocument(BaseModel):
    title: str
    text: str


@app.post("/api/documents")
def add_document(body: NewDocument):
    if not body.text.strip():
        return JSONResponse({"error": "empty document"}, status_code=400)
    return store.add_document(body.title or "Untitled", body.text)
