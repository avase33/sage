"""SQLite persistence for conversations, messages, documents and RAG chunks.

Uses only the standard library. Embeddings are stored as JSON so the vector
index can be rebuilt on startup without a vector database.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from .embeddings import Embedder, HashingEmbedder
from .rag import Chunk, RagIndex, chunk_text
from .types import Message, new_id

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY, title TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY, conversation_id TEXT, role TEXT, content TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY, title TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY, doc_id TEXT, title TEXT, text TEXT, embedding TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


class Store:
    def __init__(self, path: str = "sage.db", embedder: Embedder | None = None):
        self.path = path
        self.embedder = embedder or HashingEmbedder()
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # -- conversations -------------------------------------------------------
    def create_conversation(self, title: str = "New chat") -> str:
        cid = new_id("conv")
        self.conn.execute(
            "INSERT INTO conversations VALUES (?,?,?)", (cid, title, time.time())
        )
        self.conn.commit()
        return cid

    def list_conversations(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, title, created_at FROM conversations ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def rename_conversation(self, cid: str, title: str) -> None:
        self.conn.execute("UPDATE conversations SET title=? WHERE id=?", (title, cid))
        self.conn.commit()

    def delete_conversation(self, cid: str) -> None:
        self.conn.execute("DELETE FROM conversations WHERE id=?", (cid,))
        self.conn.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
        self.conn.commit()

    # -- messages ------------------------------------------------------------
    def add_message(self, conversation_id: str, role: str, content: str) -> str:
        mid = new_id("msg")
        self.conn.execute(
            "INSERT INTO messages VALUES (?,?,?,?,?)",
            (mid, conversation_id, role, content, time.time()),
        )
        self.conn.commit()
        return mid

    def get_messages(self, conversation_id: str) -> list[Message]:
        rows = self.conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY created_at",
            (conversation_id,),
        ).fetchall()
        return [Message(role=r["role"], content=r["content"]) for r in rows]

    # -- documents / RAG -----------------------------------------------------
    def add_document(self, title: str, text: str) -> dict:
        doc_id = new_id("doc")
        self.conn.execute(
            "INSERT INTO documents VALUES (?,?,?)", (doc_id, title, time.time())
        )
        pieces = chunk_text(text)
        for i, piece in enumerate(pieces):
            emb = self.embedder.embed(piece)
            self.conn.execute(
                "INSERT INTO chunks VALUES (?,?,?,?,?)",
                (f"{doc_id}:{i}", doc_id, title, piece, json.dumps(emb)),
            )
        self.conn.commit()
        return {"id": doc_id, "title": title, "chunks": len(pieces)}

    def list_documents(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, title, created_at FROM documents ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def load_index(self) -> RagIndex:
        index = RagIndex(embedder=self.embedder)
        rows = self.conn.execute("SELECT id, doc_id, title, text, embedding FROM chunks").fetchall()
        for r in rows:
            index.chunks.append(
                Chunk(
                    id=r["id"], doc_id=r["doc_id"], title=r["title"], text=r["text"],
                    embedding=json.loads(r["embedding"]),
                )
            )
        return index

    def close(self) -> None:
        self.conn.close()
