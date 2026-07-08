"""Retrieval-augmented generation: chunk documents, embed them, and retrieve the
most relevant passages for a query so the agent can ground its answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .embeddings import Embedder, HashingEmbedder, cosine


def chunk_text(text: str, chunk_words: int = 120, overlap: int = 20) -> list[str]:
    """Split text into overlapping word windows.

    Overlap keeps context from spilling across chunk boundaries so a sentence
    split down the middle is still retrievable from either side.
    """

    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_words]
        if window:
            chunks.append(" ".join(window))
        if start + chunk_words >= len(words):
            break
    return chunks


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    text: str
    embedding: list[float] = field(default_factory=list)


@dataclass
class Retrieved:
    chunk: Chunk
    score: float


class RagIndex:
    """An in-memory vector index over document chunks."""

    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or HashingEmbedder()
        self.chunks: list[Chunk] = []

    def add_chunk(self, chunk: Chunk) -> None:
        if not chunk.embedding:
            chunk.embedding = self.embedder.embed(chunk.text)
        self.chunks.append(chunk)

    def add_document(self, doc_id: str, title: str, text: str) -> list[Chunk]:
        created: list[Chunk] = []
        for i, piece in enumerate(chunk_text(text)):
            chunk = Chunk(id=f"{doc_id}:{i}", doc_id=doc_id, title=title, text=piece)
            self.add_chunk(chunk)
            created.append(chunk)
        return created

    def search(self, query: str, k: int = 4, min_score: float = 0.01) -> list[Retrieved]:
        if not self.chunks:
            return []
        q = self.embedder.embed(query)
        scored = [Retrieved(c, cosine(q, c.embedding)) for c in self.chunks]
        scored.sort(key=lambda r: r.score, reverse=True)
        return [r for r in scored[:k] if r.score > min_score]

    def context_for(self, query: str, k: int = 4) -> tuple[str, list[Retrieved]]:
        """Return a prompt-ready context block plus the sources used."""
        hits = self.search(query, k)
        if not hits:
            return "", []
        blocks = []
        for i, r in enumerate(hits, 1):
            blocks.append(f"[{i}] (from \"{r.chunk.title}\")\n{r.chunk.text}")
        return "\n\n".join(blocks), hits

    def __len__(self) -> int:
        return len(self.chunks)
