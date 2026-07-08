"""Tests for the ML / agent core (no server, no network, no API keys)."""

import os
import tempfile

from sage.agent import Agent
from sage.db import Store
from sage.embeddings import HashingEmbedder, cosine
from sage.memory import summarize_history
from sage.providers import get_provider
from sage.providers.mock import MockProvider
from sage.rag import RagIndex, chunk_text
from sage.tools import default_tools
from sage.types import Message, TextDelta, ToolCallEvent, ToolResultEvent


# -- embeddings --------------------------------------------------------------
def test_embedding_is_normalized_and_similarity_ranks():
    emb = HashingEmbedder(dim=256)
    v = emb.embed("machine learning models")
    assert abs(sum(x * x for x in v) ** 0.5 - 1.0) < 1e-6  # unit length
    a = emb.embed("neural networks and deep learning")
    b = emb.embed("neural networks for deep learning tasks")
    c = emb.embed("the price of bananas at the market")
    assert cosine(a, b) > cosine(a, c)


# -- rag ---------------------------------------------------------------------
def test_chunking_overlaps():
    text = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_text(text, chunk_words=100, overlap=20)
    assert len(chunks) >= 3
    assert chunks[0].split()[-1] != chunks[1].split()[0]  # overlap => not a clean cut


def test_rag_retrieves_relevant_chunk():
    idx = RagIndex()
    idx.add_document("d1", "Space", "The Saturn V rocket launched the Apollo missions to the Moon.")
    idx.add_document("d2", "Cooking", "To bake sourdough you need flour, water, salt and a starter.")
    hits = idx.search("how did astronauts get to the moon", k=1)
    assert hits and "Apollo" in hits[0].chunk.text


# -- memory ------------------------------------------------------------------
def test_memory_summarizes_old_turns():
    history = [Message("user", f"msg {i}") for i in range(20)]
    summary, recent = summarize_history(history, keep_recent=6)
    assert summary and len(recent) == 6


# -- tools -------------------------------------------------------------------
def test_calculator_tool_and_registry():
    from sage.types import ToolCall
    reg = default_tools()
    res = reg.execute(ToolCall(name="calculator", arguments={"expression": "6*7"}))
    assert res.content == "42" and not res.is_error
    assert reg.execute(ToolCall(name="ghost", arguments={})).is_error


# -- providers ---------------------------------------------------------------
def test_mock_provider_streams_text():
    prov = MockProvider()
    text, calls = prov.complete([Message("user", "tell me something")])
    assert text and not calls


def test_get_provider_defaults_to_mock(monkeypatch=None):
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "SAGE_PROVIDER"):
        os.environ.pop(k, None)
    assert get_provider().name == "mock"


# -- agent -------------------------------------------------------------------
def test_agent_uses_calculator_via_tool_loop():
    agent = Agent(MockProvider(), tools=default_tools())
    events = list(agent.stream("What is 21 * 2?"))
    assert any(isinstance(e, ToolCallEvent) for e in events)
    assert any(isinstance(e, ToolResultEvent) and e.result.content == "42" for e in events)
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert "42" in text


def test_agent_grounds_answer_in_documents():
    idx = RagIndex()
    idx.add_document("d1", "Company", "Sage was founded in 2026 to build personal AI agents.")
    agent = Agent(MockProvider(), tools=default_tools(), rag=idx)
    answer = agent.respond("When was Sage founded?")
    assert "2026" in answer
    assert agent.last_sources  # retrieval actually fired


# -- persistence -------------------------------------------------------------
def test_store_roundtrip_and_index_rebuild():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.db")
        store = Store(path)
        cid = store.create_conversation("hello")
        store.add_message(cid, "user", "hi")
        store.add_message(cid, "assistant", "hello there")
        assert len(store.get_messages(cid)) == 2
        info = store.add_document("Doc", "retrieval augmented generation grounds answers")
        assert info["chunks"] >= 1
        idx = store.load_index()
        assert len(idx) == info["chunks"]
        hits = idx.search("retrieval augmented generation", k=1)
        assert hits
        store.close()
