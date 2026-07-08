"""HTTP layer tests (require the server extras: fastapi + httpx)."""

import os
import tempfile

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SAGE_DB", str(tmp_path / "srv.db"))
    monkeypatch.setenv("SAGE_PROVIDER", "mock")
    # Import after env is set so the module-level Store uses the temp db.
    import importlib
    import sage.server as server
    importlib.reload(server)
    return TestClient(server.app)


def _collect_sse(resp):
    events = []
    import json
    for line in resp.iter_lines():
        if line and line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["provider"] == "mock"


def test_chat_stream_and_persist(client):
    with client.stream("POST", "/api/chat", json={"message": "What is 8 * 8?"}) as resp:
        assert resp.status_code == 200
        events = _collect_sse(resp)
    types = [e["type"] for e in events]
    assert "text" in types and "done" in types
    cid = next(e["id"] for e in events if e["type"] == "conversation")
    msgs = client.get(f"/api/conversations/{cid}/messages").json()
    assert len(msgs) == 2  # user + assistant persisted


def test_document_upload_then_grounded_answer(client):
    r = client.post("/api/documents", json={"title": "Facts", "text": "The Eiffel Tower is 330 metres tall."})
    assert r.json()["chunks"] >= 1
    with client.stream("POST", "/api/chat", json={"message": "How tall is the Eiffel Tower?"}) as resp:
        events = _collect_sse(resp)
    assert any(e["type"] == "sources" for e in events)
