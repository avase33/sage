"""OpenAI provider (stdlib only). Single request, re-emitted as stream events."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any, Iterator

from ..types import (
    DoneEvent,
    Message,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    Usage,
)
from .base import Provider

_URL = "https://api.openai.com/v1/chat/completions"


class OpenAIProvider(Provider):
    name = "openai"
    default_model = "gpt-4o-mini"

    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: int = 60):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.model = model or os.environ.get("SAGE_MODEL") or self.default_model
        self.timeout = timeout

    def _convert(self, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "tool":
                out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
            elif m.role == "assistant" and m.tool_calls:
                out.append({
                    "role": "assistant", "content": m.content or None,
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                        for tc in m.tool_calls
                    ],
                })
            else:
                out.append({"role": m.role, "content": m.content})
        return out

    def stream(  # pragma: no cover - network
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> Iterator[StreamEvent]:
        payload: dict[str, Any] = {
            "model": self.model, "messages": self._convert(messages),
            "temperature": 0.7 if temperature is None else temperature,
        }
        if tools:
            payload["tools"] = [
                {"type": "function", "function": {
                    "name": t["name"], "description": t.get("description", ""), "parameters": t["parameters"]}}
                for t in tools
            ]
        req = urllib.request.Request(
            _URL, data=json.dumps(payload).encode(),
            headers={"content-type": "application/json", "authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read())
        choice = body["choices"][0]["message"]
        for tc in choice.get("tool_calls") or []:
            fn = tc["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            yield ToolCallEvent(ToolCall(id=tc.get("id", ""), name=fn["name"], arguments=args))
        if choice.get("tool_calls"):
            yield DoneEvent()
            return
        for tok in re.findall(r"\S+\s*", choice.get("content") or ""):
            yield TextDelta(tok)
        u = body.get("usage", {})
        yield DoneEvent(Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0)))
