"""Anthropic provider (stdlib only).

Uses the Messages API. For simplicity and robustness it makes a single request
and then re-emits the response as streaming deltas, so it plugs into the same
event interface as the mock provider.
"""

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

_URL = "https://api.anthropic.com/v1/messages"


class AnthropicProvider(Provider):
    name = "anthropic"
    default_model = "claude-3-5-sonnet-latest"

    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: int = 60):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.model = model or os.environ.get("SAGE_MODEL") or self.default_model
        self.timeout = timeout

    def _convert(self, messages: list[Message]):
        system: list[str] = []
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system.append(m.content)
            elif m.role == "user":
                out.append({"role": "user", "content": m.content})
            elif m.role == "assistant":
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
                out.append({"role": "assistant", "content": blocks or m.content})
            elif m.role == "tool":
                out.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}],
                })
        return "\n\n".join(s for s in system if s), out

    def stream(  # pragma: no cover - network
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> Iterator[StreamEvent]:
        system, converted = self._convert(messages)
        payload: dict[str, Any] = {
            "model": self.model, "max_tokens": 1024, "messages": converted,
            "temperature": 0.7 if temperature is None else temperature,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [
                {"name": t["name"], "description": t.get("description", ""), "input_schema": t["parameters"]}
                for t in tools
            ]
        req = urllib.request.Request(
            _URL, data=json.dumps(payload).encode(),
            headers={"content-type": "application/json", "x-api-key": self.api_key,
                     "anthropic-version": "2023-06-01"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read())
        tool_calls = []
        text = ""
        for block in body.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append(ToolCall(id=block["id"], name=block["name"], arguments=block.get("input", {})))
        for tc in tool_calls:
            yield ToolCallEvent(tc)
        if tool_calls:
            yield DoneEvent()
            return
        for tok in re.findall(r"\S+\s*", text):
            yield TextDelta(tok)
        u = body.get("usage", {})
        yield DoneEvent(Usage(u.get("input_tokens", 0), u.get("output_tokens", 0)))
