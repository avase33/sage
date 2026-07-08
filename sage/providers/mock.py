"""Offline mock provider.

Deterministic and dependency-free so the entire app — chat, tools, RAG — runs
and is testable with no API key. It streams token-by-token like a real model,
uses the calculator tool when it sees arithmetic, grounds answers in retrieved
document context when present, and finalises after a tool result.
"""

from __future__ import annotations

import re
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

_ARITH = re.compile(r"-?\d+(?:\.\d+)?(?:\s*[-+*/^]\s*-?\d+(?:\.\d+)?)+")

# The agent marks injected RAG context with this header.
CONTEXT_MARKER = "### Retrieved context"


def _last_user(messages: list[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


def _find_context(messages: list[Message]) -> str:
    for m in messages:
        if m.role == "system" and CONTEXT_MARKER in m.content:
            return m.content.split(CONTEXT_MARKER, 1)[1].strip()
    return ""


class MockProvider(Provider):
    name = "mock"
    default_model = "sage-mock-1"

    def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> Iterator[StreamEvent]:
        tools = tools or []
        tool_names = {t["name"] for t in tools}
        last = messages[-1] if messages else Message(role="user")
        user_text = _last_user(messages)
        had_tool = any(m.role == "tool" for m in messages) or any(
            m.tool_calls for m in messages
        )

        # Just received a tool result -> finalize.
        if last.role == "tool":
            yield from self._emit(f"The result is {last.content}.")
            return

        # Arithmetic -> call the calculator tool (once).
        if not had_tool and "calculator" in tool_names:
            match = _ARITH.search(user_text)
            if match:
                yield ToolCallEvent(
                    ToolCall(name="calculator", arguments={"expression": match.group(0).strip()})
                )
                yield DoneEvent()
                return

        low = user_text.strip().lower()
        if low in {"hi", "hello", "hey", "yo"} or low.startswith("hello"):
            yield from self._emit(
                "Hello! I'm Sage, your personal AI agent. Ask me anything, or upload a "
                "document and I'll answer questions grounded in it."
            )
            return

        context = _find_context(messages)
        if context:
            snippet = " ".join(context.split()[:60])
            answer = (
                f"Based on your documents: {snippet} "
                "(This grounded answer cites the retrieved passages above.)"
            )
        else:
            topic = user_text.strip().split("\n")[0][:160] or "that"
            answer = (
                f"Here's a concise take on \"{topic}\": it depends on the specifics, but "
                "the key considerations are the goal, the constraints, and the trade-offs "
                "involved. Ask a follow-up and I'll go deeper."
            )
        yield from self._emit(answer)

    def _emit(self, text: str) -> Iterator[StreamEvent]:
        tokens = re.findall(r"\S+\s*", text)
        for tok in tokens:
            yield TextDelta(tok)
        yield DoneEvent(usage=Usage(prompt_tokens=0, completion_tokens=len(tokens)))
