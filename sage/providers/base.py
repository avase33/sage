"""LLM provider interface.

Providers turn a list of messages (and optional tool schemas) into a *stream* of
events: text deltas, tool-call requests, and a final done event. The agent and
the HTTP layer consume this stream to give a token-by-token chat experience.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator

from ..types import Message, StreamEvent


class Provider(ABC):
    name: str = "base"
    default_model: str = ""

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> Iterator[StreamEvent]:
        """Yield streaming events for the given conversation."""

    def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> tuple[str, list]:
        """Non-streaming convenience: collect the stream into (text, tool_calls)."""
        from ..types import TextDelta, ToolCallEvent

        text_parts: list[str] = []
        tool_calls = []
        for ev in self.stream(messages, tools):
            if isinstance(ev, TextDelta):
                text_parts.append(ev.text)
            elif isinstance(ev, ToolCallEvent):
                tool_calls.append(ev.tool_call)
        return "".join(text_parts), tool_calls
