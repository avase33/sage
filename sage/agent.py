"""The Sage agent.

Assembles the prompt (persona + conversation memory + retrieved document
context + history + the new message), streams the model's reply, and runs a
bounded tool-use loop — re-prompting after each tool result. Emits the same
streaming events the provider does, so the HTTP layer can forward them verbatim.
"""

from __future__ import annotations

from typing import Iterator

from .memory import summarize_history
from .providers.base import Provider
from .providers.mock import CONTEXT_MARKER
from .rag import RagIndex, Retrieved
from .tools import ToolRegistry
from .types import (
    DoneEvent,
    Message,
    StreamEvent,
    TextDelta,
    ToolCallEvent,
    ToolResultEvent,
)

PERSONA = (
    "You are Sage, a helpful, thoughtful personal AI agent. Answer clearly and "
    "concisely. If document context is provided, ground your answer in it and "
    "cite the passage numbers like [1], [2]. Use tools when they help."
)


class Agent:
    def __init__(
        self,
        provider: Provider,
        tools: ToolRegistry | None = None,
        rag: RagIndex | None = None,
        *,
        persona: str = PERSONA,
        max_steps: int = 6,
        rag_top_k: int = 4,
    ):
        self.provider = provider
        self.tools = tools
        self.rag = rag
        self.persona = persona
        self.max_steps = max_steps
        self.rag_top_k = rag_top_k
        self.last_sources: list[Retrieved] = []

    def build_messages(self, user_text: str, history: list[Message]) -> list[Message]:
        messages: list[Message] = [Message(role="system", content=self.persona)]

        # Long conversations: compress older turns into a memory summary.
        summary, recent = summarize_history(history)
        if summary:
            messages.append(Message(role="system", content=f"Conversation so far: {summary}"))

        # Retrieval-augmented context.
        self.last_sources = []
        if self.rag is not None and len(self.rag) > 0:
            context, sources = self.rag.context_for(user_text, k=self.rag_top_k)
            if context:
                self.last_sources = sources
                messages.append(
                    Message(role="system", content=f"{CONTEXT_MARKER}\n{context}")
                )

        messages.extend(recent)
        messages.append(Message(role="user", content=user_text))
        return messages

    def stream(self, user_text: str, history: list[Message] | None = None) -> Iterator[StreamEvent]:
        messages = self.build_messages(user_text, history or [])
        schemas = self.tools.schemas() if self.tools else None

        for _step in range(self.max_steps):
            assistant_text = ""
            tool_calls = []
            for ev in self.provider.stream(messages, schemas):
                if isinstance(ev, TextDelta):
                    assistant_text += ev.text
                    yield ev
                elif isinstance(ev, ToolCallEvent):
                    tool_calls.append(ev.tool_call)
                    yield ev
                # swallow the provider's DoneEvent; the agent emits its own.

            if tool_calls and self.tools is not None:
                messages.append(
                    Message(role="assistant", content=assistant_text, tool_calls=tool_calls)
                )
                for call in tool_calls:
                    result = self.tools.execute(call)
                    yield ToolResultEvent(result)
                    messages.append(
                        Message(role="tool", content=result.content,
                                tool_call_id=call.id, name=call.name)
                    )
                continue  # re-prompt with tool results

            yield DoneEvent()
            return

        yield DoneEvent()

    def respond(self, user_text: str, history: list[Message] | None = None) -> str:
        """Non-streaming convenience: return the full reply text."""
        parts: list[str] = []
        for ev in self.stream(user_text, history):
            if isinstance(ev, TextDelta):
                parts.append(ev.text)
        return "".join(parts)
