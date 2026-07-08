"""Core data types shared across Sage."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


def new_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class Message:
    role: Role
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list["ToolCall"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        return d


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("call"))

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class ToolResult:
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# Streaming events emitted by providers / the agent.
@dataclass
class TextDelta:
    text: str
    type: Literal["text"] = "text"


@dataclass
class ToolCallEvent:
    tool_call: ToolCall
    type: Literal["tool_call"] = "tool_call"


@dataclass
class ToolResultEvent:
    result: ToolResult
    type: Literal["tool_result"] = "tool_result"


@dataclass
class DoneEvent:
    usage: Usage = field(default_factory=Usage)
    type: Literal["done"] = "done"


StreamEvent = TextDelta | ToolCallEvent | ToolResultEvent | DoneEvent
