"""Tools the agent can call.

`@tool` turns a typed function into a schema-carrying Tool; ToolRegistry runs
tool calls and turns failures into recoverable error results.
"""

from __future__ import annotations

import ast
import inspect
import operator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from .types import ToolCall, ToolResult

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _schema(fn: Callable) -> dict[str, Any]:
    props, required = {}, []
    for name, p in inspect.signature(fn).parameters.items():
        if name in ("self", "cls"):
            continue
        props[name] = {"type": _JSON_TYPES.get(p.annotation, "string")}
        if p.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": props, "required": required}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable

    def run(self, **kwargs: Any) -> str:
        allowed = set(self.parameters.get("properties", {}))
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        return str(self.func(**kwargs))

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def tool(fn: Callable | None = None, *, name: str | None = None):
    def wrap(f: Callable) -> Tool:
        return Tool(name or f.__name__, (inspect.getdoc(f) or "").strip(), _schema(f), f)

    return wrap(fn) if fn else wrap


# -- safe calculator ---------------------------------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.FloorDiv: operator.floordiv}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _ev(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _ev(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_ev(node.left), _ev(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_ev(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression (+, -, *, /, %, **)."""
    value = _ev(ast.parse(expression.replace("^", "**"), mode="eval"))
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


@tool
def current_time() -> str:
    """Return the current UTC date and time (ISO-8601)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, t: Tool) -> None:
        self.tools[t.name] = t

    def names(self) -> list[str]:
        return sorted(self.tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.tools.values()]

    def execute(self, call: ToolCall) -> ToolResult:
        t = self.tools.get(call.name)
        if not t:
            return ToolResult(call.id, call.name, f"No tool named {call.name!r}", is_error=True)
        try:
            return ToolResult(call.id, call.name, t.run(**call.arguments))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(call.id, call.name, f"Tool error: {exc}", is_error=True)


def default_tools() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(calculator)
    reg.register(current_time)
    return reg
