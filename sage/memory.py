"""Conversation memory.

To keep prompts bounded on long chats, older turns are compressed into a short
running summary while the most recent turns are kept verbatim. The summary here
is extractive (dependency-free); swap in an LLM call for abstractive summaries.
"""

from __future__ import annotations

from .types import Message


def summarize_history(
    history: list[Message], keep_recent: int = 6, max_chars: int = 600
) -> tuple[str, list[Message]]:
    """Return (summary_of_older_turns, recent_turns_kept_verbatim)."""
    if len(history) <= keep_recent:
        return "", list(history)

    older = history[:-keep_recent]
    recent = history[-keep_recent:]
    parts = [
        f"{m.role}: {m.content}"
        for m in older
        if m.role in ("user", "assistant") and m.content
    ]
    summary = " | ".join(parts)
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3] + "..."
    return summary, recent
