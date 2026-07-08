"""Index a document and ask a grounded question — fully offline.

    python examples/seed_and_chat.py
"""

from sage.agent import Agent
from sage.providers.mock import MockProvider
from sage.rag import RagIndex
from sage.tools import default_tools

KNOWLEDGE = """
Sage is a self-hostable personal AI agent. It streams answers token by token,
can call tools such as a calculator, remembers the conversation, and grounds its
replies in your own documents using retrieval-augmented generation (RAG).
Embeddings are computed with a dependency-free hashing vectorizer by default,
and answers cite the passages they came from.
"""


def main() -> None:
    rag = RagIndex()
    rag.add_document("readme", "Sage overview", KNOWLEDGE)

    agent = Agent(MockProvider(), tools=default_tools(), rag=rag)

    for question in ["What does Sage use for grounding answers?", "What is 15 * 12?"]:
        print(f"\nYou:  {question}\nSage: ", end="")
        print(agent.respond(question))
        for i, s in enumerate(agent.last_sources, 1):
            print(f"      source [{i}] {s.chunk.title} (score {s.score:.3f})")


if __name__ == "__main__":
    main()
