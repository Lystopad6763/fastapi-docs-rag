"""End-to-end RAG answer in the terminal: retrieve -> grounded prompt -> LLM stream.

Run:  python scripts/ask.py "How do I upload a file in FastAPI?"
      python scripts/ask.py            (uses the default question)
Prerequisites: OPENROUTER_API_KEY in .env and index.py already run.
"""
from __future__ import annotations
import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.rag import retrieve, build_context, build_messages   # noqa: E402
from app.llm import stream_chat                                # noqa: E402


async def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "How do I declare an optional query parameter?"
    hits = retrieve(query)
    context, sources = build_context(hits)
    messages = build_messages(query, context)

    print("Q:", query)
    print("-" * 72)
    async for token in stream_chat(messages):
        print(token, end="", flush=True)
    print("\n" + "-" * 72)
    print("SOURCES:", sources)


if __name__ == "__main__":
    asyncio.run(main())