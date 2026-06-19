"""Fallback test: a chain with an INVALID primary model -> should switch to a working one.

Calls stream_chat directly (no HTTP) to test the fallback logic in isolation.

Run:  python scripts/test_fallback.py
Prerequisites: OPENROUTER_API_KEY in .env and index.py already run.
"""
from __future__ import annotations
import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.rag import retrieve, build_context, build_messages   # noqa: E402
from app.llm import stream_chat                                # noqa: E402

# primary is intentionally nonexistent; fallback is a working cheap model
MODELS = ["openai/this-does-not-exist", "meta-llama/llama-3.1-8b-instruct"]


async def main() -> None:
    q = "How do I declare an optional query parameter?"
    hits = retrieve(q)
    context, _ = build_context(hits)
    messages = build_messages(q, context)
    stats: dict = {}

    print("Model chain:", MODELS)
    print("-" * 60)
    async for token in stream_chat(messages, models=MODELS, stats=stats):
        print(token, end="", flush=True)
    print("\n" + "-" * 60)
    print("Model used:", stats.get("model"))
    print("fallback_used     :", stats.get("fallback_used"), "(expect True)")


if __name__ == "__main__":
    asyncio.run(main())