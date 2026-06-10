"""Тест fallback (§7): ланцюжок з НЕВАЛІДНОЮ primary-моделлю -> має перемкнутись на робочу.

Кличе stream_chat напряму (без HTTP), щоб ізольовано перевірити логіку fallback.

Запуск:  python scripts/test_fallback.py
Передумова: OPENROUTER_API_KEY у .env, виконаний index.py.
"""
from __future__ import annotations
import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.rag import retrieve, build_context, build_messages   # noqa: E402
from app.llm import stream_chat                                # noqa: E402

# primary — навмисно неіснуюча; fallback — робоча дешева модель
MODELS = ["openai/this-does-not-exist", "meta-llama/llama-3.1-8b-instruct"]


async def main() -> None:
    q = "How do I declare an optional query parameter?"
    hits = retrieve(q)
    context, _ = build_context(hits)
    messages = build_messages(q, context)
    stats: dict = {}

    print("Ланцюжок моделей:", MODELS)
    print("-" * 60)
    async for token in stream_chat(messages, models=MODELS, stats=stats):
        print(token, end="", flush=True)
    print("\n" + "-" * 60)
    print("Використана модель:", stats.get("model"))
    print("fallback_used     :", stats.get("fallback_used"), "(очікуємо True)")


if __name__ == "__main__":
    asyncio.run(main())