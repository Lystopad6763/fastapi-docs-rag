"""Показати, який grounded-промпт rag.py збирає для LLM (без виклику LLM).

Запуск:  python scripts/test_rag.py
"""
from __future__ import annotations
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.rag import retrieve, build_context, build_messages   # noqa: E402

QUERY = "How do I declare an optional query parameter?"


def main() -> None:
    hits = retrieve(QUERY)
    context, sources = build_context(hits)
    messages = build_messages(QUERY, context)

    print("SOURCES (підуть у done-event):", sources)
    print("=" * 72)
    print("SYSTEM PROMPT:\n" + messages[0]["content"])
    print("=" * 72)
    user = messages[1]["content"]
    print(f"USER MESSAGE (перші 1400 з {len(user)} символів):\n")
    print(user[:1400] + ("\n... [обрізано]" if len(user) > 1400 else ""))


if __name__ == "__main__":
    main()