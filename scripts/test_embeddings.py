"""Швидкий тест ембедингу (перед індексацією).

Доводить 3 речі:
  1) ембединг працює, розмірність = 1536, вектори нормалізовані (norm ~ 1.0);
  2) семантичний пошук ранжує за ЗМІСТОМ (релевантне зверху, банан — знизу);
  3) Q1-експеримент: чи код у тексті допомагає/заважає матчингу.

Запуск:  python scripts/test_embeddings.py
Передумова: OPENAI_API_KEY у .env.
"""
from __future__ import annotations
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np                                  # noqa: E402
from app.config import settings                     # noqa: E402
from app.embeddings import embed_texts, embed_query  # noqa: E402


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    if not settings.openai_api_key:
        print("!! OPENAI_API_KEY порожній — встав ключ у .env"); return

    # --- 1) Sanity: розмірність + нормалізація + детермінізм -----------------
    print("=" * 64)
    print("1) SANITY")
    v1 = np.array(embed_query("FastAPI query parameters"))
    v1b = np.array(embed_query("FastAPI query parameters"))  # той самий текст вдруге
    print(f"   розмірність вектора : {v1.shape[0]}  (очікуємо 1536)")
    print(f"   норма вектора       : {np.linalg.norm(v1):.4f}  (OpenAI віддає ~1.0)")
    print(f"   детермінізм (cos того самого тексту двічі): {cos(v1, v1b):.5f}  (~1.0)")

    # --- 2) Семантичний пошук ------------------------------------------------
    print("\n" + "=" * 64)
    print("2) СЕМАНТИЧНИЙ ПОШУК — запит проти 4 текстів")
    query = "How do I declare an optional query parameter in FastAPI?"
    docs = [
        "Query parameters: function params not part of the path are interpreted as query parameters with default values.",
        "Path parameters are part of the URL path and are always required.",
        "Use Depends() to declare dependencies that FastAPI injects into your endpoint.",
        "Bananas are a good source of potassium and grow in tropical climates.",
    ]
    qv = np.array(embed_query(query))
    dv = np.array(embed_texts(docs))
    sims = dv @ qv / (np.linalg.norm(dv, axis=1) * np.linalg.norm(qv))
    print(f"   Q: {query}")
    for rank, i in enumerate(np.argsort(-sims), 1):
        print(f"   {rank}. sim={sims[i]:.3f}  {docs[i][:62]}")
    print("   -> очікуємо #1 = query-parameters, останній = банан")

    # --- 3) Q1: код у ембедінгу — допомагає чи розмиває? ---------------------
    print("\n" + "=" * 64)
    print("3) Q1-ЕКСПЕРИМЕНТ: проза vs код vs проза+код")
    q2 = "example of Annotated Query validation with max_length"
    prose = "You can add extra validation and metadata to query parameters using Annotated and Query."
    code = (
        "```python\n"
        "from typing import Annotated\n"
        "from fastapi import FastAPI, Query\n"
        "app = FastAPI()\n"
        "@app.get('/items/')\n"
        "async def read_items(q: Annotated[str | None, Query(max_length=50)] = None):\n"
        "    return {'q': q}\n"
        "```"
    )
    both = prose + "\n\n" + code
    q2v = np.array(embed_query(q2))
    pv, cv, bv = (np.array(embed_query(t)) for t in (prose, code, both))
    print(f"   Q: {q2}")
    print(f"   проза тільки : sim={cos(q2v, pv):.3f}")
    print(f"   код тільки   : sim={cos(q2v, cv):.3f}")
    print(f"   проза + код  : sim={cos(q2v, bv):.3f}")
    print("   -> якщо 'проза+код' >= 'проза', код НЕ шкодить (для наших коротких прикладів)")


if __name__ == "__main__":
    main()