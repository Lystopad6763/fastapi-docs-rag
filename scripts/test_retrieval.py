"""Тест retrieval на РЕАЛЬНОМУ індексі (Qdrant).

Ембедить кілька реальних питань і показує top-3 знайдені чанки (heading + score).
Це справжній показник якості пошуку — на відміну від іграшкових речень.

Запуск:  python scripts/test_retrieval.py
Передумова: виконаний scripts/index.py (колекція docs_chunks непорожня).
"""
from __future__ import annotations
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.config import settings                 # noqa: E402
from app.embeddings import embed_query           # noqa: E402
from app.vectorstore import get_client, search   # noqa: E402

QUERIES = [
    "How do I declare an optional query parameter?",
    "How to validate a request body using a Pydantic model?",
    "How do I use dependency injection with Depends?",
    "How to set a custom HTTP status code for a response?",
    "How to upload a file in FastAPI?",
    "Annotated Query max_length validation",       # точний термін — перевірка слабкого місця
]


def main() -> None:
    client = get_client()
    for q in QUERIES:
        qv = embed_query(q)
        hits = search(client, settings.chunks_collection, qv, top_k=3)
        print("=" * 72)
        print("Q:", q)
        for rank, h in enumerate(hits, 1):
            src = h.payload.get("source", "")
            heading = h.payload.get("heading", "")
            print(f"  {rank}. score={h.score:.3f}  [{src}]  {heading[:58]}")


if __name__ == "__main__":
    main()