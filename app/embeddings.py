"""OpenAI embeddings — text-embedding-3-small (1536d), метрика cosine.

Без torch -> малий Docker-образ. Перевикористовується індексацією
(чанки) і retrieval (запит) — щоб вектори були в одному просторі.
"""
from __future__ import annotations
from openai import OpenAI
from app.config import settings

_client = OpenAI(api_key=settings.openai_api_key)


def embed_texts(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Ембедити список текстів батчами (економить кількість HTTP-викликів)."""
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        resp = _client.embeddings.create(model=settings.embed_model, input=texts[i:i + batch_size])
        # API повертає поле index; сортуємо, щоб гарантовано зберегти порядок входу
        out.extend(item.embedding for item in sorted(resp.data, key=lambda d: d.index))
    return out


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]