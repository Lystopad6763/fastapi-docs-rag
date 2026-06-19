"""OpenAI embeddings — text-embedding-3-small (1536d), cosine metric.

No torch dependency, which keeps the Docker image small. The same functions are reused
by both indexing (chunks) and retrieval (queries) so that all vectors live in the same space.
"""
from __future__ import annotations
from openai import OpenAI
from app.config import settings

_client = OpenAI(api_key=settings.openai_api_key)


def embed_texts(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Embed a list of texts in batches to reduce the number of HTTP calls."""
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        resp = _client.embeddings.create(model=settings.embed_model, input=texts[i:i + batch_size])
        # The API returns an `index` field; sort by it to preserve the input order
        out.extend(item.embedding for item in sorted(resp.data, key=lambda d: d.index))
    return out


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]