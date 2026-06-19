"""Semantic cache backed by Qdrant.

Key points:
  - reuses the same embedding as RAG (passed in from outside — we do NOT embed twice);
  - a separate `semantic_cache` collection; a HIT requires cosine similarity > threshold (0.90);
  - a 1-hour TTL via the payload field `expire_at` (Qdrant has no built-in TTL);
  - the cache is GLOBAL for the document set (a public Q&A bot — all keys share one cache).
"""
from __future__ import annotations
import time
import uuid

from qdrant_client.models import Distance, VectorParams, PointStruct
from app.config import settings
from app.vectorstore import get_client

_client = get_client()


def ensure_cache_collection() -> None:
    if not _client.collection_exists(settings.cache_collection):
        _client.create_collection(
            collection_name=settings.cache_collection,
            vectors_config=VectorParams(size=settings.embed_dim, distance=Distance.COSINE),
        )


def lookup(query_vector: list[float]) -> dict | None:
    """Return the cached payload if a close, non-expired query exists; otherwise None."""
    points = _client.query_points(
        collection_name=settings.cache_collection,
        query=query_vector, limit=1, with_payload=True,
    ).points
    if not points:
        return None
    hit = points[0]
    if hit.score < settings.cache_threshold:          # not similar enough
        return None
    if hit.payload.get("expire_at", 0) < time.time():  # expired (TTL)
        return None
    return hit.payload


def store(query_vector: list[float], query: str, response: str,
          sources: list[str], model: str) -> None:
    _client.upsert(
        collection_name=settings.cache_collection,
        points=[PointStruct(
            id=str(uuid.uuid4()),
            vector=query_vector,
            payload={
                "query": query,
                "response": response,
                "sources": sources,
                "model": model,
                "expire_at": time.time() + settings.cache_ttl_seconds,
            },
        )],
    )