"""Two-tier cache for the RAG answers.

  - L1 (exact): an in-process dict keyed by the normalised query text. It is checked
    BEFORE embedding, so a repeated identical question is answered instantly without a
    call to the (remote) embedding API. This is what keeps repeat latency low — with a
    network embedding model the semantic lookup alone costs ~1-3s.
  - L2 (semantic): backed by Qdrant. Catches PARAPHRASES (cosine similarity > threshold,
    0.90) that L1 misses, and like L1 it returns the stored answer at $0 — skipping the
    expensive LLM generation. Reuses the same embedding as RAG (we never embed twice).

Both tiers share a 1-hour TTL and are GLOBAL for the document set (a public Q&A bot —
all API keys share one cache). L1 is per-process and warms up after the first answer;
L2 (Qdrant) survives restarts and is shared across workers.
"""
from __future__ import annotations
import time
import uuid

from qdrant_client.models import Distance, VectorParams, PointStruct
from app.config import settings
from app.vectorstore import get_client

_client = get_client()

# --- L1: exact-match cache (in-process) --------------------------------------
_exact: dict[str, dict] = {}


def _normalize(query: str) -> str:
    """Key for L1: lowercased, whitespace-collapsed — so trivial spacing/case
    variations of the same question still hit without an embedding call."""
    return " ".join(query.lower().split())


def exact_lookup(query: str) -> dict | None:
    """Return the cached payload for an identical (normalised) query, or None.
    Expired entries are dropped lazily on access."""
    entry = _exact.get(_normalize(query))
    if entry is None:
        return None
    if entry.get("expire_at", 0) < time.time():
        _exact.pop(_normalize(query), None)
        return None
    return entry


def exact_store(query: str, response: str, sources: list[str], model: str) -> None:
    _exact[_normalize(query)] = {
        "query": query,
        "response": response,
        "sources": sources,
        "model": model,
        "expire_at": time.time() + settings.cache_ttl_seconds,
    }


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


def flush() -> None:
    """Invalidate both cache tiers: clear L1 and recreate the L2 collection.
    Useful after re-indexing (cached answers may reference stale chunks) and for tests."""
    _exact.clear()
    if _client.collection_exists(settings.cache_collection):
        _client.delete_collection(settings.cache_collection)
    ensure_cache_collection()


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