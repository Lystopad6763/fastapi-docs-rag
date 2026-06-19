"""Qdrant vector store: HNSW index with cosine metric.

Each collection stores one vector per chunk plus its payload (text, source, heading).
HNSW and cosine are Qdrant's defaults; at our scale (~hundreds of vectors) even a flat
index would be instant, but HNSW costs nothing here (index in RAM, payload on disk).
"""
from __future__ import annotations
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.config import settings


def get_client() -> QdrantClient:
    # api_key is required for Qdrant Cloud; pass None for a local container so the
    # client doesn't send an empty auth header.
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)


def ensure_collection(client: QdrantClient, name: str, recreate: bool = False) -> None:
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=settings.embed_dim, distance=Distance.COSINE),
        )


def upsert_chunks(client: QdrantClient, name: str,
                  ids: list[int], vectors: list[list[float]], payloads: list[dict]) -> None:
    points = [PointStruct(id=i, vector=v, payload=p)
              for i, v, p in zip(ids, vectors, payloads)]
    client.upsert(collection_name=name, points=points)


def search(client: QdrantClient, name: str, query_vector: list[float], top_k: int):
    """Return a list of points (.id, .score, .payload) sorted by similarity."""
    return client.query_points(
        collection_name=name, query=query_vector, limit=top_k, with_payload=True
    ).points