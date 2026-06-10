"""Qdrant — векторне сховище: HNSW-індекс + метрика cosine.

Колекція зберігає вектор кожного чанка + payload (текст, source, heading).
HNSW і cosine — дефолт Qdrant; на нашому масштабі (~сотні векторів) навіть
Flat був би миттєвим, але HNSW нічого не коштує (index у RAM, payload на диску).
"""
from __future__ import annotations
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.config import settings


def get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


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
    """Повертає список точок (.id, .score, .payload), відсортований за similarity."""
    return client.query_points(
        collection_name=name, query=query_vector, limit=top_k, with_payload=True
    ).points