"""CrossEncoder reranker (optional, gated by settings.rerank_enabled).

Pattern: dense retrieval produces top-N candidates -> the CrossEncoder reads each
[query, chunk] pair with full cross-attention -> re-ranks -> top-k. More accurate than the
bi-encoder (dense) on recall@1, but slow (~hundreds of ms on CPU for ~30 pairs).

The model is loaded LAZILY (lru_cache) on first use so that importing `app` does not pull in
torch when the reranker is disabled (important for keeping the Docker image small).
"""
from __future__ import annotations
from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder   # import inside the function -> torch only when needed
    return CrossEncoder(settings.rerank_model)


def score(query: str, texts: list[str]) -> list[float]:
    """CrossEncoder relevance score for each text against the query."""
    if not texts:
        return []
    return [float(s) for s in _model().predict([(query, t) for t in texts])]


def rerank(query: str, hits: list, top_k: int) -> list:
    """hits: objects with .payload['text'] (Qdrant points). Returns top_k in the new order."""
    if not hits:
        return hits
    scores = score(query, [h.payload.get("text", "") for h in hits])
    order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
    return [hits[i] for i in order[:top_k]]