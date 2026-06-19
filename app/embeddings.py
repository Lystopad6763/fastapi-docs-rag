"""OpenAI embeddings — text-embedding-3-small (1536d), cosine metric.

No torch dependency, which keeps the Docker image small. The same functions are reused
by both indexing (chunks) and retrieval (queries) so that all vectors live in the same space.
"""
from __future__ import annotations
import httpx
from openai import OpenAI
from app.config import settings

# Keep the HTTPS connection to the embeddings API warm across requests. The default
# httpx keep-alive (5s) expires while an LLM answer streams (~3-5s), so the next query
# would pay a full TLS handshake (~2.5s) instead of reusing the socket (~0.1s). A longer
# keep-alive makes cache-hit latency consistently low.
_http = httpx.Client(
    limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=120.0),
    timeout=httpx.Timeout(30.0, connect=10.0),
)
_client = OpenAI(api_key=settings.openai_api_key, http_client=_http)


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