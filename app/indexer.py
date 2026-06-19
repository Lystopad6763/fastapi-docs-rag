"""Corpus re-indexing exposed as a function — backs the admin endpoint POST /index/rebuild.

Reads data/docs/*.md -> structure-aware chunking -> embed -> recreates the Qdrant collection.
(Same logic as scripts/index.py, but callable from the API.)
"""
from __future__ import annotations
import pathlib

from app.config import settings
from app.chunking import chunk_markdown, prose_only
from app.embeddings import embed_texts
from app.vectorstore import get_client, ensure_collection, upsert_chunks


def rebuild_index(collection: str | None = None, no_code: bool = False) -> dict:
    """Rebuild the index from scratch. Returns a summary (used as the endpoint response)."""
    collection = collection or settings.chunks_collection
    docs = sorted(pathlib.Path(settings.docs_dir).glob("*.md"))
    if not docs:
        return {"status": "error", "detail": f"No .md files in {settings.docs_dir} — run fetch_docs first"}

    chunks = []
    for path in docs:
        chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), source=path.stem))
    embed_input = [prose_only(c.text) if no_code else c.text for c in chunks]
    vectors = embed_texts(embed_input)

    client = get_client()
    ensure_collection(client, collection, recreate=True)
    ids = list(range(len(chunks)))
    payloads = [
        {"chunk_id": f"{c.source}#{i}", "source": c.source, "heading": c.heading, "text": c.text}
        for i, c in enumerate(chunks)
    ]
    upsert_chunks(client, collection, ids, vectors, payloads)

    return {
        "status": "ok",
        "collection": collection,
        "documents": len(docs),
        "chunks": len(chunks),
        "points": client.count(collection).count,
    }