"""Retrieval test against the REAL index (Qdrant).

Embeds a handful of realistic questions and prints the top-3 retrieved chunks
(heading + score). Unlike toy sentences, this reflects real search quality.

Run:  python scripts/test_retrieval.py
Prerequisites: scripts/index.py already run (docs_chunks collection is non-empty).
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
    "Annotated Query max_length validation",       # exact term — probes a known weak spot
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