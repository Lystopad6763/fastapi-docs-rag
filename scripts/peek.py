"""Qdrant inspector: browse chunks OR find candidates for labeling a dataset.

Modes:
  browse N:   python scripts/peek.py --collection docs_chunks --limit 3
  by id:      python scripts/peek.py --id 42
  full text:  python scripts/peek.py --id 42 --full
  SEARCH:     python scripts/peek.py --query "How to upload a file?" --topk 5
              prints candidate chunk_ids -> copy the one you want into the
              dataset's gold_chunk_id field.

Prerequisites: Qdrant running + OPENAI_API_KEY (for --query mode).
"""
from __future__ import annotations
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np                                   # noqa: E402
from app.config import settings                      # noqa: E402
from app.chunking import prose_only                  # noqa: E402
from app.embeddings import embed_query               # noqa: E402
from app.vectorstore import get_client, search       # noqa: E402


def _clip(s: str, full: bool, n: int = 400) -> str:
    return s if full else s[:n] + (" …[truncated]" if len(s) > n else "")


def show_point(p, full: bool) -> None:
    text = p.payload.get("text", "")
    print("=" * 72)
    print(f"chunk_id : {p.payload.get('chunk_id')}   (point id={p.id})")
    print(f"source   : {p.payload.get('source')}")
    print(f"heading  : {p.payload.get('heading')}")
    if getattr(p, "vector", None) is not None:
        v = np.array(p.vector)
        print(f"vector   : dim={v.shape[0]}  norm={np.linalg.norm(v):.4f}")
    print(f"\n--- PAYLOAD.text (full, with code; this is what BASELINE embedded) ---\n{_clip(text, full)}")
    print(f"\n--- PROSE_ONLY (no code; this is what --no-code embedded) ---\n{_clip(prose_only(text), full)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Qdrant chunk inspector.")
    ap.add_argument("--collection", default=settings.chunks_collection)
    ap.add_argument("--id", type=int, default=None, help="show a specific point by id")
    ap.add_argument("--limit", type=int, default=3, help="how many chunks to browse")
    ap.add_argument("--query", default=None, help="search for labeling candidates (prints chunk_id)")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--full", action="store_true", help="show texts in full")
    a = ap.parse_args()
    client = get_client()

    if a.query:
        hits = search(client, a.collection, embed_query(a.query), top_k=a.topk)
        print(f"Q: {a.query}   (collection: {a.collection})\n" + "-" * 72)
        for rank, h in enumerate(hits, 1):
            print(f"  {rank}. score={h.score:.3f}  chunk_id={h.payload.get('chunk_id')}  "
                  f"[{h.payload.get('source')}]  {h.payload.get('heading', '')[:48]}")
        print("\n-> copy the chunk_id you want into the gold_chunk_id field of your dataset")
        return

    if a.id is not None:
        pts = client.retrieve(a.collection, ids=[a.id], with_payload=True, with_vectors=True)
    else:
        pts, _ = client.scroll(a.collection, limit=a.limit, with_payload=True, with_vectors=True)
    for p in pts:
        show_point(p, a.full)


if __name__ == "__main__":
    main()