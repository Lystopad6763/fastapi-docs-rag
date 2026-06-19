"""Index the corpus into Qdrant.

Pipeline: data/docs/*.md  ->  structure-aware chunking (by heading)  ->  OpenAI
embeddings  ->  Qdrant.

Run:  python scripts/index.py
Prerequisites: Qdrant running (docker compose up -d) and OPENAI_API_KEY in .env.
"""
from __future__ import annotations
import argparse
import pathlib
import sys

# allow `import app...` when run as a script
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.config import settings                       # noqa: E402
from app.chunking import chunk_markdown, prose_only, ntok  # noqa: E402
from app.embeddings import embed_texts                # noqa: E402
from app.vectorstore import get_client, ensure_collection, upsert_chunks  # noqa: E402


def main(collection: str, no_code: bool) -> None:
    docs = sorted(pathlib.Path(settings.docs_dir).glob("*.md"))
    if not docs:
        print(f"No .md files in {settings.docs_dir} — run `python scripts/fetch_docs.py` first")
        return

    # 1) Chunking
    chunks = []
    for path in docs:
        chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), source=path.stem))
    toks = [c.tokens for c in chunks]
    print(f"{len(docs)} documents -> {len(chunks)} chunks "
          f"(tokens: min={min(toks)} max={max(toks)} avg={sum(toks) // len(toks)})")

    # 2) What we embed: the full text (with code) OR prose only (--no-code mode).
    #    The payload always keeps the FULL text (c.text), so the bot still returns
    #    code and BM25 stays accurate.
    if no_code:
        embed_input = [prose_only(c.text) for c in chunks]
        etoks = [ntok(t) for t in embed_input]
        print(f"--no-code MODE: embedding prose only "
              f"(embed-input tokens: avg={sum(etoks) // len(etoks)} vs {sum(toks) // len(toks)} with code)")
    else:
        embed_input = [c.text for c in chunks]

    print(f"Embedding {len(chunks)} chunks via {settings.embed_model} -> collection '{collection}' ...")
    vectors = embed_texts(embed_input)

    # 3) Write to Qdrant (recreate the collection from scratch)
    client = get_client()
    ensure_collection(client, collection, recreate=True)
    ids = list(range(len(chunks)))
    payloads = [
        {"chunk_id": f"{c.source}#{idx}", "source": c.source, "heading": c.heading, "text": c.text}
        for idx, c in enumerate(chunks)
    ]
    upsert_chunks(client, collection, ids, vectors, payloads)

    count = client.count(collection).count
    print(f"Done. Collection '{collection}' in Qdrant: {count} points.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Index the corpus into Qdrant.")
    ap.add_argument("--no-code", action="store_true",
                    help="embed prose only (code stays in the payload but NOT in the vector)")
    ap.add_argument("--collection", default=None,
                    help=f"collection name (default: {settings.chunks_collection})")
    args = ap.parse_args()
    main(args.collection or settings.chunks_collection, args.no_code)