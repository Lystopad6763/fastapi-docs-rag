"""Індексація корпусу.

data/docs/*.md  ->  structure-aware чанкінг (по заголовках)  ->  OpenAI embed  ->  Qdrant.

Запуск:  python scripts/index.py
Передумова: піднятий Qdrant (docker compose up -d) і OPENAI_API_KEY у .env.
"""
from __future__ import annotations
import pathlib
import sys

# дозволяємо `import app...` при запуску як скрипт
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.config import settings                       # noqa: E402
from app.chunking import chunk_markdown               # noqa: E402
from app.embeddings import embed_texts                # noqa: E402
from app.vectorstore import get_client, ensure_collection, upsert_chunks  # noqa: E402


def main() -> None:
    docs = sorted(pathlib.Path(settings.docs_dir).glob("*.md"))
    if not docs:
        print(f"Нема .md у {settings.docs_dir} — спершу `python scripts/fetch_docs.py`")
        return

    # 1) Чанкінг
    chunks = []
    for path in docs:
        chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), source=path.stem))
    toks = [c.tokens for c in chunks]
    print(f"{len(docs)} документів -> {len(chunks)} чанків "
          f"(токени: min={min(toks)} max={max(toks)} avg={sum(toks) // len(toks)})")

    # 2) Embedding
    print(f"Ембедимо {len(chunks)} чанків через {settings.embed_model} ...")
    vectors = embed_texts([c.text for c in chunks])

    # 3) Запис у Qdrant (перестворюємо колекцію з нуля)
    client = get_client()
    ensure_collection(client, settings.chunks_collection, recreate=True)
    ids = list(range(len(chunks)))
    payloads = [
        {"chunk_id": f"{c.source}#{idx}", "source": c.source, "heading": c.heading, "text": c.text}
        for idx, c in enumerate(chunks)
    ]
    upsert_chunks(client, settings.chunks_collection, ids, vectors, payloads)

    count = client.count(settings.chunks_collection).count
    print(f"Готово. Колекція '{settings.chunks_collection}' у Qdrant: {count} точок.")


if __name__ == "__main__":
    main()