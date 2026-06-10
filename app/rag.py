"""RAG core: retrieve relevant chunks -> build a grounded, cited prompt.

Grounding + citations + abstention (відмова, коли відповіді нема в контексті) — це
ключові «production» вимоги якості: бот відповідає ТІЛЬКИ з наданих чанків, цитує
їх [chunk_id], і каже «не знаю» замість галюцинації.
"""
from __future__ import annotations
from app.config import settings
from app.embeddings import embed_query
from app.vectorstore import get_client, search

_client = get_client()

SYSTEM_PROMPT = (
    "You are a precise assistant for the FastAPI documentation.\n"
    "Rules:\n"
    "1. Answer ONLY using the provided context snippets.\n"
    "2. If the answer is not in the context, say you don't know — never invent.\n"
    "3. Cite the snippets you use by their [source] tag.\n"
    "4. Prefer including a short code example when the context has one."
)


def retrieve_by_vector(query_vector: list[float], top_k: int | None = None):
    """Пошук top-k чанків за вже готовим вектором (без повторного embed)."""
    return search(_client, settings.chunks_collection, query_vector, top_k or settings.top_k)


def retrieve(query: str, top_k: int | None = None):
    """Embed the query and return top-k chunks from Qdrant."""
    return retrieve_by_vector(embed_query(query), top_k)


def build_context(hits) -> tuple[str, list[str]]:
    """Format retrieved chunks into a context block + list of source ids."""
    blocks, sources = [], []
    for h in hits:
        cid = h.payload.get("chunk_id", str(h.id))
        blocks.append(f"[{cid}]\n{h.payload.get('text', '')}")
        sources.append(cid)
    return "\n\n---\n\n".join(blocks), sources


def build_messages(query: str, context: str) -> list[dict]:
    """Assemble chat messages. User query is wrapped in tags (injection defense)."""
    user = (
        f"Context snippets:\n\n{context}\n\n"
        "----------\n"
        f"<user_question>{query}</user_question>\n\n"
        "Answer using only the context above and cite the [source] tags you used."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]