"""RAG core: retrieve relevant chunks -> build a grounded, cited prompt.

Grounding, citations, and abstention (declining to answer when the context has no answer)
are the key production-grade quality requirements: the bot answers ONLY from the retrieved
chunks, cites them by [chunk_id], and says "I don't know" instead of hallucinating.
"""
from __future__ import annotations
from app.config import settings
from app.embeddings import embed_query
from app.vectorstore import get_client, search

_client = get_client()

SYSTEM_PROMPT = (
    "You are a Q&A assistant for the FastAPI documentation. You only answer questions "
    "about FastAPI, grounded in the documentation snippets provided below.\n"
    "Rules:\n"
    "1. Use ONLY the provided context snippets. Do NOT use outside knowledge, and do NOT "
    "infer, assume, or invent anything that is not in them.\n"
    "2. First judge whether the snippets actually answer the question. If they do not, or "
    "the question is not about FastAPI (e.g. it asks about you as an assistant, or an "
    "unrelated topic), do NOT force an answer from the snippets — briefly say that you "
    "answer questions about the FastAPI documentation and invite a FastAPI question.\n"
    "3. When the snippets do answer the question, cite each one you use by its [source] tag.\n"
    "4. Include a short code example when the context contains relevant code.\n"
    "5. Be concise. If the context is insufficient, say so rather than guess."
)


def retrieve_by_vector(query_vector: list[float], top_k: int | None = None,
                       query_text: str | None = None):
    """Retrieve the top-k chunks for an already-computed vector (no re-embedding).

    When settings.rerank_enabled is set and query_text is provided, pull a wider dense pool
    (rerank_candidates) and re-rank it with a CrossEncoder down to top-k (better recall@1,
    at the cost of a few hundred ms of latency).
    """
    k = top_k or settings.top_k
    if settings.rerank_enabled and query_text:
        from app.rerank import rerank
        cand = search(_client, settings.chunks_collection, query_vector, settings.rerank_candidates)
        cand = [h for h in cand if h.score >= settings.min_relevance_score]
        return rerank(query_text, cand, k)
    hits = search(_client, settings.chunks_collection, query_vector, k)
    return [h for h in hits if h.score >= settings.min_relevance_score]


def retrieve(query: str, top_k: int | None = None):
    """Embed the query and return top-k chunks from Qdrant."""
    return retrieve_by_vector(embed_query(query), top_k, query_text=query)


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