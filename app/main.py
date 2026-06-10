import asyncio
import json
import re
import time
import uuid

from fastapi import FastAPI, Request, Depends
from fastapi.responses import StreamingResponse

from app.schemas import ChatRequest
from app.auth import require_api_key
from app.embeddings import embed_query
from app.rag import retrieve_by_vector, build_context, build_messages
from app.llm import stream_chat
from app import cache
from app.cost import init_db, log_request, get_today, get_breakdown
from app.ratelimit import check_rate_limit, record_usage

app = FastAPI(title="Q&A FastAPI-docs Assistant", version="0.1.0")
init_db()                        # таблиця costs
cache.ensure_cache_collection()  # колекція semantic_cache


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _stream_text(text: str) -> list[str]:
    """Розбити готовий текст на дрібні шматки для стрімінгу кешованої відповіді."""
    return re.findall(r"\S+\s*|\s+", text)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request, auth: dict = Depends(require_api_key)):
    await check_rate_limit(auth["key"], auth["tokens_per_min"])   # 429 до стріму

    async def generate():
        t0 = time.perf_counter()
        ttft_ms = None
        stats: dict = {}
        try:
            # ОДИН embedding -> і для cache, і для retrieval (не два виклики ембедера)
            qv = await asyncio.to_thread(embed_query, req.message)

            # --- Semantic cache lookup ---
            cached = await asyncio.to_thread(cache.lookup, qv)
            if cached is not None:
                for piece in _stream_text(cached["response"]):
                    if await request.is_disconnected():
                        return
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                    yield _sse({"type": "token", "content": piece})
                latency_ms = (time.perf_counter() - t0) * 1000
                await asyncio.to_thread(
                    log_request, request_id=str(uuid.uuid4()), api_key=auth["key"],
                    model="cache", input_tokens=0, output_tokens=0,
                    latency_ms=latency_ms, ttft_ms=ttft_ms or latency_ms,
                    cache_hit=True, fallback_used=False,
                )
                yield _sse({
                    "type": "done", "sources": cached.get("sources", []),
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                    "cost_usd": 0.0, "cache_hit": True,
                })
                return

            # --- MISS: retrieval + LLM ---
            hits = await asyncio.to_thread(retrieve_by_vector, qv)
            context, sources = build_context(hits)
            messages = build_messages(req.message, context)

            parts: list[str] = []
            async for token in stream_chat(messages, models=auth["models"], stats=stats):
                if await request.is_disconnected():
                    return                          # клієнт пішов -> НЕ логуємо/не кешуємо
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000
                parts.append(token)
                yield _sse({"type": "token", "content": token})

            # зберегти відповідь у кеш
            answer = "".join(parts)
            await asyncio.to_thread(cache.store, qv, req.message, answer, sources, stats.get("model", ""))

            # вартість + rate-limit
            latency_ms = (time.perf_counter() - t0) * 1000
            in_tok = stats.get("input_tokens", 0)
            out_tok = stats.get("output_tokens", 0)
            cost = await asyncio.to_thread(
                log_request, request_id=str(uuid.uuid4()), api_key=auth["key"],
                model=stats.get("model", ""), input_tokens=in_tok, output_tokens=out_tok,
                latency_ms=latency_ms, ttft_ms=ttft_ms or latency_ms,
                cache_hit=False, fallback_used=stats.get("fallback_used", False),
            )
            await record_usage(auth["key"], in_tok + out_tok)
            yield _sse({
                "type": "done", "sources": sources,
                "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
                "cost_usd": round(cost, 6), "cache_hit": False,
            })
        except Exception as e:                      # noqa: BLE001
            yield _sse({"type": "error", "detail": str(e)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/usage/today")
async def usage_today(auth: dict = Depends(require_api_key)):
    return get_today(auth["key"])


@app.get("/usage/breakdown")
async def usage_breakdown(auth: dict = Depends(require_api_key)):
    return get_breakdown(auth["key"])