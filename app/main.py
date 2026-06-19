import asyncio
import json
import re
import time
import uuid

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.config import settings
from app.schemas import ChatRequest
from app.auth import require_api_key
from app.embeddings import embed_query
from app.rag import retrieve_by_vector, build_context, build_messages
from app.llm import stream_chat
from app import cache
from app.cost import init_db, log_request, get_today, get_breakdown
from app.indexer import rebuild_index
from app.ratelimit import check_rate_limit, record_usage
from app.security import check_input, filter_output
from app import observability as obs

app = FastAPI(title="Q&A FastAPI-docs Assistant", version="0.1.0")
init_db()                        # costs table
cache.ensure_cache_collection()  # semantic_cache collection

# Cap on concurrent LLM calls + counters for active / aborted streams
_llm_semaphore = asyncio.Semaphore(settings.max_concurrent_llm)
_metrics = {"active_streams": 0, "aborted_streams": 0}


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _stream_text(text: str) -> list[str]:
    """Split a ready-made text into small pieces so a cached answer can be streamed."""
    return re.findall(r"\S+\s*|\s+", text)


@app.get("/health")
async def health():
    return {"status": "ok", **_metrics}


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request, auth: dict = Depends(require_api_key)):
    check_input(req.message, auth["key"])                         # length + injection checks -> 400
    await check_rate_limit(auth["key"], auth["tokens_per_min"])   # 429 before streaming

    async def generate():
        t0 = time.perf_counter()
        ttft_ms = None
        stats: dict = {}
        # Trace spanning the whole request (no-op if Langfuse is disabled)
        trace = obs.start_trace("chat/stream", input={"message": req.message},
                                metadata={"api_key": auth["key"], "tier": auth.get("tier", "")})
        try:
            # ONE embedding -> used for both cache and retrieval (avoids calling the embedder twice)
            es = trace.start_observation(name="embed", as_type="span")
            qv = await asyncio.to_thread(embed_query, req.message)
            es.end()

            # --- Semantic cache lookup ---
            cs = trace.start_observation(name="cache_lookup", as_type="span")
            cached = await asyncio.to_thread(cache.lookup, qv)
            cs.update(metadata={"hit": cached is not None})
            cs.end()
            if cached is not None:
                for piece in _stream_text(cached["response"]):
                    if await request.is_disconnected():
                        _metrics["aborted_streams"] += 1
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
                trace.update(output=cached["response"], metadata={"cache_hit": True})
                yield _sse({
                    "type": "done", "sources": cached.get("sources", []),
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                    "cost_usd": 0.0, "cache_hit": True,
                })
                return

            # --- MISS: retrieval + LLM ---
            rs = trace.start_observation(name="retrieval", as_type="span")
            hits = await asyncio.to_thread(retrieve_by_vector, qv, query_text=req.message)
            context, sources = build_context(hits)
            messages = build_messages(req.message, context)
            rs.update(output={"sources": sources})
            rs.end()

            parts: list[str] = []
            gen = trace.start_observation(name="llm", as_type="generation", input=messages)
            async with _llm_semaphore:              # at most N concurrent LLM streams
                _metrics["active_streams"] += 1     # count "active" only during the LLM phase (<= semaphore)
                try:
                    async for token in stream_chat(messages, models=auth["models"], stats=stats):
                        if await request.is_disconnected():
                            _metrics["aborted_streams"] += 1
                            gen.end()
                            return                  # client disconnected -> do NOT log or cache
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000
                        parts.append(token)
                        yield _sse({"type": "token", "content": token})
                finally:
                    _metrics["active_streams"] -= 1

            # store the answer in the cache
            answer = "".join(parts)
            output_filtered = filter_output(answer, auth["key"])   # did the system prompt leak?
            in_tok = stats.get("input_tokens", 0)
            out_tok = stats.get("output_tokens", 0)
            gen.update(output=answer, model=stats.get("model", ""),
                       usage_details={"input": in_tok, "output": out_tok},
                       metadata={"fallback_used": stats.get("fallback_used", False)})
            gen.end()
            await asyncio.to_thread(cache.store, qv, req.message, answer, sources, stats.get("model", ""))

            # cost + rate limiting
            latency_ms = (time.perf_counter() - t0) * 1000
            cost = await asyncio.to_thread(
                log_request, request_id=str(uuid.uuid4()), api_key=auth["key"],
                model=stats.get("model", ""), input_tokens=in_tok, output_tokens=out_tok,
                latency_ms=latency_ms, ttft_ms=ttft_ms or latency_ms,
                cache_hit=False, fallback_used=stats.get("fallback_used", False),
                cost_usd=stats.get("cost_usd_api"),   # authoritative cost from OpenRouter
                output_filtered=output_filtered,      # system-prompt leak flag
            )
            await record_usage(auth["key"], in_tok + out_tok)
            trace.update(output=answer, metadata={"cache_hit": False,
                         "model": stats.get("model", ""),
                         "fallback_used": stats.get("fallback_used", False)})
            yield _sse({
                "type": "done", "sources": sources,
                "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
                "cost_usd": round(cost, 6), "cache_hit": False,
            })
        except Exception as e:                      # noqa: BLE001
            yield _sse({"type": "error", "detail": str(e)})
        finally:
            trace.end()

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


@app.post("/index/rebuild")
async def index_rebuild(auth: dict = Depends(require_api_key)):
    """Admin: re-index the document corpus from scratch (enterprise tier only)."""
    if auth.get("tier") != "enterprise":
        raise HTTPException(status_code=403, detail="Admin only: requires enterprise tier")
    return await asyncio.to_thread(rebuild_index)