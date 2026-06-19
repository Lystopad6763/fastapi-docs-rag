"""Token-based rate limiting in Redis.

A fixed 60-second window per API key. The counter tracks ACTUAL tokens consumed
(input+output), not the number of requests. Implemented with INCR + EXPIRE
(portable, no Lua — works with managed Redis such as Upstash).

Flow: BEFORE a request, check whether the key has already exhausted its budget (check);
AFTER the response, charge the real token count (record_usage).
"""
from __future__ import annotations
import redis.asyncio as redis
from fastapi import HTTPException
from app.config import settings

_r = redis.from_url(settings.redis_url, decode_responses=True)
WINDOW_SECONDS = 60


def _key(api_key: str) -> str:
    return f"ratelimit:{api_key}"


async def check_rate_limit(api_key: str, budget: int) -> None:
    """Raise 429 + Retry-After if the key has already exhausted its token budget in the current window."""
    k = _key(api_key)
    used = int(await _r.get(k) or 0)
    if used >= budget:
        ttl = await _r.ttl(k)
        retry = ttl if ttl and ttl > 0 else WINDOW_SECONDS
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({used}/{budget} tokens per minute)",
            headers={"Retry-After": str(retry)},
        )


async def record_usage(api_key: str, tokens: int) -> None:
    """Charge the actual tokens consumed; on the first charge, start the 60s window."""
    k = _key(api_key)
    await _r.incrby(k, tokens)
    if await _r.ttl(k) < 0:            # -1 = key has no TTL -> set the window
        await _r.expire(k, WINDOW_SECONDS)