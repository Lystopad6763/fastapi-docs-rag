"""Token-based rate limiting у Redis.

Фіксоване 60-секундне вікно per API key. Лічильник = РЕАЛЬНО витрачені токени
(input+output), а не кількість запитів. Реалізація через INCR + EXPIRE
(портативно, без Lua — як радить ДЗ для Upstash).

Логіка: ПЕРЕД запитом перевіряємо, чи ключ уже не вичерпав бюджет (check);
ПІСЛЯ відповіді списуємо реальні токени (record_usage).
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
    """429 + Retry-After, якщо ключ уже вичерпав бюджет токенів у поточному вікні."""
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
    """Списати реально витрачені токени; на першому списанні почати 60с вікно."""
    k = _key(api_key)
    await _r.incrby(k, tokens)
    if await _r.ttl(k) < 0:            # -1 = ключ без TTL -> виставляємо вікно
        await _r.expire(k, WINDOW_SECONDS)