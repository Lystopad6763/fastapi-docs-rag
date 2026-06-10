"""Тест semantic cache (§5): MISS -> HIT (той самий + перефраз), порівняння latency.

На старті чистить колекцію кешу -> відтворюваний MISS на #1.
Передумова: піднятий uvicorn.

Запуск:  python scripts/test_cache.py
"""
from __future__ import annotations
import json
import pathlib
import sys
import time

import httpx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.config import settings   # noqa: E402
from app import cache             # noqa: E402

# свіжий кеш для чистого MISS на #1
if cache._client.collection_exists(settings.cache_collection):
    cache._client.delete_collection(settings.cache_collection)
cache.ensure_cache_collection()

URL = "http://localhost:8000/chat/stream"
HEADERS = {"X-API-Key": "demo-pro"}


def ask(q: str) -> tuple[float, object]:
    t = time.perf_counter()
    r = httpx.post(URL, json={"message": q}, headers=HEADERS, timeout=120)
    dt = time.perf_counter() - t
    cache_hit = None
    for line in r.text.splitlines():
        if line.startswith("data:"):
            try:
                ev = json.loads(line[5:])
            except Exception:
                continue
            if ev.get("type") == "done":
                cache_hit = ev.get("cache_hit")
    return dt, cache_hit


def main() -> None:
    q1 = "How do I declare an optional query parameter in FastAPI?"
    q2 = "Explain how to make a query parameter optional in FastAPI"
    t1, c1 = ask(q1); print(f"#1 (новий запит)   {t1:6.2f}s  cache_hit={c1}")
    t2, c2 = ask(q1); print(f"#2 (той самий)     {t2:6.2f}s  cache_hit={c2}")
    t3, c3 = ask(q2); print(f"#3 (перефраз)      {t3:6.2f}s  cache_hit={c3}")
    if t2 > 0:
        print(f"\nПрискорення HIT vs MISS: {t1 / t2:.1f}x  (ДЗ очікує >=5x)")


if __name__ == "__main__":
    main()