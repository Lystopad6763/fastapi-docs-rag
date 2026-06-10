"""Тест rate-limit: шле запити з demo-free (5000 ток/хв), доки не отримає 429.

Передумова: піднятий uvicorn. УВАГА: кожен успішний запит ~15с (повільна модель),
тож до 429 може пройти ~1-2 хв (бюджет 5000 / ~700 ток на запит ≈ 7 запитів).

Запуск:  python scripts/test_ratelimit.py
"""
from __future__ import annotations
import httpx

URL = "http://localhost:8000/chat/stream"
HEADERS = {"X-API-Key": "demo-free"}
QUESTION = "How do I declare an optional query parameter in FastAPI?"

for i in range(1, 12):
    r = httpx.post(URL, json={"message": QUESTION}, headers=HEADERS, timeout=120)
    if r.status_code == 429:
        print(f"#{i}: 429  Retry-After={r.headers.get('retry-after')}  | {r.text}")
        break
    print(f"#{i}: {r.status_code} (ok)")
else:
    print("Не досягли 429 за 11 запитів — збільш кількість або зменш бюджет demo-free.")