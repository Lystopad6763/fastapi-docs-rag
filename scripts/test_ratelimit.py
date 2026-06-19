"""Rate-limit test: sends requests as demo-free (5000 tok/min) until it gets a 429.

Prerequisite: uvicorn running. NOTE: each successful request takes ~15s (slow model),
so reaching 429 may take ~1-2 min (budget 5000 / ~700 tok per request ~= 7 requests).

Run:  python scripts/test_ratelimit.py
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
    print("Did not reach 429 in 11 requests — increase the count or lower the demo-free budget.")