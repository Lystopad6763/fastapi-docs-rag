"""Rate-limit test: sends requests as demo-free (5000 tok/min) until it gets a 429.

Each request asks a DIFFERENT question on purpose: identical queries would hit the
semantic cache, and cache hits cost 0 tokens (no rate-limit charge), so the budget
would never deplete. Distinct questions force cache MISSes that consume real tokens.

Prerequisite: uvicorn running. NOTE: each successful request takes ~15s (slow model),
so reaching 429 may take ~1-2 min (budget 5000 / ~700 tok per request ~= 7 requests).

Run:  python scripts/test_ratelimit.py
"""
from __future__ import annotations
import httpx

URL = "http://localhost:8000/chat/stream"
HEADERS = {"X-API-Key": "demo-free"}

QUESTIONS = [
    "How do I declare an optional query parameter in FastAPI?",
    "How do I upload a file in FastAPI?",
    "How do I define a request body with a Pydantic model?",
    "How do I use path parameters with type validation?",
    "How do I add custom response headers in FastAPI?",
    "How do I handle form data in a FastAPI endpoint?",
    "How do I set up dependency injection in FastAPI?",
    "How do I return a custom status code from a route?",
    "How do I validate query parameters with constraints?",
    "How do I structure a FastAPI app with routers?",
    "How do I serve static files in FastAPI?",
    "How do I configure CORS middleware in FastAPI?",
]

for i, question in enumerate(QUESTIONS, 1):
    r = httpx.post(URL, json={"message": question}, headers=HEADERS, timeout=120)
    if r.status_code == 429:
        print(f"#{i}: 429  Retry-After={r.headers.get('retry-after')}  | {r.text}")
        break
    print(f"#{i}: {r.status_code} (ok)")
else:
    print("Did not reach 429 — increase the question count or lower the demo-free budget.")