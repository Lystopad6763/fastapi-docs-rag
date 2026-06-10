"""Тест /chat/stream — друкує SSE-кадри в міру надходження (без curl-escaping мороки).

Передумова: піднятий `uvicorn app.main:app --port 8000`.
Запуск:
    python scripts/test_stream.py
    python scripts/test_stream.py "How to upload a file in FastAPI?"
"""
from __future__ import annotations
import sys
import httpx

URL = "http://localhost:8000/chat/stream"


def main() -> None:
    msg = sys.argv[1] if len(sys.argv) > 1 else "How do I declare an optional query parameter?"
    headers = {"X-API-Key": "demo-pro"}        # без цього хедера ендпоінт поверне 401
    with httpx.stream("POST", URL, json={"message": msg}, headers=headers, timeout=60) as r:
        for line in r.iter_lines():
            if line:
                print(line, flush=True)


if __name__ == "__main__":
    main()