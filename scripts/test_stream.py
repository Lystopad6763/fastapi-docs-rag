"""Test /chat/stream — prints SSE frames as they arrive (no curl-escaping hassle).

Prerequisite: `uvicorn app.main:app --port 8000` running.
Run:
    python scripts/test_stream.py
    python scripts/test_stream.py "How to upload a file in FastAPI?"
"""
from __future__ import annotations
import sys
import httpx

URL = "http://localhost:8000/chat/stream"


def main() -> None:
    msg = sys.argv[1] if len(sys.argv) > 1 else "How do I declare an optional query parameter?"
    headers = {"X-API-Key": "demo-pro"}        # without this header the endpoint returns 401
    with httpx.stream("POST", URL, json={"message": msg}, headers=headers, timeout=60) as r:
        for line in r.iter_lines():
            if line:
                print(line, flush=True)


if __name__ == "__main__":
    main()