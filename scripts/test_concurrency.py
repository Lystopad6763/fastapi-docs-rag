"""Concurrency test: N parallel requests -> active_streams never exceeds the semaphore (20).

Prerequisite: uvicorn restarted after the concurrency changes.
Run:  python scripts/test_concurrency.py
Abort check (separate): start `python scripts/test_stream.py "..."`, interrupt with
                Ctrl+C, then verify that aborted_streams in GET /health increased.
"""
from __future__ import annotations
import asyncio
import httpx

URL = "http://localhost:8000/chat/stream"
HEALTH = "http://localhost:8000/health"
H = {"X-API-Key": "demo-pro"}
N = 30


async def fire(client: httpx.AsyncClient, i: int) -> None:
    try:
        async with client.stream("POST", URL, headers=H, timeout=120,
                                  json={"message": f"Explain FastAPI dependency injection, variant {i}"}) as r:
            async for _ in r.aiter_lines():
                pass
    except Exception:
        pass


async def poll(client: httpx.AsyncClient, stop: asyncio.Event, peak: list[int]) -> None:
    while not stop.is_set():
        try:
            d = (await client.get(HEALTH, timeout=5)).json()
            peak[0] = max(peak[0], d.get("active_streams", 0))
        except Exception:
            pass
        await asyncio.sleep(0.15)


async def main() -> None:
    async with httpx.AsyncClient() as client:
        stop, peak = asyncio.Event(), [0]
        watcher = asyncio.create_task(poll(client, stop, peak))
        print(f"Launching {N} parallel requests ...")
        await asyncio.gather(*[fire(client, i) for i in range(N)])
        stop.set()
        await watcher
        final = (await client.get(HEALTH)).json()
        print(f"Peak active_streams : {peak[0]}   (semaphore=20 -> expect <= 20)")
        print(f"Final /health       : {final}   (active_streams should drop to 0)")
        verdict = "OK ✅" if peak[0] <= 20 else "FAIL ❌ (exceeded the semaphore)"
        print(f"Verdict: {verdict}")


if __name__ == "__main__":
    asyncio.run(main())