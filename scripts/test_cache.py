"""Semantic cache test: MISS -> HIT (same query + paraphrase), with latency comparison.

Measures with `curl -N` (the streaming SSE client the assignment specifies). A buffering
HTTP client such as `httpx`/`requests` mis-times a cache HIT on Windows: it can sit ~2.5s
before delivering a small chunked SSE response that the server actually flushed in ~15ms,
which would understate the speedup. curl consumes the stream as it arrives, so the number
reflects real latency. `-w %{time_total}` reports curl's own timing (no process-spawn skew).

Flushes both cache tiers (L1 in-memory + L2 semantic) via the admin endpoint at startup,
so request #1 is a reproducible MISS even across repeated runs without a server restart.
Prerequisite: uvicorn running.

Run:  python scripts/test_cache.py
"""
from __future__ import annotations
import json
import subprocess

BASE = "http://localhost:8000"
HDR = ["-H", "X-API-Key: demo-pro", "-H", "Content-Type: application/json"]
MARK = "@@TIME@@"


def _curl(args: list[str], body: str | None = None) -> str:
    return subprocess.run(["curl", "-s", *args], input=body, capture_output=True,
                          text=True).stdout


# Fresh cache (both tiers). Flush runs inside the server process, so it also clears the
# in-memory L1 that an external client cannot reach.
print("cache flush:", _curl(["-X", "POST", f"{BASE}/cache/flush",
                              "-H", "X-API-Key: demo-enterprise"]).strip())


def ask(q: str) -> tuple[float, object]:
    body = json.dumps({"message": q})
    out = _curl(["-N", "-w", f"\n{MARK}%{{time_total}}", "-X", "POST",
                 f"{BASE}/chat/stream", *HDR, "--data-binary", "@-"], body=body)
    secs, hit = 0.0, None
    for line in out.splitlines():
        if line.startswith(MARK):
            secs = float(line[len(MARK):])
        elif line.startswith("data:"):
            try:
                ev = json.loads(line[5:])
            except Exception:
                continue
            if ev.get("type") == "done":
                hit = ev.get("cache_hit")
    return secs, hit


def main() -> None:
    q1 = "How do I declare an optional query parameter in FastAPI?"
    q2 = "Explain how to make a query parameter optional in FastAPI"
    t1, c1 = ask(q1); print(f"#1 MISS  (new query)   {t1:6.2f}s  cache_hit={c1}")
    t2, c2 = ask(q1); print(f"#2 L1    (same query)  {t2:6.2f}s  cache_hit={c2}")
    t3, c3 = ask(q2); print(f"#3 L2    (paraphrase)  {t3:6.2f}s  cache_hit={c3}")
    print()
    if t2 > 0:
        print(f"L1 exact    speedup vs MISS: {t1 / t2:5.1f}x  (expected >=5x)")
    if t3 > 0:
        print(f"L2 semantic speedup vs MISS: {t1 / t3:5.1f}x  "
              f"(embeds the query, so floored by the embedding-API latency, ~1-2s)")


if __name__ == "__main__":
    main()
