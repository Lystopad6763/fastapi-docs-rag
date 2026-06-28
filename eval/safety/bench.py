"""Latency + cost benchmark for the eval.

A production eval pipeline reports latency percentiles; a readiness report should
state what a request costs and how slow it is. Sends a set of varied FastAPI questions (cache
flushed first, so these are real cold-path numbers) and reports p50/p95/mean latency and the
per-request + total cost.

Run:  python eval/safety/bench.py
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

import harness                                            # noqa: E402

OUT = HERE / "results" / "bench.json"
QUESTIONS = [
    "How do I declare a path parameter in FastAPI?",
    "How do I upload a file?",
    "How do I add CORS middleware?",
    "How do I handle errors with HTTPException?",
    "How do I declare query parameters?",
    "How do I use a response_model?",
    "How do I run background tasks?",
    "How do I read cookie parameters?",
    "How do I implement OAuth2 with JWT?",
    "How do I serve static files?",
    "How do I test a FastAPI app?",
    "How do I add numeric validation to a path parameter?",
]


def _pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = int(round((p / 100) * (len(xs) - 1)))
    return xs[k]


async def run(key: str, base: str) -> list[dict]:
    await harness.flush_cache(base_url=base)
    recs = []
    for q in QUESTIONS:
        r = await harness.ask(q, key=key, base_url=base)
        recs.append({"q": q, "latency_ms": round(r.latency_ms, 1), "cost_usd": r.cost_usd,
                     "cache_hit": r.cache_hit, "out_tokens": r.usage.get("output_tokens")})
        print(f"  {r.latency_ms:7.0f} ms  ${r.cost_usd:.6f}  | {q[:44]}")
    return recs


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Latency/cost benchmark.")
    ap.add_argument("--key", default=harness.DEFAULT_KEY)
    ap.add_argument("--base", default=harness.BASE_URL)
    args = ap.parse_args()

    print(f"=== BENCH | key={args.key} | base={args.base} ===")
    recs = asyncio.run(run(args.key, args.base))
    lat = [x["latency_ms"] for x in recs]
    cost = [x["cost_usd"] for x in recs]
    summary = {
        "n": len(recs),
        "latency_ms": {"p50": round(_pct(lat, 50), 1), "p95": round(_pct(lat, 95), 1),
                       "mean": round(statistics.mean(lat), 1)},
        "cost_usd": {"total": round(sum(cost), 6), "avg_per_request": round(statistics.mean(cost), 6)},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "results": recs}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("\n--- BENCH summary ---")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
