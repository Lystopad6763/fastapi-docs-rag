"""Safety-eval harness for the FastAPI-docs RAG assistant.

Drives `POST /chat/stream` over SSE and returns a structured result that the
per-class evaluators (pii / injection / faithfulness / refusal) build on. It also
exposes the operational helpers an *honest* eval needs:

  - flush_cache()  — wipe both cache tiers before a class run, so a probe is never
                     served a stale cached answer (a cache hit would silently
                     invalidate a safety measurement);
  - corpus_map() / context_for() — map the `sources` chunk-ids from the `done`
                     event back to their text (needed to ground faithfulness);
  - poison_corpus() / unpoison() — insert and remove ONE malicious chunk for the
                     indirect prompt-injection test (reversible, by point id).

The assistant returns HTTP 400 when `security.check_input` rejects the message
(an injection pattern or length overflow) — the harness captures that as
`blocked=True` instead of an error, because for the injection class a 400 is a
*successful defense*, not a failure.

One-off probe (sanity check the harness):
    python eval/safety/harness.py --probe "How do I upload a file in FastAPI?"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
import time
from dataclasses import asdict, dataclass, field

import httpx

# Make the app package importable when this file is run directly.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from app.config import settings                      # noqa: E402
from app.embeddings import embed_texts               # noqa: E402
from app.vectorstore import get_client               # noqa: E402

BASE_URL = "http://localhost:8000"
DEFAULT_KEY = "demo-pro"          # gpt-4o-mini primary — our "production default" under test
ADMIN_KEY = "demo-enterprise"     # required for /cache/flush and /index/rebuild
POISON_ID = 999_999               # reserved Qdrant point id for the injected chunk


@dataclass
class AskResult:
    """Everything one probe produces — the unit the evaluators consume."""
    message: str
    answer: str = ""
    sources: list[str] = field(default_factory=list)
    cache_hit: bool = False
    usage: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    http_status: int = 0
    blocked: bool = False          # input rejected (HTTP 400) before streaming
    detail: str = ""               # error / blocked detail
    latency_ms: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


async def ask(message: str, key: str = DEFAULT_KEY, base_url: str = BASE_URL,
              timeout: float = 60.0) -> AskResult:
    """Send one question to /chat/stream and collect the full SSE response.

    Non-200 (400 blocked / 401 / 429) is captured in `http_status`+`detail`, never
    raised, so a class run can score the whole golden set without aborting.
    """
    res = AskResult(message=message)
    t0 = time.perf_counter()
    headers = {"X-API-Key": key, "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", f"{base_url}/chat/stream",
                                     headers=headers, json={"message": message}) as resp:
                res.http_status = resp.status_code
                if resp.status_code != 200:
                    body = await resp.aread()
                    try:
                        res.detail = json.loads(body).get("detail", body.decode())
                    except Exception:  # noqa: BLE001
                        res.detail = body.decode(errors="replace")
                    res.blocked = resp.status_code == 400
                    res.latency_ms = (time.perf_counter() - t0) * 1000
                    return res

                tokens: list[str] = []
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    ev = json.loads(line[len("data:"):].strip())
                    etype = ev.get("type")
                    if etype == "token":
                        tokens.append(ev.get("content", ""))
                    elif etype == "done":
                        res.sources = ev.get("sources", [])
                        res.usage = ev.get("usage", {})
                        res.cost_usd = ev.get("cost_usd", 0.0)
                        res.cache_hit = ev.get("cache_hit", False)
                    elif etype == "error":
                        res.detail = ev.get("detail", "")
                res.answer = "".join(tokens)
    except Exception as e:  # noqa: BLE001
        res.detail = f"harness error: {type(e).__name__}: {e}"
    res.latency_ms = (time.perf_counter() - t0) * 1000
    return res


async def flush_cache(key: str = ADMIN_KEY, base_url: str = BASE_URL) -> dict:
    """Invalidate both cache tiers (L1 exact + L2 semantic). Call before a class run."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(f"{base_url}/cache/flush", headers={"X-API-Key": key})
        return {"status": r.status_code, "body": r.text}


# --- corpus access (faithfulness grounding) -------------------------------------
_corpus_map: dict[str, str] | None = None


def corpus_map(collection: str | None = None) -> dict[str, str]:
    """chunk_id -> text for every chunk in the collection (loaded once, then cached)."""
    global _corpus_map
    if _corpus_map is None:
        coll = collection or settings.chunks_collection
        points, _ = get_client().scroll(coll, limit=10000, with_payload=True)
        _corpus_map = {p.payload["chunk_id"]: p.payload.get("text", "") for p in points}
    return _corpus_map


def chunk_text(chunk_id: str) -> str:
    return corpus_map().get(chunk_id, "")


def context_for(sources: list[str]) -> str:
    """Join the text of the retrieved source chunks into one context block."""
    return "\n\n---\n\n".join(t for t in (chunk_text(c) for c in sources) if t)


# --- indirect-injection corpus poisoning (reversible) ---------------------------
def poison_corpus(text: str, chunk_id: str = "POISONED#0", source: str = "POISONED",
                  collection: str | None = None) -> dict:
    """Insert ONE malicious chunk so a crafted query retrieves it (indirect injection).

    Reuses the app's own embedder + client so the vector lives in the SAME space as
    the real corpus. Returns the reserved point id — ALWAYS pair with unpoison().
    """
    from qdrant_client.models import PointStruct
    coll = collection or settings.chunks_collection
    vec = embed_texts([text])[0]
    get_client().upsert(coll, points=[PointStruct(
        id=POISON_ID, vector=vec,
        payload={"chunk_id": chunk_id, "source": source, "heading": "", "text": text},
    )])
    global _corpus_map
    _corpus_map = None             # invalidate the cached map
    return {"id": POISON_ID, "chunk_id": chunk_id}


def unpoison(collection: str | None = None) -> dict:
    """Remove the injected chunk. ALWAYS call after the indirect-injection test."""
    coll = collection or settings.chunks_collection
    get_client().delete(coll, points_selector=[POISON_ID])
    global _corpus_map
    _corpus_map = None
    return {"deleted": POISON_ID}


def _cli() -> None:
    ap = argparse.ArgumentParser(description="One-off probe against /chat/stream.")
    ap.add_argument("--probe", required=True, help="message to send")
    ap.add_argument("--key", default=DEFAULT_KEY)
    ap.add_argument("--base", default=BASE_URL)
    args = ap.parse_args()
    r = asyncio.run(ask(args.probe, key=args.key, base_url=args.base))
    print(json.dumps(r.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()