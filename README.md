# fastapi-docs-rag

A production-grade **RAG (Retrieval-Augmented Generation) API** that answers questions about the
**FastAPI documentation**. Not "a script that calls an LLM," but a full service with the layers you
need to run RAG for real: SSE streaming, API-key auth, rate limiting, semantic caching, cost
tracking, multi-provider fallback, prompt-injection defense, concurrency control, and observability.

> 📐 Deep dive into the design and data flow: [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Live demo

**Deployed:** https://fastapi-docs-rag.fly.dev  ·  Fly.io + Qdrant Cloud + Upstash Redis

Open in a browser:
- https://fastapi-docs-rag.fly.dev/health — liveness probe
- https://fastapi-docs-rag.fly.dev/docs — interactive Swagger UI

Try it from a terminal. Demo keys: `demo-free` (5K), `demo-pro` (20K), `demo-enterprise` (100K tokens/min):

```bash
# RAG answer streamed token-by-token, with sources + cost in the final event
curl -N -X POST https://fastapi-docs-rag.fly.dev/chat/stream \
  -H "X-API-Key: demo-pro" -H "Content-Type: application/json" \
  -d '{"message":"How do I upload a file in FastAPI?"}'

# usage today, and the per-model breakdown (cache-hit rate, fallback rate, p95 latency)
curl https://fastapi-docs-rag.fly.dev/usage/today     -H "X-API-Key: demo-pro"
curl https://fastapi-docs-rag.fly.dev/usage/breakdown -H "X-API-Key: demo-pro"
```

Run the full acceptance check (every §1–§11 criterion) against the live service or a local one:

```bash
python scripts/acceptance.py                      # default: the deployed URL
python scripts/acceptance.py --base http://localhost:8000 --ratelimit
```

Reproduce the deployment from scratch: [DEPLOY.md](DEPLOY.md).

---

## Features

| Capability | What it does |
|---|---|
| **RAG with sources** | Structure-aware chunking; grounded prompt with citations and abstention; the final event returns the source chunk ids |
| **SSE streaming** | Token-by-token `text/event-stream`; client-disconnect handling |
| **API-key auth** | `X-API-Key` with 3 tiers (free / pro / enterprise), each with its own model chain |
| **Rate limiting** | Redis token bucket counting **real tokens**, 60s window → `429` + `Retry-After` |
| **Semantic cache** | Qdrant, cosine threshold (0.90), 1h TTL — paraphrases hit the cache and stream instantly at $0 |
| **Cost tracking** | Per-request cost (authoritative `usage.cost` from OpenRouter), latency p95, cache/fallback rates |
| **Multi-provider fallback** | Per-tier model chain + circuit breaker (skip a model after repeated failures) |
| **Prompt-injection defense** | 4k length limit, ≥9 injection patterns → `400` + audit log, post-stream output filtering |
| **Concurrency control** | `asyncio.Semaphore` cap on concurrent LLM calls; `active_streams` / `aborted_streams` in `/health` |
| **Observability** | Langfuse tracing of the full pipeline (`embed → cache → retrieval → llm`) |
| **Admin** | `POST /index/rebuild` to re-index the corpus |

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| LLM | OpenRouter (one key → many models, with a fallback chain) |
| Embeddings | OpenAI `text-embedding-3-small` (1536d, no local `torch`) |
| Vector DB | Qdrant (document chunks + semantic cache) |
| Rate limit / counters | Redis |
| Cost DB | SQLite |
| Observability | Langfuse |
| Reranker (optional, eval) | `sentence-transformers` CrossEncoder |

## Request flow (`/chat/stream`)

```
auth (X-API-Key) → rate-limit check (Redis) → embed query (1×, OpenAI)
   → semantic cache lookup (Qdrant)
        ├─ HIT  → stream the cached answer  → done(cache_hit=true, cost=0)
        └─ MISS → retrieve top-k (Qdrant) → grounded prompt (citations + abstention)
                  → LLM stream (OpenRouter, fallback chain) → stream tokens
                  → store in cache → log cost (SQLite) → record tokens (rate limit)
                  → done(sources, usage, cost_usd)
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness + `active_streams` / `aborted_streams` |
| POST | `/chat/stream` | Main RAG endpoint (SSE); requires `X-API-Key` |
| GET | `/usage/today` | Today's requests, tokens, cost |
| GET | `/usage/breakdown` | Per-model usage, cache-hit rate, fallback rate, avg/p95 latency |
| POST | `/index/rebuild` | Re-index the corpus (enterprise tier only) |

**Demo API keys:** `demo-free` (5K tokens/min) · `demo-pro` (20K) · `demo-enterprise` (100K).

## Quick start

```powershell
# 1. environment
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. secrets: copy the template and fill in your keys
Copy-Item .env.example .env
#   set OPENAI_API_KEY and OPENROUTER_API_KEY (LANGFUSE_* are optional)

# 3. infrastructure (Qdrant + Redis)
docker compose up -d

# 4. corpus + index (one-time)
python scripts/fetch_docs.py
python scripts/index.py

# 5. run the server
uvicorn app.main:app --reload --port 8000

# 6. try it
python scripts/test_stream.py "How do I upload a file in FastAPI?"
```

## Configuration

All settings live in `app/config.py` and are read from `.env`. Key variables:

```
OPENAI_API_KEY        # embeddings
OPENROUTER_API_KEY    # LLM (chat)
QDRANT_URL            # default http://localhost:6333 (https URL for Qdrant Cloud)
QDRANT_API_KEY        # empty for a local container; required for Qdrant Cloud
REDIS_URL             # default redis://localhost:6379/0 (rediss:// for Upstash)
LANGFUSE_PUBLIC_KEY   # optional — enables tracing when both keys are set
LANGFUSE_SECRET_KEY
```

## Retrieval evaluation

The `eval/` directory contains a small benchmark that measures retrieval quality the way a real RAG
team would, comparing **dense vs hybrid (BM25 + RRF) vs CrossEncoder reranker** on Recall@1/@10,
MRR@10, and latency percentiles (see [eval/results/benchmark_results.md](eval/results/benchmark_results.md)).
Takeaway for this corpus size: **dense retrieval is the sweet spot**; the reranker improves Recall@1
by a few points but adds ~2.7s of latency, so it ships behind the `rerank_enabled` flag (off by default).

The benchmark and reranker need extra (torch-based) deps that the API itself does not — install them
only for eval: `pip install -r requirements-eval.txt`. The production image stays torch-free.

## Safety evaluation (PII · injection · faithfulness · refusal)

Beyond retrieval quality, [`eval/safety/`](eval/safety/) is a full **safety eval pipeline** that drives the
live bot over its HTTP API and scores four production risk classes — **PII leakage, prompt injection,
hallucinations/faithfulness, refusal patterns** — against gates fixed up front, using Microsoft Presidio
and an independent LLM-as-judge. The headline result: the default config is **NOT ship-ready** (indirect
prompt injection + planted-PII both leak via poisoned retrieved content), and a small retrieved-content
guardrail ([`app/guardrails.py`](app/guardrails.py), `guardrails_enabled`) takes both to **0%** with no
quality regression. Full numbers and the **ship / not-ship verdict** are in **[REPORT.md](REPORT.md)**.

## Project structure

```
app/         FastAPI application (config, auth, rag, llm, cache, cost, ratelimit,
             security, observability, indexer, main)
scripts/     Utilities + smoke tests (fetch_docs, index, test_*) and acceptance.py
eval/        Retrieval datasets and benchmark harness
data/docs/   The corpus (FastAPI tutorial docs)
docker-compose.yml         Qdrant + Redis for local development
Dockerfile / .dockerignore Torch-free multi-stage image (~428MB) for deployment
fly.toml / DEPLOY.md       Fly.io deployment config and runbook
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full file-by-file map and the rationale behind each
design decision.
