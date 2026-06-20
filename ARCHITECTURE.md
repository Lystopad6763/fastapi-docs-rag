# ARCHITECTURE — `fastapi-docs-rag`

> Living map of the project. If you come back after a break and don't remember anything, read this file.
> README = quick start; this file = "how it works and WHY".

---

## 1. What this is and why

**Production-grade RAG API** that answers questions about the **FastAPI documentation**.
Not "a script that calls an LLM," but a service with all the production layers: auth, rate-limit,
semantic cache, cost tracking, multi-provider fallback.

- **The service** is the RAG API with its 11 production capabilities.
- **Containerization:** multi-stage Docker (<800MB).
- **Corpus:** FastAPI `tutorial/` docs → **509 chunks** in Qdrant.

### Glossary (so nothing gets confused)
| Abbr. | What it is |
|---|---|
| **RAG** | Retrieval-Augmented Generation: find relevant fragments → give them to the LLM as context |
| **SSE** | Server-Sent Events: stream tokens to the browser (`text/event-stream`) |
| **TTFT** | Time To First Token — latency until the first token (a streaming UX metric) |
| **BM25** | lexical (keyword) search; catches exact identifiers (`UploadFile`) |
| **RRF** | Reciprocal Rank Fusion: merges dense+BM25 rankings (k=60) |
| **CB** | Circuit Breaker: temporarily disables a model that fails often |
| **TTL** | Time To Live: lifetime of a cache entry (1 hour) |

---

## 2. Request flow `/chat/stream` (lifecycle)

```
HTTP POST /chat/stream  {message}  + Header X-API-Key
   │
   ├─ require_api_key (auth.py) ───────────── 401 if the key is invalid
   ├─ check_rate_limit (ratelimit.py) ─────── 429 + Retry-After if the budget is exhausted
   │
   └─ generate()  [SSE stream]
        │
        ├─ embed_query (1×, OpenAI) ─────────── one vector for both cache AND retrieval
        │
        ├─ cache.lookup (Qdrant) ─────────────┐
        │      ├─ HIT (cosine>0.90, not TTL): stream the cache → done(cost=0, cache_hit=true)
        │      └─ MISS ↓
        │
        ├─ retrieve_by_vector → top-3 chunks (Qdrant)
        ├─ build_context + build_messages ──── grounded prompt (citations + abstention)
        ├─ stream_chat (llm.py) ────────────── OpenRouter, model chain + CB
        │      └─ async for token: yield SSE {"type":"token"}
        │            (if the client disconnects → return, do NOT log/cache)
        ├─ cache.store ──────────────────────── store the answer (TTL 1h)
        ├─ log_request (SQLite) ─────────────── cost + latency + ttft
        ├─ record_usage (Redis) ─────────────── deduct real tokens from the budget
        └─ done {sources, usage, cost_usd, cache_hit:false}
```

**Key idea:** everything runs inside `try` → any error is sent to the client as `SSE {"type":"error"}`,
not as a 500 in the middle of the stream.

---

## 3. Build timeline (how we got here)

We built it layer by layer, validating each one. The order follows the dependency logic (data first, then the service).

| # | Stage | Files | Feature | Status |
|---|---|---|---|---|
| 1 | venv + dependencies + `.env` | `requirements.txt`, `.env` | — | ✅ |
| 2 | Infra: Qdrant + Redis | `docker-compose.yml` | — | ✅ |
| 3 | Fetch the corpus (resolve code-includes) | `scripts/fetch_docs.py` | — | ✅ |
| 4 | Structure-aware chunking | `app/chunking.py` | Chunking | ✅ |
| 5 | Embeddings (OpenAI) | `app/embeddings.py` | Embeddings | ✅ |
| 6 | Vector store + indexing (509 chunks) | `app/vectorstore.py`, `scripts/index.py` | Vector store | ✅ |
| 7 | RAG core: retrieve + grounded prompt | `app/rag.py` | RAG core | ✅ |
| 8 | FastAPI app + SSE stream | `app/main.py` | API + Streaming | ✅ |
| 9 | Auth: X-API-Key + 3 tiers | `app/auth.py` | Auth | ✅ |
| 10 | Cost tracking + `/usage/*` | `app/cost.py`, `app/pricing.py` | Cost tracking | ✅ |
| 11 | Rate limiting (Redis token bucket) | `app/ratelimit.py` | Rate limiting | ✅ |
| 12 | Multi-provider fallback + circuit breaker | `app/llm.py` | Fallback | ✅ |
| 13 | Semantic cache (Qdrant, TTL) | `app/cache.py` | Semantic cache | ✅ |
| 14 | Retrieval eval (dense vs hybrid, per-style) | `eval/` | — | ✅ |
| 15 | **Security** (injection, length-limit, output filter) | `app/security.py` | Security | ✅ |
| 16 | **Concurrency** (semaphore, disconnect, counters) | `app/main.py` | Concurrency | ✅ |
| 17 | **Observability** (Langfuse v4) | `app/observability.py` | Observability | ✅ |
| 18 | `/index/rebuild` (admin) | `app/indexer.py`, `app/main.py` | Admin | ✅ |
| 19 | **Containerization:** torch-free multi-stage image (~428MB) | `Dockerfile`, `.dockerignore` | Docker | ✅ |
| 20 | **Public deployment** | `fly.toml`, `DEPLOY.md` | Deployment | ✅ |

**All 11 production capabilities are done, plus the eval section.** The service is containerized
(torch-free, ~428MB) and deployed at https://fastapi-docs-rag.fly.dev (Fly.io + Qdrant Cloud + Upstash Redis).

---

## 4. File map (role + key decisions)

### `app/` — the application
| File | Role | Key decision / WHY |
|---|---|---|
| `config.py` | all settings (pydantic-settings, reads `.env`) | a single `settings` object; secrets only from `.env` |
| `schemas.py` | `ChatRequest {message}` | Q&A bot without history → a single field |
| `chunking.py` | structure-aware chunker | 1 chunk = 1 section by headings; **code inside ```…``` is never split**; a breadcrumb of headings is added to the text (better embedding) |
| `embeddings.py` | OpenAI embeddings | `text-embedding-3-small` (1536d); **no torch** → small Docker image; `sort by index` (preserves batch order) |
| `vectorstore.py` | Qdrant: collections, upsert, search | HNSW + cosine; the same wrapper for both chunks and cache |
| `rag.py` | retrieve top-k + grounded prompt | system prompt: answer ONLY from context, cite `[chunk_id]`, say "I don't know" (anti-hallucination); the query is wrapped in `<user_question>` (injection defense) |
| `main.py` | FastAPI app + all endpoints + pipeline | `init_db()`+`ensure_cache_collection()` on startup; all stream logic in `generate()`; errors → SSE error |
| `auth.py` | X-API-Key + 3 tiers | each tier has `tokens_per_min` (for rate-limit) + `models` (fallback chain) |
| `ratelimit.py` | Redis token bucket | counter = **real tokens**, not requests; INCR+EXPIRE, 60s window; check BEFORE, record AFTER |
| `llm.py` | OpenRouter stream + fallback + CB | 15s timeout on **establishing** the stream; `NON_RETRYABLE={401,403,422}` (everything else → next model); CB: 5 failures/60s → skip; token count via `include_usage` (fallback — tiktoken) |
| `cache.py` | semantic cache in Qdrant | HIT if cosine > **0.90**; TTL 1h via `expire_at` (Qdrant has no built-in TTL); the cache is global |
| `pricing.py` | model prices (USD/1M tokens) | the single source of truth; unknown model = $0 |
| `cost.py` | SQLite log + aggregations | `request_costs` table; `/usage/breakdown` computes cache-rate, fallback-rate, p95 latency |

### `scripts/` — utilities and tests (run with: `python scripts/<x>.py`)
| File | What it does |
|---|---|
| `fetch_docs.py` | downloads the FastAPI docs → `data/docs/`; **resolves `{* docs_src/*.py *}` includes** (otherwise the code is missing) |
| `index.py` | chunking → embed → Qdrant (509 chunks) |
| `ask.py` | `python scripts/ask.py "question"` — answer in the terminal |
| `test_embeddings.py` | embedding sanity check |
| `test_retrieval.py` | retrieval against the real index |
| `test_rag.py` | show the built grounded prompt |
| `test_stream.py` | test `/chat/stream` SSE (sends `X-API-Key: demo-pro`) |
| `test_ratelimit.py` | demonstrate 429 |
| `test_fallback.py` | fallback (invalid primary model) |
| `test_cache.py` | MISS → HIT |
| `list_free_models.py` | live list of OpenRouter `:free` models |

### `eval/` — retrieval evaluation (for the report)
| File | What |
|---|---|
| `build_dataset.py` | generates the dataset: 15 chunks × 3 styles (natural / keyword / identifier), seed=42 |
| `evaluate.py` | dense vs hybrid (Okapi/Plus/weighted), overall + **per-style**; metrics recall@k, src_recall@k, MRR |
| `dataset.json` | the generated dataset (gold = chunk_id + source) |
| `hyde.json`, `rewrites.json` | experiment artifacts (HyDE / query-rewrite — **they hurt results**, kept for the record) |

### Root
| File | What |
|---|---|
| `requirements.txt` | dependencies (versions pinned to what is actually installed) |
| `.env` / `.env.example` | secrets (`.env` is in `.gitignore`!) / template |
| `docker-compose.yml` | Qdrant (6333/6334) + Redis (6379), named volumes |
| `data/docs/` | corpus (50 .md) · `data/costs.db` — SQLite (in `.gitignore`) |
| `README.md` | quick start + status |
| `ARCHITECTURE.md` | this file |

---

## 5. Key engineering decisions (where "production" comes from)

1. **One embedding per request** — we cache and search with the same vector (not two embedder calls).
2. **Grounding + citations + abstention** — the bot does not hallucinate: it answers only from chunks, cites `[id]`, and says "I don't know."
3. **Rate-limit by tokens, not by requests** — fair across different request sizes; 60s window in Redis.
4. **Per-tier fallback chain + circuit breaker** — primary model failed → next one; constantly failing → skip it.
5. **Semantic cache** — near paraphrases (cosine>0.90) are served for free and instantly.
6. **Cost tracking with p95/cache-rate/fallback-rate** — visibility into spend and behavior (not a "black box").
7. **Structure-aware chunking** — code kept whole, heading breadcrumb in the text → better retrieval.
8. **Eval-driven retrieval** — we measured dense vs hybrid; `hybrid_plus` is the best, `src_recall@3=0.933`.

### Eval summary (for the report)
| method | recall@1 | recall@3 | recall@5 | src_recall@3 | mrr |
|---|---|---|---|---|---|
| dense | 0.40 | 0.622 | 0.778 | 0.911 | 0.532 |
| hybrid_okapi | 0.511 | 0.733 | 0.822 | 0.933 | 0.62 |
| **hybrid_plus** | **0.556** | 0.711 | **0.844** | **0.933** | **0.653** |

> **Lesson:** BM25 tokenization must strip punctuation (`re.findall(r"[a-z0-9_]+")`), otherwise code identifiers don't match. The fix raised identifier-MRR 0.402→0.652 and "flipped" the result in favor of hybrid.
> **Future work:** enable hybrid in production retrieval + a cross-encoder reranker (for strict recall@1).

---

## 6. Main settings (`config.py`)
| Parameter | Value | Meaning |
|---|---|---|
| `embed_model` / `embed_dim` | `text-embedding-3-small` / 1536 | embedding space |
| `top_k` | 3 | how many chunks go into the context |
| `chunk_tokens` / overlap | 500 / 50 | (the chunker has its own `MAX_TOKENS=700`) |
| `cache_threshold` | 0.90 | cosine threshold for a HIT |
| `cache_ttl_seconds` | 3600 | cache TTL |
| `max_input_chars` | 4000 | request limit (security) |
| `max_concurrent_llm` | 20 | semaphore (concurrency) |
| `llm_timeout_seconds` | 15.0 | timeout for establishing the stream |

---

## 7. How to run
```powershell
.venv\Scripts\Activate.ps1            # (.venv) should appear in the prompt
pip install -r requirements.txt
# .env: paste OPENROUTER_API_KEY and OPENAI_API_KEY
docker compose up -d                  # Qdrant + Redis
python scripts/fetch_docs.py          # once
python scripts/index.py               # once → 509 chunks
uvicorn app.main:app --reload --port 8000
python scripts/test_stream.py         # smoke check
```

**API keys:** `demo-free` (5K tok/min) · `demo-pro` (20K) · `demo-enterprise` (100K).

---

## 8. Gotchas (from experience)
- **Always check for `(.venv)` in the prompt** — otherwise the wrong Python gets picked up (`ModuleNotFoundError`).
- **`docker compose up -d` BEFORE starting** — the app crashes if Qdrant is down (`WinError 10061`).
- **Changing `.env` or batch code edits → FULL uvicorn restart** (`--reload` is unreliable, holds the old process).
- **OpenRouter:** an empty key → 401; Credit limit=0 blocks even `:free` → 403; `:free` is often 429 (upstream).
- **PowerShell:** `&`/spaces in a path → use quotes; a bare `.py` won't run → `python scripts/x.py`.
