# ARCHITECTURE — `fastapi-docs-rag`

> Постійна карта проєкту. Якщо повернувся після паузи й нічого не пам'ятаєш — читай цей файл.
> README = швидкий старт; цей файл = «як воно влаштоване і ЧОМУ».

---

## 1. Що це і навіщо

**Production-grade RAG API**, що відповідає на питання про **документацію FastAPI**.
Не «скрипт, що кличе LLM», а сервіс із усіма продакшн-шарами: auth, rate-limit,
semantic cache, cost tracking, multi-provider fallback.

- **ДЗ10** — збудувати цей сервіс (11 вимог §1–§11).
- **ДЗ13** — контейнеризувати (multi-stage Docker <800MB).
- **Корпус:** FastAPI `tutorial/` docs → **509 чанків** у Qdrant.

### Глосарій (щоб не плутатись)
| Скор. | Що це |
|---|---|
| **RAG** | Retrieval-Augmented Generation: знайти релевантні фрагменти → дати LLM як контекст |
| **SSE** | Server-Sent Events: стрім токенів у браузер (`text/event-stream`) |
| **TTFT** | Time To First Token — затримка до першого токена (UX-метрика стрімінгу) |
| **BM25** | лексичний (keyword) пошук; ловить точні ідентифікатори (`UploadFile`) |
| **RRF** | Reciprocal Rank Fusion: зливає ранжування dense+BM25 (k=60) |
| **CB** | Circuit Breaker: тимчасово вимикає модель, що часто падає |
| **TTL** | Time To Live: термін життя запису в кеші (1 год) |

---

## 2. Потік запиту `/chat/stream` (життєвий цикл)

```
HTTP POST /chat/stream  {message}  + Header X-API-Key
   │
   ├─ require_api_key (auth.py) ───────────── 401 якщо ключ невалідний
   ├─ check_rate_limit (ratelimit.py) ─────── 429 + Retry-After якщо бюджет вичерпано
   │
   └─ generate()  [SSE-стрім]
        │
        ├─ embed_query (1×, OpenAI) ─────────── один вектор на cache І retrieval
        │
        ├─ cache.lookup (Qdrant) ─────────────┐
        │      ├─ HIT (cosine>0.90, не TTL): стрімимо кеш → done(cost=0, cache_hit=true)
        │      └─ MISS ↓
        │
        ├─ retrieve_by_vector → top-3 чанки (Qdrant)
        ├─ build_context + build_messages ──── grounded-промпт (цитати + abstention)
        ├─ stream_chat (llm.py) ────────────── OpenRouter, ланцюжок моделей + CB
        │      └─ async for token: yield SSE {"type":"token"}
        │            (якщо клієнт відключився → return, НЕ логуємо/не кешуємо)
        ├─ cache.store ──────────────────────── зберегти відповідь (TTL 1год)
        ├─ log_request (SQLite) ─────────────── вартість + латентність + ttft
        ├─ record_usage (Redis) ─────────────── списати реальні токени з бюджету
        └─ done {sources, usage, cost_usd, cache_hit:false}
```

**Ключова ідея:** усе всередині `try` → будь-яка помилка йде клієнту як `SSE {"type":"error"}`,
а не як 500 посеред стріму.

---

## 3. Хронологія побудови (як ми до цього дійшли)

Будували шар за шаром, перевіряючи кожен. Порядок = логіка залежностей (спершу дані, потім сервіс).

| # | Крок | Файли | § ДЗ | Статус |
|---|---|---|---|---|
| 1 | venv + залежності + `.env` | `requirements.txt`, `.env` | — | ✅ |
| 2 | Інфра: Qdrant + Redis | `docker-compose.yml` | — | ✅ |
| 3 | Завантажити корпус (резолв code-include'ів) | `scripts/fetch_docs.py` | — | ✅ |
| 4 | Structure-aware чанкінг | `app/chunking.py` | §1 | ✅ |
| 5 | Embeddings (OpenAI) | `app/embeddings.py` | §1 | ✅ |
| 6 | Vector store + індексація (509 чанків) | `app/vectorstore.py`, `scripts/index.py` | §1 | ✅ |
| 7 | RAG core: retrieve + grounded-промпт | `app/rag.py` | §1 | ✅ |
| 8 | FastAPI app + SSE-стрім | `app/main.py` | §1,§2 | ✅ |
| 9 | Auth: X-API-Key + 3 тарифи | `app/auth.py` | §3 | ✅ |
| 10 | Cost tracking + `/usage/*` | `app/cost.py`, `app/pricing.py` | §6 | ✅ |
| 11 | Rate limiting (Redis token bucket) | `app/ratelimit.py` | §4 | ✅ |
| 12 | Multi-provider fallback + circuit breaker | `app/llm.py` | §7 | ✅ |
| 13 | Semantic cache (Qdrant, TTL) | `app/cache.py` | §5 | ✅ |
| 14 | Retrieval eval (dense vs hybrid, per-style) | `eval/` | — | ✅ |
| 15 | **Security** (injection, length-limit, output filter) | `app/security.py` | §8 | ⬜ next |
| 16 | **Concurrency** (semaphore, disconnect, counters) | `app/main.py` | §9 | ⬜ |
| 17 | **Observability** (Langfuse v4) | — | §10 | ⬜ |
| 18 | `/index/rebuild` (адмін) | `app/main.py` | — | ⬜ |
| 19 | **ДЗ13:** multi-stage Docker + метрики | `Dockerfile`, `.dockerignore` | §11 | ⬜ |

**Зроблено 7/11 вимог ДЗ10 + eval-розділ.** Далі: §8 → §9 → §10 → ДЗ13.

---

## 4. Карта файлів (роль + ключові рішення)

### `app/` — застосунок
| Файл | Роль | Ключове рішення / ЧОМУ |
|---|---|---|
| `config.py` | усі налаштування (pydantic-settings, читає `.env`) | один `settings` об'єкт; секрети лише з `.env` |
| `schemas.py` | `ChatRequest {message}` | Q&A-бот без історії → одне поле |
| `chunking.py` | structure-aware чанкер | 1 чанк = 1 секція по заголовках; **код у ```…``` ніколи не ріжеться**; breadcrumb заголовків додається в текст (краще embedding) |
| `embeddings.py` | OpenAI embeddings | `text-embedding-3-small` (1536d); **без torch** → малий Docker; `sort by index` (порядок батча) |
| `vectorstore.py` | Qdrant: колекції, upsert, search | HNSW + cosine; та сама обгортка і для чанків, і для кешу |
| `rag.py` | retrieve top-k + grounded-промпт | system-промпт: відповідай ЛИШЕ з контексту, цитуй `[chunk_id]`, кажи «не знаю» (anti-галюцинація); запит у `<user_question>` (injection defense) |
| `main.py` | FastAPI app + усі ендпоінти + конвеєр | `init_db()`+`ensure_cache_collection()` на старті; уся логіка стріму в `generate()`; помилки → SSE error |
| `auth.py` | X-API-Key + 3 тарифи | кожен tier має `tokens_per_min` (для rate-limit) + `models` (ланцюжок fallback) |
| `ratelimit.py` | Redis token bucket | лічильник = **реальні токени**, не запити; INCR+EXPIRE, вікно 60с; check ДО, record ПІСЛЯ |
| `llm.py` | OpenRouter стрім + fallback + CB | timeout 15с на **встановлення** стріму; `NON_RETRYABLE={401,403,422}` (решта → наступна модель); CB: 5 фейлів/60с → пропуск; рахунок токенів через `include_usage` (fallback — tiktoken) |
| `cache.py` | semantic cache у Qdrant | HIT якщо cosine > **0.90**; TTL 1год через `expire_at` (Qdrant не має вбуд. TTL); кеш глобальний |
| `pricing.py` | ціни моделей (USD/1M ток.) | єдине джерело правди; невідома модель = $0 |
| `cost.py` | SQLite-лог + агрегації | таблиця `request_costs`; `/usage/breakdown` рахує cache-rate, fallback-rate, p95 latency |

### `scripts/` — утиліти й тести (запуск: `python scripts/<x>.py`)
| Файл | Що робить |
|---|---|
| `fetch_docs.py` | завантажує FastAPI docs → `data/docs/`; **резолвить `{* docs_src/*.py *}` include'и** (інакше код відсутній) |
| `index.py` | чанкінг → embed → Qdrant (509 чанків) |
| `ask.py` | `python scripts/ask.py "питання"` — відповідь у терміналі |
| `test_embeddings.py` | sanity ембедингу |
| `test_retrieval.py` | retrieval на реальному індексі |
| `test_rag.py` | показати побудований grounded-промпт |
| `test_stream.py` | тест `/chat/stream` SSE (шле `X-API-Key: demo-pro`) |
| `test_ratelimit.py` | довести 429 |
| `test_fallback.py` | fallback (невалідна primary-модель) |
| `test_cache.py` | MISS → HIT |
| `list_free_models.py` | живий список `:free`-моделей OpenRouter |

### `eval/` — retrieval-оцінка (для звіту)
| Файл | Що |
|---|---|
| `build_dataset.py` | генерує датасет: 15 чанків × 3 стилі (natural / keyword / identifier), seed=42 |
| `evaluate.py` | dense vs hybrid (Okapi/Plus/weighted), overall + **per-style**; метрики recall@k, src_recall@k, MRR |
| `dataset.json` | згенерований датасет (gold = chunk_id + source) |
| `hyde.json`, `rewrites.json` | артефакти експериментів (HyDE / query-rewrite — **зашкодили**, лишені для історії) |

### Корінь
| Файл | Що |
|---|---|
| `requirements.txt` | залежності (версії пінимо за фактично встановленими) |
| `.env` / `.env.example` | секрети (`.env` у `.gitignore`!) / шаблон |
| `docker-compose.yml` | Qdrant (6333/6334) + Redis (6379), named volumes |
| `data/docs/` | корпус (50 .md) · `data/costs.db` — SQLite (у `.gitignore`) |
| `README.md` | швидкий старт + статус |
| `JOURNAL.md` | приватний лог рішень+помилок (у `.gitignore`, не публікується) |
| `ARCHITECTURE.md` | цей файл |

---

## 5. Ключові інженерні рішення (звідки «production»)

1. **Один embedding на запит** — кешуємо й шукаємо тим самим вектором (не два виклики ембедера).
2. **Grounding + цитати + abstention** — бот не галюцинує: відповідає лише з чанків, цитує `[id]`, каже «не знаю».
3. **Rate-limit по токенах, не по запитах** — справедливо до різних розмірів запитів; вікно 60с у Redis.
4. **Fallback-ланцюжок per tier + circuit breaker** — впала primary-модель → наступна; постійно падає → пропускаємо.
5. **Semantic cache** — близькі перефрази (cosine>0.90) віддаємо безкоштовно й миттєво.
6. **Cost tracking з p95/cache-rate/fallback-rate** — видимість витрат і поведінки (не «чорна скринька»).
7. **Structure-aware чанкінг** — код цілим, breadcrumb заголовків у тексті → кращий retrieval.
8. **Eval-driven retrieval** — зміряли dense vs hybrid; `hybrid_plus` найкращий, `src_recall@3=0.933`.

### Підсумок eval (для звіту)
| method | recall@1 | recall@3 | recall@5 | src_recall@3 | mrr |
|---|---|---|---|---|---|
| dense | 0.40 | 0.622 | 0.778 | 0.911 | 0.532 |
| hybrid_okapi | 0.511 | 0.733 | 0.822 | 0.933 | 0.62 |
| **hybrid_plus** | **0.556** | 0.711 | **0.844** | **0.933** | **0.653** |

> **Урок:** BM25-токенізація мусить зривати пунктуацію (`re.findall(r"[a-z0-9_]+")`), інакше код-ідентифікатори не матчаться. Фікс підняв identifier-MRR 0.402→0.652 і «перевернув» результат на користь hybrid.
> **Future work:** вмикання hybrid у прод-retrieval + cross-encoder reranker (для строгого recall@1).

---

## 6. Головні налаштування (`config.py`)
| Параметр | Значення | Сенс |
|---|---|---|
| `embed_model` / `embed_dim` | `text-embedding-3-small` / 1536 | простір ембедингів |
| `top_k` | 3 | скільки чанків у контекст |
| `chunk_tokens` / overlap | 500 / 50 | (чанкер має власний `MAX_TOKENS=700`) |
| `cache_threshold` | 0.90 | поріг cosine для HIT |
| `cache_ttl_seconds` | 3600 | TTL кешу |
| `max_input_chars` | 4000 | ліміт запиту (§8) |
| `max_concurrent_llm` | 20 | семафор (§9) |
| `llm_timeout_seconds` | 15.0 | таймаут на встановлення стріму |

---

## 7. Як запустити
```powershell
.venv\Scripts\Activate.ps1            # має з'явитись (.venv) у промпті
pip install -r requirements.txt
# .env: вставити OPENROUTER_API_KEY і OPENAI_API_KEY
docker compose up -d                  # Qdrant + Redis
python scripts/fetch_docs.py          # один раз
python scripts/index.py               # один раз → 509 чанків
uvicorn app.main:app --reload --port 8000
python scripts/test_stream.py         # перевірка
```

**API-ключі:** `demo-free` (5K tok/min) · `demo-pro` (20K) · `demo-enterprise` (100K).

---

## 8. Граблі (з досвіду — деталі в JOURNAL.md)
- **Завжди перевіряй `(.venv)` у промпті** — інакше підхопиться чужий Python (`ModuleNotFoundError`).
- **`docker compose up -d` ПЕРЕД запуском** — app падає, якщо Qdrant down (`WinError 10061`).
- **Зміна `.env` або пакетні правки коду → ПОВНИЙ рестарт uvicorn** (`--reload` ненадійний, тримає старий процес).
- **OpenRouter:** порожній ключ → 401; Credit limit=0 блокує навіть `:free` → 403; `:free` часто 429 (upstream).
- **PowerShell:** `&`/пробіли в шляху → лапки; `bare .py` не запускається → `python scripts/x.py`.