# Deploying to Fly.io

The API is stateless except for two managed services it talks to over the network:
**Qdrant** (vectors: document chunks + semantic cache) and **Redis** (rate-limit counters).
Locally these run in Docker (`docker compose up -d`); in the cloud we point the same
config at **Qdrant Cloud** and **Upstash Redis**, then run the container on **Fly.io**.

Everything below is reproducible from a fresh machine. Free tiers are enough.

---

## 1. Qdrant Cloud (vectors)

1. Sign up at https://cloud.qdrant.io and create a **free** cluster (1GB).
2. Copy the **cluster URL** (looks like `https://xxxx.eu-central.aws.cloud.qdrant.io:6333`).
3. Create an **API key** and copy it.

You do not create collections by hand — the app creates them automatically. Collection names
are namespaced (`fastapi_docs_chunks`, `fastapi_semantic_cache`) via `[env]` in `fly.toml`, so
a shared or reused Qdrant cluster never collides with other projects' collections.

## 2. Upstash Redis (rate limiting)

1. Sign up at https://upstash.com and create a **free** Redis database.
2. Pick the region closest to the Fly region (Frankfurt → `eu-central-1`).
3. Copy the **`rediss://` connection URL** (TLS; includes user + password).

## 3. Fly.io (the app)

1. Install flyctl: https://fly.io/docs/flyctl/install/ and `fly auth signup` (a card is
   required for verification; the small machine we use fits the free allowance).
2. From the project root, create the app (it reuses the provided `fly.toml`):

   ```bash
   fly apps create fastapi-docs-rag --org personal   # pick a unique name; update fly.toml app= to match
   ```

3. Set the secrets (never commit these — they are injected at runtime). The quickest way is to
   import them straight from your `.env` (flyctl never prints the values):

   ```bash
   fly secrets import --stage < .env
   ```

   …or set them explicitly:

   ```bash
   fly secrets set \
     OPENAI_API_KEY="sk-..." OPENROUTER_API_KEY="sk-or-..." \
     QDRANT_URL="https://xxxx.eu-central.aws.cloud.qdrant.io:6333" QDRANT_API_KEY="<key>" \
     REDIS_URL="rediss://default:<password>@<host>:6379" \
     LANGFUSE_PUBLIC_KEY="pk-lf-..." LANGFUSE_SECRET_KEY="sk-lf-..."
   ```

   (Langfuse keys are optional — omit them to run without tracing. The non-secret collection
   names live in `fly.toml`, not here.)

4. Deploy:

   ```bash
   fly deploy
   ```

## 4. Build the index in the cloud

The image ships the corpus (`data/docs/`) but not the vectors. Build them once into
Qdrant Cloud via the admin endpoint (enterprise key):

```bash
curl -X POST https://<your-app>.fly.dev/index/rebuild -H "X-API-Key: demo-enterprise"
# -> {"status":"ok","collection":"fastapi_docs_chunks","documents":50,"chunks":509,"points":509}
```

Alternatively, index from your machine before deploying — point `.env` at the cloud cluster and run
`python scripts/index.py --collection fastapi_docs_chunks`. This also confirms cloud connectivity early.

## 5. Verify

```bash
# liveness
curl https://<your-app>.fly.dev/health
# -> {"status":"ok","active_streams":0,"aborted_streams":0}

# a real question (SSE stream)
curl -N -X POST https://<your-app>.fly.dev/chat/stream \
  -H "X-API-Key: demo-pro" -H "Content-Type: application/json" \
  -d '{"message":"How do I upload a file in FastAPI?"}'
```

Put the public URL in the README once it answers.

---

### Notes

- **Streaming:** `auto_stop_machines = false` in `fly.toml` keeps the machine warm so an
  SSE stream is never cut by a cold start.
- **Cost DB:** SQLite lives on the machine's ephemeral disk and resets on redeploy. That is
  fine for a demo; attach a Fly volume if you need the cost history to persist.
- **Memory:** 512MB is enough for the API. If startup OOMs, bump `memory` in `fly.toml` to
  `1024mb` and `fly deploy` again.
