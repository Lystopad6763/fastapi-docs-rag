# Multi-stage build for the Q&A RAG API.
#
# The service reaches OpenAI (embeddings), OpenRouter (LLM), Qdrant and Redis over the
# network — there are no model weights and no torch in the image, so the runtime stays
# small (well under the 800MB target). The builder stage carries a compiler for any wheel
# that needs it; the runtime stage is a clean slim image with only the installed packages.

# --- Stage 1: builder ---------------------------------------------------------
FROM python:3.13 AS builder

WORKDIR /build
COPY requirements.txt .
# Install into a relocatable prefix so the runtime can copy just this directory.
RUN pip install --no-cache-dir --target=/deps -r requirements.txt

# --- Stage 2: runtime ---------------------------------------------------------
FROM python:3.13-slim AS runtime

# Never run as root in production.
RUN useradd --create-home --uid 1000 app
USER app
WORKDIR /home/app

ENV PYTHONPATH=/deps \
    PATH=/deps/bin:$PATH \
    PYTHONUNBUFFERED=1

# A writable data dir for the SQLite cost log (created at runtime by the app).
RUN mkdir -p /home/app/data

COPY --from=builder /deps /deps
COPY --chown=app:app app/ ./app/
COPY --chown=app:app data/docs/ ./data/docs/

EXPOSE 8000

# /health reports {"status":"ok"} only after startup completed; a bare TCP check would lie.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import httpx,sys; r=httpx.get('http://localhost:8000/health',timeout=3); sys.exit(0 if r.json().get('status')=='ok' else 1)"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
