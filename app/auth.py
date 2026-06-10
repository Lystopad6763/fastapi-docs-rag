"""Auth: X-API-Key + тарифи.

3 хардкод-ключі. Кожен tier має:
  - tokens_per_min — бюджет для rate-limit,
  - models — ланцюжок моделей (primary + fallback).
"""
from fastapi import Header, HTTPException

API_KEYS: dict[str, dict] = {
    "demo-free": {
        "tier": "free",
        "tokens_per_min": 5_000,
        "models": [
            "meta-llama/llama-3.1-8b-instruct",
            "google/gemini-flash-1.5",
            "meta-llama/llama-3.2-3b-instruct",
        ],
    },
    "demo-pro": {
        "tier": "pro",
        "tokens_per_min": 20_000,
        "models": [
            "openai/gpt-4o-mini",
            "google/gemini-flash-1.5",
            "meta-llama/llama-3.1-8b-instruct",
        ],
    },
    "demo-enterprise": {
        "tier": "enterprise",
        "tokens_per_min": 100_000,
        "models": [
            "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o-mini",
        ],
    },
}


async def require_api_key(x_api_key: str = Header(default="")) -> dict:
    """FastAPI-залежність: 401, якщо ключ відсутній або невалідний."""
    cfg = API_KEYS.get(x_api_key)
    if not cfg:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key")
    return {"key": x_api_key, **cfg}