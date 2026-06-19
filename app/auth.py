"""Authentication: X-API-Key plus per-tier configuration.

Three hardcoded demo keys. Each tier defines:
  - tokens_per_min — the budget used for rate limiting,
  - models — the model chain (primary + fallbacks) tried in order.
"""
from fastapi import Header, HTTPException

API_KEYS: dict[str, dict] = {
    "demo-free": {
        "tier": "free",
        "tokens_per_min": 5_000,
        "models": [
            "meta-llama/llama-3.1-8b-instruct",
            "google/gemini-2.5-flash-lite",
            "meta-llama/llama-3.2-3b-instruct",
        ],
    },
    "demo-pro": {
        "tier": "pro",
        "tokens_per_min": 20_000,
        "models": [
            "openai/gpt-4o-mini",
            "google/gemini-2.5-flash-lite",
            "meta-llama/llama-3.1-8b-instruct",
        ],
    },
    "demo-enterprise": {
        "tier": "enterprise",
        "tokens_per_min": 100_000,
        "models": [
            "openai/gpt-4o",
            "anthropic/claude-sonnet-4.5",
            "openai/gpt-4o-mini",
        ],
    },
}


async def require_api_key(x_api_key: str = Header(default="")) -> dict:
    """FastAPI dependency: raises 401 if the key is missing or invalid."""
    cfg = API_KEYS.get(x_api_key)
    if not cfg:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key")
    return {"key": x_api_key, **cfg}