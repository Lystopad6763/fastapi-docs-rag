"""LLM client over OpenRouter with a fallback chain and a circuit breaker.

stream_chat tries models in order:
  - 15s timeout on ESTABLISHING the stream (not on full generation — streams start fast);
  - a retryable error (timeout / network / 429 / 5xx / invalid model) -> try the next model;
  - a non-retryable error (401/403/422) -> propagate to the client, no fallback;
  - circuit breaker: 5+ failures for a model within 60s -> skip it for 60s.

`stats` is populated with the model actually used + fallback_used + input/output tokens.
"""
from __future__ import annotations
import asyncio
import time
from collections.abc import AsyncIterator

import tiktoken
from openai import AsyncOpenAI, APIStatusError, APITimeoutError, APIConnectionError
from app.config import settings

_client = AsyncOpenAI(
    base_url=settings.openrouter_base_url,
    api_key=settings.openrouter_api_key,
)
_enc = tiktoken.get_encoding("cl100k_base")

# 401/403/422 are genuine client errors -> NO fallback. (400/404 from OpenRouter for an
# invalid model are treated as retryable so that fallback kicks in precisely on a bad model.)
NON_RETRYABLE = {401, 403, 422}
CB_THRESHOLD = 5          # failures per window -> open the circuit
CB_WINDOW = 60
_failures: dict[str, list[float]] = {}    # model -> timestamps of recent failures


def _count_messages(messages: list[dict]) -> int:
    return sum(len(_enc.encode(m.get("content", ""))) + 4 for m in messages)


def _circuit_open(model: str) -> bool:
    """True if the model has accumulated >= CB_THRESHOLD failures in the last CB_WINDOW seconds."""
    now = time.time()
    fails = [t for t in _failures.get(model, []) if now - t < CB_WINDOW]
    _failures[model] = fails
    return len(fails) >= CB_THRESHOLD


def _record_failure(model: str) -> None:
    _failures.setdefault(model, []).append(time.time())


async def stream_chat(messages: list[dict], models: list[str] | str | None = None,
                      stats: dict | None = None) -> AsyncIterator[str]:
    if models is None:
        models = [settings.llm_model]
    elif isinstance(models, str):
        models = [models]

    last_err: Exception | None = None
    for i, model in enumerate(models):
        if _circuit_open(model):
            continue                          # circuit breaker: temporarily skip this model

        try:
            stream = await asyncio.wait_for(
                _client.chat.completions.create(
                    model=model, messages=messages, stream=True,
                    temperature=0.1, stream_options={"include_usage": True},
                ),
                timeout=settings.llm_timeout_seconds,
            )
        except asyncio.TimeoutError as e:
            _record_failure(model); last_err = e; continue
        except (APITimeoutError, APIConnectionError) as e:
            _record_failure(model); last_err = e; continue
        except APIStatusError as e:
            if e.status_code in NON_RETRYABLE:
                raise                         # 401/403/422 -> propagate to the client
            _record_failure(model); last_err = e; continue

        # success -> commit to this model and stream
        if stats is not None:
            stats["model"] = model
            stats["fallback_used"] = i > 0    # not the primary -> a fallback was used
        parts: list[str] = []
        usage = None
        async for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                parts.append(delta)
                yield delta
        if stats is not None:
            if usage:
                stats["input_tokens"] = usage.prompt_tokens
                stats["output_tokens"] = usage.completion_tokens
                # OpenRouter reports the ACTUAL charged cost -> more authoritative than our PRICING
                stats["cost_usd_api"] = getattr(usage, "cost", None)
            else:
                stats["input_tokens"] = _count_messages(messages)
                stats["output_tokens"] = len(_enc.encode("".join(parts)))
        return                                # streamed successfully -> done

    raise last_err or RuntimeError("All models in the chain are unavailable")