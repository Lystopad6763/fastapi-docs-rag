"""LLM-клієнт через OpenRouter з fallback-ланцюжком + circuit breaker.

stream_chat пробує моделі по черзі:
  - timeout 15с на ВСТАНОВЛЕННЯ стріму (не на всю генерацію — стрім стартує швидко);
  - retryable помилка (timeout / network / 429 / 5xx / невалідна модель) -> наступна модель;
  - non-retryable (401/403/422) -> віддаємо клієнту, без fallback;
  - circuit breaker: 5+ помилок моделі за 60с -> 60с її пропускаємо.

У stats кладе реально використану model + fallback_used + input/output tokens.
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

# 401/403/422 — справді клієнтські помилки -> НЕ fallback. (400/404 від OpenRouter на
# невалідну модель трактуємо як retryable, щоб fallback спрацьовував саме на невалідну модель.)
NON_RETRYABLE = {401, 403, 422}
CB_THRESHOLD = 5          # помилок моделі за вікно -> розмикаємо коло
CB_WINDOW = 60
_failures: dict[str, list[float]] = {}    # model -> час останніх помилок


def _count_messages(messages: list[dict]) -> int:
    return sum(len(_enc.encode(m.get("content", ""))) + 4 for m in messages)


def _circuit_open(model: str) -> bool:
    """True, якщо модель набрала >= CB_THRESHOLD помилок за останні CB_WINDOW секунд."""
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
            continue                          # circuit breaker: модель тимчасово пропускаємо

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
                raise                         # 401/403/422 -> віддаємо клієнту
            _record_failure(model); last_err = e; continue

        # успіх -> комітимось у цю модель і стрімимо
        if stats is not None:
            stats["model"] = model
            stats["fallback_used"] = i > 0    # не primary -> fallback спрацював
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
            else:
                stats["input_tokens"] = _count_messages(messages)
                stats["output_tokens"] = len(_enc.encode("".join(parts)))
        return                                # успішно відстрімили -> виходимо

    raise last_err or RuntimeError("Усі моделі ланцюжка недоступні")