"""Model prices — a FALLBACK for cost calculation.

The authoritative source is `usage.cost` from OpenRouter (the amount actually charged). This
table is used only when the provider does not return a cost. USD per 1M tokens; unknown model = $0.
Cross-checked against openrouter.ai/api/v1/models in 2026-06 (prices drift — re-verify periodically).
"""

PRICING: dict[str, dict[str, float]] = {
    "meta-llama/llama-3.1-8b-instruct":      {"input": 0.02, "output": 0.03},
    "meta-llama/llama-3.2-3b-instruct":      {"input": 0.051, "output": 0.335},
    "openai/gpt-4o-mini":                    {"input": 0.15, "output": 0.60},
    "openai/gpt-4o":                         {"input": 2.50, "output": 10.00},
    "google/gemini-2.5-flash-lite":          {"input": 0.10, "output": 0.40},
    "anthropic/claude-sonnet-4.5":           {"input": 3.00, "output": 15.00},
    # :free models cost $0
    "meta-llama/llama-3.3-70b-instruct:free": {"input": 0.0, "output": 0.0},
}


def chat_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICING.get(model, {"input": 0.0, "output": 0.0})
    return input_tokens / 1_000_000 * p["input"] + output_tokens / 1_000_000 * p["output"]