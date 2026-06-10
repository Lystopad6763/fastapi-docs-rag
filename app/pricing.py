"""Ціни моделей — єдине джерело правди для розрахунку вартості.

USD за 1M токенів. Моделі, яких нема в словнику, рахуються як $0.
"""

PRICING: dict[str, dict[str, float]] = {
    "meta-llama/llama-3.1-8b-instruct":      {"input": 0.05, "output": 0.08},
    "meta-llama/llama-3.2-3b-instruct":      {"input": 0.015, "output": 0.025},
    "openai/gpt-4o-mini":                    {"input": 0.15, "output": 0.60},
    "openai/gpt-4o":                         {"input": 2.50, "output": 10.00},
    "google/gemini-flash-1.5":               {"input": 0.075, "output": 0.30},
    "anthropic/claude-3.5-sonnet":           {"input": 3.00, "output": 15.00},
    # :free-моделі коштують $0
    "meta-llama/llama-3.3-70b-instruct:free": {"input": 0.0, "output": 0.0},
}


def chat_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICING.get(model, {"input": 0.0, "output": 0.0})
    return input_tokens / 1_000_000 * p["input"] + output_tokens / 1_000_000 * p["output"]