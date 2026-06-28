"""LLM-as-judge for the safety eval.

A judge should not grade its own model family. The
assistant under test runs on gpt-4o-mini (demo-pro tier), so the judge defaults to
a different, stronger model (gpt-4o) reached over the same OpenRouter key. The judge
is asked for a single JSON object with an explicit rubric (no "is it good?").
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import Counter

from openai import OpenAI

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from app.config import settings                       # noqa: E402

# Under test = gpt-4o-mini (demo-pro). Judge = a different, stronger model.
JUDGE_MODEL = "openai/gpt-4o"

_client = OpenAI(base_url=settings.openrouter_base_url, api_key=settings.openrouter_api_key)


def _extract_json(text: str) -> dict:
    """Pull the first {...} object out of the judge reply (tolerates ```json fences)."""
    m = re.search(r"\{.*\}", text.strip(), re.DOTALL)
    if not m:
        return {"_parse_error": True, "_raw": text[:300]}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"_parse_error": True, "_raw": text[:300]}


def judge_json(system: str, user: str, model: str = JUDGE_MODEL,
               retries: int = 2, temperature: float = 0.0) -> dict:
    """Call the judge and return its parsed JSON verdict."""
    last = {"_parse_error": True, "_raw": ""}
    for _ in range(retries):
        resp = _client.chat.completions.create(
            model=model, temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        last = _extract_json(resp.choices[0].message.content or "")
        if not last.get("_parse_error"):
            return last
    return last


def judge_vote(system: str, user: str, label_of, n: int = 3,
               model: str = JUDGE_MODEL, temperature: float = 0.5) -> tuple:
    """Multi-judge ensemble for *critical* decisions: call the
    judge n times at temperature>0 so the votes actually vary, then take the majority.

    `label_of(verdict_dict) -> hashable label`. Returns (majority_label, [labels], [verdicts]).
    """
    labels, verdicts = [], []
    for _ in range(n):
        v = judge_json(system, user, model=model, temperature=temperature)
        verdicts.append(v)
        labels.append(label_of(v))
    majority = Counter(labels).most_common(1)[0][0]
    return majority, labels, verdicts