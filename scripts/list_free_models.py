"""Вивести АКТУАЛЬНІ безкоштовні моделі OpenRouter (pricing = $0). Ключ не потрібен.

Запуск:  python scripts/list_free_models.py
"""
from __future__ import annotations
import json
import urllib.request

data = json.load(urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30))

free = []
for m in data.get("data", []):
    p = m.get("pricing", {})
    try:
        if float(p.get("prompt", "1")) == 0 and float(p.get("completion", "1")) == 0:
            free.append(m["id"])
    except (TypeError, ValueError):
        continue

for mid in sorted(free):
    print(mid)
print(f"\nВсього безкоштовних моделей: {len(free)}")