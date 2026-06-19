"""Build a realistic evaluation dataset: 3 query styles per chunk, mirroring how users ask.

Styles:
  natural    — a full-sentence question (dense-friendly)
  keyword    — terse keywords, like a Google search (hybrid-friendly)
  identifier — exact API names / identifiers (BM25-friendly)

All 3 share the same gold target (chunk_id + source). seed=42 makes generation reproducible.
Run:  python eval/build_dataset.py
"""
from __future__ import annotations
import collections
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from openai import OpenAI                       # noqa: E402
from app.config import settings                 # noqa: E402
from app.vectorstore import get_client          # noqa: E402

N_CHUNKS = 15
random.seed(42)
GEN_MODEL = "meta-llama/llama-3.1-8b-instruct"
_oai = OpenAI(base_url=settings.openrouter_base_url, api_key=settings.openrouter_api_key)

STYLES = {
    "natural": "Write ONE natural full-sentence question a developer would ask, "
               "answerable by this FastAPI docs excerpt. Output only the question.",
    "keyword": "Write a SHORT keyword search query (3-6 words, NOT a sentence, like a Google "
               "search) a developer would type to find this. Output only the query.",
    "identifier": "Output ONLY the 1-3 key FastAPI identifiers/API names from this excerpt a "
                  "developer would search (e.g. 'UploadFile', 'Annotated Query', 'status_code'). "
                  "No sentence, no explanation.",
}


def gen(style_prompt: str, text: str) -> str:
    r = _oai.chat.completions.create(
        model=GEN_MODEL, temperature=0.3, max_tokens=40,
        messages=[{"role": "system", "content": style_prompt},
                  {"role": "user", "content": text[:1500]}],
    )
    return r.choices[0].message.content.strip().strip('"').strip("`").strip()


def main() -> None:
    points, _ = get_client().scroll(settings.chunks_collection, limit=2000, with_payload=True)
    by_source: dict[str, list] = collections.defaultdict(list)
    for p in points:
        if len(p.payload.get("text", "")) > 200:
            by_source[p.payload["source"]].append(p)

    sources = list(by_source)
    random.shuffle(sources)
    data = []
    for src in sources[:N_CHUNKS]:
        p = random.choice(by_source[src])
        for style, prompt in STYLES.items():
            q = gen(prompt, p.payload["text"])
            data.append({
                "query": q, "style": style,
                "gold_chunk_id": p.payload["chunk_id"], "gold_source": p.payload["source"],
            })
            print(f"  [{style:<10}] {src:<34} {q}")
        print()

    out = pathlib.Path(__file__).parent / "dataset.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(data)} queries ({N_CHUNKS} chunks x {len(STYLES)} styles) -> {out}")


if __name__ == "__main__":
    main()