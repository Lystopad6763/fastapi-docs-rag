"""Quick embedding smoke test (run before indexing).

Verifies three things:
  1) embeddings work, dimension = 1536, vectors are normalized (norm ~ 1.0);
  2) semantic search ranks by MEANING (relevant text on top, banana at the bottom);
  3) whether code inside the text helps or hurts matching.

Run:  python scripts/test_embeddings.py
Prerequisites: OPENAI_API_KEY in .env.
"""
from __future__ import annotations
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np                                  # noqa: E402
from app.config import settings                     # noqa: E402
from app.embeddings import embed_texts, embed_query  # noqa: E402


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    if not settings.openai_api_key:
        print("!! OPENAI_API_KEY is empty — set the key in .env"); return

    # --- 1) Sanity: dimension + normalization + determinism ------------------
    print("=" * 64)
    print("1) SANITY")
    v1 = np.array(embed_query("FastAPI query parameters"))
    v1b = np.array(embed_query("FastAPI query parameters"))  # same text a second time
    print(f"   vector dimension    : {v1.shape[0]}  (expected 1536)")
    print(f"   vector norm         : {np.linalg.norm(v1):.4f}  (OpenAI returns ~1.0)")
    print(f"   determinism (cos of the same text twice): {cos(v1, v1b):.5f}  (~1.0)")

    # --- 2) Semantic search --------------------------------------------------
    print("\n" + "=" * 64)
    print("2) SEMANTIC SEARCH — query against 4 texts")
    query = "How do I declare an optional query parameter in FastAPI?"
    docs = [
        "Query parameters: function params not part of the path are interpreted as query parameters with default values.",
        "Path parameters are part of the URL path and are always required.",
        "Use Depends() to declare dependencies that FastAPI injects into your endpoint.",
        "Bananas are a good source of potassium and grow in tropical climates.",
    ]
    qv = np.array(embed_query(query))
    dv = np.array(embed_texts(docs))
    sims = dv @ qv / (np.linalg.norm(dv, axis=1) * np.linalg.norm(qv))
    print(f"   Q: {query}")
    for rank, i in enumerate(np.argsort(-sims), 1):
        print(f"   {rank}. sim={sims[i]:.3f}  {docs[i][:62]}")
    print("   -> expect #1 = query-parameters, last = banana")

    # --- 3) Does code in the embedding help or dilute it? --------------------
    print("\n" + "=" * 64)
    print("3) EXPERIMENT: prose vs code vs prose+code")
    q2 = "example of Annotated Query validation with max_length"
    prose = "You can add extra validation and metadata to query parameters using Annotated and Query."
    code = (
        "```python\n"
        "from typing import Annotated\n"
        "from fastapi import FastAPI, Query\n"
        "app = FastAPI()\n"
        "@app.get('/items/')\n"
        "async def read_items(q: Annotated[str | None, Query(max_length=50)] = None):\n"
        "    return {'q': q}\n"
        "```"
    )
    both = prose + "\n\n" + code
    q2v = np.array(embed_query(q2))
    pv, cv, bv = (np.array(embed_query(t)) for t in (prose, code, both))
    print(f"   Q: {q2}")
    print(f"   prose only   : sim={cos(q2v, pv):.3f}")
    print(f"   code only    : sim={cos(q2v, cv):.3f}")
    print(f"   prose + code : sim={cos(q2v, bv):.3f}")
    print("   -> if 'prose+code' >= 'prose', code does NOT hurt (for our short examples)")


if __name__ == "__main__":
    main()