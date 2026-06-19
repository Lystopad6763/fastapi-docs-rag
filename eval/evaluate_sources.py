"""Source-level evaluation against the eval/retrieval_gold.json dataset.

Differs from evaluate.py: here gold is a LIST of sources (relevant_sources), not a single
chunk_id. We check whether retrieval surfaces the right documents in top-k (and for multi-doc
questions, whether it surfaces ALL of them). Abstain questions (out of corpus) are scored
separately: success means the best match is WEAK (below threshold), i.e. the bot should
have replied "I don't know".

Metrics (single + multi):
  any_hit@k    — at least one of relevant_sources in the top-k sources (topic was found)
  full_cov@k   — ALL relevant_sources present in top-k (matters for multi-doc)
  src_prec@k   — fraction of top-k sources that are relevant (noise in the window)
Abstain:
  abstain_ok   — top-1 max cosine < ABSTAIN_THRESHOLD (no confident match -> correct refusal)

Run:
  python eval/evaluate_sources.py
  python eval/evaluate_sources.py --collection docs_chunks --k 5 --method hybrid_plus
"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rank_bm25 import BM25Okapi, BM25Plus              # noqa: E402
from app.embeddings import embed_texts                 # noqa: E402
from app.config import settings                        # noqa: E402
from evaluate import load_corpus, dense_search, hybrid_search, _tok  # noqa: E402

HERE = pathlib.Path(__file__).parent
ABSTAIN_THRESHOLD = 0.45   # max cosine below this -> no relevant doc -> correct to refuse


def ordered_sources(chunk_ids: list[str], id_to_src: dict[str, str]) -> list[str]:
    """Unique sources in the order their chunks appear in the results (top source = top chunk)."""
    seen, out = set(), []
    for cid in chunk_ids:
        s = id_to_src.get(cid)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def main(collection: str | None, k: int, method: str) -> None:
    ds = json.loads((HERE / "retrieval_gold.json").read_text(encoding="utf-8"))
    ids, srcs, texts, vecs = load_corpus(collection)
    id_to_src = dict(zip(ids, srcs))
    bm25 = {"dense": None, "hybrid_okapi": BM25Okapi, "hybrid_plus": BM25Plus}[method]
    bm = bm25([_tok(t) for t in texts]) if bm25 else None

    queries = [ex["query"] for ex in ds]
    qv = [np.array(v, dtype=np.float32) for v in embed_texts(queries)]

    # retrieve a wide window so we can count sources (there are more chunks than sources)
    wide = max(k * 6, 20)
    retrieved_src, top_cos = [], []
    for v, q in zip(qv, queries):
        if method == "dense":
            chunks = dense_search(v, vecs, ids, top_k=wide)
        else:
            chunks = hybrid_search(v, q, bm, vecs, ids, top_k=wide)
        retrieved_src.append(ordered_sources(chunks, id_to_src)[:k])
        top_cos.append(float((vecs @ (v / (np.linalg.norm(v) + 1e-9))).max()))

    answerable = [ex for ex in ds if ex["type"] in ("single", "multi")]
    abstain = [ex for ex in ds if ex["type"] == "abstain"]

    any_hit = full_cov = prec = 0.0
    multi_full = multi_n = 0
    for ex, got in zip(ds, retrieved_src):
        if ex["type"] == "abstain":
            continue
        gold = set(ex["relevant_sources"])
        hit = gold & set(got)
        any_hit += 1 if hit else 0
        full_cov += 1 if gold <= set(got) else 0
        prec += len(hit) / max(len(got), 1)
        if ex["type"] == "multi":
            multi_n += 1
            multi_full += 1 if gold <= set(got) else 0

    n = max(len(answerable), 1)
    print(f"Collection: {collection or settings.chunks_collection} | method: {method} | top-{k} sources")
    print(f"Dataset: {len(ds)} questions ({len(answerable)} answerable, {len(abstain)} abstain)\n")
    print("=== ANSWERABLE (single + multi) ===")
    print(f"  any_hit@{k}   = {any_hit / n:.3f}  (topic found)")
    print(f"  full_cov@{k}  = {full_cov / n:.3f}  (all relevant sources in top-{k})")
    print(f"  src_prec@{k}  = {prec / n:.3f}  (fraction relevant among top-{k})")
    if multi_n:
        print(f"  multi full_cov@{k} = {multi_full / multi_n:.3f}  ({multi_full}/{multi_n} multi-doc questions fully covered)")

    if abstain:
        ok = sum(1 for ex, c in zip(ds, top_cos) if ex["type"] == "abstain" and c < ABSTAIN_THRESHOLD)
        print(f"\n=== ABSTAIN (out of corpus, threshold cos<{ABSTAIN_THRESHOLD}) ===")
        print(f"  abstain_ok = {ok}/{len(abstain)}  (weak top-1 -> correct refusal)")
        for ex, c in zip(ds, top_cos):
            if ex["type"] == "abstain":
                flag = "ok " if c < ABSTAIN_THRESHOLD else "LEAK"
                print(f"    [{flag}] cos={c:.3f}  {ex['query'][:60]}")

    # which questions were NOT fully covered — for diagnostics
    misses = [(ex, got) for ex, got in zip(ds, retrieved_src)
              if ex["type"] in ("single", "multi") and not set(ex["relevant_sources"]) <= set(got)]
    if misses:
        print(f"\n=== MISSES (not all relevant sources in top-{k}) ===")
        for ex, got in misses:
            missing = set(ex["relevant_sources"]) - set(got)
            print(f"  [{ex['id']}] {ex['query'][:55]}")
            print(f"      missing: {sorted(missing)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Source-level retrieval eval (multi-doc + abstain).")
    ap.add_argument("--collection", default=None)
    ap.add_argument("--k", type=int, default=5, help="how many top sources to consider (default 5)")
    ap.add_argument("--method", default="hybrid_plus",
                    choices=["dense", "hybrid_okapi", "hybrid_plus"])
    args = ap.parse_args()
    main(args.collection, args.k, args.method)
