"""Evaluation harness: dense vs hybrid retrieval — overall and per query style.

Real users phrase queries differently (natural / keyword / identifier). The per-style
breakdown shows exactly where hybrid (BM25) wins and where dense alone is enough.

Configurations:
  dense              — cosine only (baseline)
  hybrid_okapi       — BM25Okapi + dense, fused with RRF
  hybrid_plus        — BM25Plus + dense, fused with RRF
  hybrid_w(dense2)   — BM25Okapi + dense, weighted RRF (dense weighted x2)

Run:  python eval/evaluate.py
"""
from __future__ import annotations
import argparse
import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rank_bm25 import BM25Okapi, BM25Plus        # noqa: E402
from app.config import settings                  # noqa: E402
from app.embeddings import embed_texts           # noqa: E402
from app.vectorstore import get_client           # noqa: E402

KS = [1, 3, 5]
TOP_K = 5
RRF_K = 60
N_CAND = 100
HERE = pathlib.Path(__file__).parent


def _tok(s: str) -> list[str]:
    """Tokenizer for BM25: strips punctuation/brackets/backticks while keeping code identifiers.
    'UploadFile(' / '`UploadFile`' / 'status_code=201' -> ['uploadfile'] / ['status_code','201']."""
    return re.findall(r"[a-z0-9_]+", s.lower())


def load_corpus(collection: str | None = None):
    points, _ = get_client().scroll(collection or settings.chunks_collection, limit=2000,
                                    with_payload=True, with_vectors=True)
    ids = [p.payload["chunk_id"] for p in points]
    srcs = [p.payload["source"] for p in points]
    texts = [p.payload["text"] for p in points]
    vecs = np.array([p.vector for p in points], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return ids, srcs, texts, vecs


def dense_search(qv, vecs, ids, top_k=TOP_K):
    scores = vecs @ (qv / (np.linalg.norm(qv) + 1e-9))
    return [ids[i] for i in np.argsort(scores)[::-1][:top_k]]


def hybrid_search(qv, qtext, bm25, vecs, ids, top_k=TOP_K, dense_w=1.0, bm25_w=1.0):
    k = min(N_CAND, len(ids))
    ds = vecs @ (qv / (np.linalg.norm(qv) + 1e-9))
    dense_top = np.argsort(ds)[::-1][:k]
    bs = np.array(bm25.get_scores(_tok(qtext)))
    bm25_top = np.argsort(bs)[::-1][:k]
    rrf: dict[int, float] = {}
    for rank, idx in enumerate(dense_top):
        rrf[idx] = rrf.get(idx, 0.0) + dense_w / (RRF_K + rank + 1)
    for rank, idx in enumerate(bm25_top):
        rrf[idx] = rrf.get(idx, 0.0) + bm25_w / (RRF_K + rank + 1)
    return [ids[i] for i in sorted(rrf, key=rrf.__getitem__, reverse=True)[:top_k]]


def metrics(pairs, id_to_src) -> dict:
    """pairs: list of (retrieved_ids, example)."""
    rec = {k: 0 for k in KS}
    srec = {k: 0 for k in KS}
    mrr = 0.0
    for retrieved, ex in pairs:
        gold, gsrc = ex["gold_chunk_id"], ex["gold_source"]
        for k in KS:
            if gold in retrieved[:k]:
                rec[k] += 1
            if any(id_to_src.get(c) == gsrc for c in retrieved[:k]):
                srec[k] += 1
        if gold in retrieved:
            mrr += 1.0 / (retrieved.index(gold) + 1)
    n = max(len(pairs), 1)
    out = {f"recall@{k}": round(rec[k] / n, 3) for k in KS}
    out |= {f"src_recall@{k}": round(srec[k] / n, 3) for k in KS}
    out["mrr"] = round(mrr / n, 3)
    return out


def main(collection: str | None = None, dataset_path: str | None = None) -> None:
    ds_file = pathlib.Path(dataset_path) if dataset_path else HERE / "dataset.json"
    dataset = json.loads(ds_file.read_text(encoding="utf-8"))
    ids, srcs, texts, vecs = load_corpus(collection)
    id_to_src = dict(zip(ids, srcs))
    bm25o = BM25Okapi([_tok(t) for t in texts])
    bm25p = BM25Plus([_tok(t) for t in texts])
    queries = [ex["query"] for ex in dataset]
    styles = [ex["style"] for ex in dataset]
    print(f"Collection: {collection or settings.chunks_collection} | {len(ids)} chunks | "
          f"Dataset: {len(dataset)} queries ({len(set(styles))} styles)\n")

    qv = [np.array(v, dtype=np.float32) for v in embed_texts(queries)]

    methods = {
        "dense":            [dense_search(v, vecs, ids) for v in qv],
        "hybrid_okapi":     [hybrid_search(v, t, bm25o, vecs, ids) for v, t in zip(qv, queries)],
        "hybrid_plus":      [hybrid_search(v, t, bm25p, vecs, ids) for v, t in zip(qv, queries)],
        "hybrid_w(dense2)": [hybrid_search(v, t, bm25o, vecs, ids, dense_w=2.0) for v, t in zip(qv, queries)],
    }

    # --- Overall ---
    cols = ["recall@1", "recall@3", "recall@5", "src_recall@3", "mrr"]
    print("=== OVERALL ===")
    print(f"{'method':<18}" + "".join(f"{c:>14}" for c in cols))
    print("-" * (18 + 14 * len(cols)))
    for name, results in methods.items():
        m = metrics(list(zip(results, dataset)), id_to_src)
        print(f"{name:<18}" + "".join(f"{m[c]:>14}" for c in cols))

    # --- Per style (MRR) ---
    uniq_styles = sorted(set(styles))
    print("\n=== MRR BY STYLE (where does hybrid win?) ===")
    print(f"{'style':<12}" + "".join(f"{n:>18}" for n in methods))
    print("-" * (12 + 18 * len(methods)))
    for st in uniq_styles:
        row = f"{st:<12}"
        for name, results in methods.items():
            pairs = [(r, ex) for r, ex in zip(results, dataset) if ex["style"] == st]
            row += f"{metrics(pairs, id_to_src)['mrr']:>18}"
        print(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Retrieval eval: dense vs hybrid.")
    ap.add_argument("--collection", default=None,
                    help=f"Qdrant collection (default: {settings.chunks_collection})")
    ap.add_argument("--dataset", default=None,
                    help="path to a custom .json dataset (default: eval/dataset.json)")
    args = ap.parse_args()
    main(args.collection, args.dataset)