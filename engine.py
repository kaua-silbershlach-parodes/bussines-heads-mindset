#!/usr/bin/env python3
"""Semantic search over the 82 Business Heads Mindset articles.

The engine is Oracle's own augmented ONNX embedding models, run
unchanged: the tokenizer and the pre/post-processing are baked into
each model file by Oracle, so the input is raw text and the output is
a 384-dimension embedding. Ranking is cosine similarity between the
query embedding and each article's embedding.

Models (in models/, Oracle's prebuilt augmented exports):
  minilm  all_MiniLM_L12_v2.onnx   - English-first, 384 dimensions
  e5      multilingual_e5_small.onnx - 100 languages, 384 dimensions

Usage:
  python engine.py "how to start a business without capital"
  python engine.py "how to delegate authority" --model e5 --top 3
  python engine.py --chunks "finding an investor"

Requires: onnxruntime, onnxruntime-extensions, numpy
(see requirements.txt). The augmented graphs use a custom tokenizer
operator from the ONNX Runtime extensions library; without it the
model files do not load.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import onnxruntime as ort
from onnxruntime_extensions import get_library_path

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "minilm": os.path.join(ROOT, "models", "all_MiniLM_L12_v2.onnx"),
    "e5": os.path.join(ROOT, "models", "multilingual_e5_small.onnx"),
}
DIMS = 384
CACHE_DIR = os.path.join(ROOT, "data", ".embedding-cache")
WEAK_MATCH = 0.35


def load_session(model_path):
    opts = ort.SessionOptions()
    opts.register_custom_ops_library(get_library_path())
    return ort.InferenceSession(
        model_path, sess_options=opts, providers=["CPUExecutionProvider"]
    )


def embed(session, texts):
    # One text per call, like the workshop's reference embedding
    # generator and like VECTOR_EMBEDDING in the database. Batching
    # changes padding inside the augmented graph and moves e5
    # distances by up to ~0.006; single-text calls keep the local
    # engine and the database on the same numbers.
    inp = session.get_inputs()[0]
    rows = []
    for text in texts:
        outs = session.run(None, {inp.name: np.array([text], dtype=object)})
        for out in outs:
            arr = np.asarray(out, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] == DIMS:
                rows.append(arr[0])
                break
        else:
            raise RuntimeError("model produced no 384-dimension embedding output")
    return np.vstack(rows)


def embed_cached(session, model_name, kind, texts):
    """Embeddings for a fixed text set, computed once and reused.

    The cache file is keyed by the SHA-256 of the exact texts, so any
    corpus change invalidates it automatically. Cached vectors are the
    same numbers embed() produces — the cache only stops the engine
    from redoing finished work on every run.
    """
    digest = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()
    path = os.path.join(CACHE_DIR, f"{model_name}-{kind}-{digest[:16]}.npy")
    if os.path.exists(path):
        vecs = np.load(path)
        if vecs.shape == (len(texts), DIMS):
            return vecs
    vecs = embed(session, texts)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(path, vecs)
    return vecs


def rank(session, docs, query, top, model_name=None, kind="docs"):
    if model_name is not None:
        doc_vecs = embed_cached(session, model_name, kind, docs)
    else:
        doc_vecs = embed(session, docs)
    query_vec = embed(session, [query])[0]
    doc_vecs = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    query_vec = query_vec / np.linalg.norm(query_vec)
    scores = doc_vecs @ query_vec
    return np.argsort(-scores)[:top], scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="the question to match against the articles")
    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        default="e5",
        help="e5 (multilingual, default) or minilm (English-first)",
    )
    parser.add_argument("--top", type=int, default=3, help="how many results to print")
    parser.add_argument(
        "--chunks",
        action="store_true",
        help="search the article chunks (data/article_chunks.json) instead of whole articles",
    )
    args = parser.parse_args()

    session = load_session(MODELS[args.model])
    if args.chunks:
        with open(os.path.join(ROOT, "data", "article_chunks.json"), encoding="utf-8") as f:
            chunks = json.load(f)
        order, scores = rank(
            session, [c["text"] for c in chunks], args.query, args.top,
            model_name=args.model, kind="chunks",
        )
        for rank_no, idx in enumerate(order, 1):
            c = chunks[idx]
            print(f"{rank_no}. [{c['article_id']} chunk {c['chunk_index']}] {scores[idx]:.3f}")
            print(f"   {c['text'][:110]}...")
        if len(order) and scores[order[0]] < WEAK_MATCH:
            print("Weak match: the collection may not cover this question.")
        return 0

    with open(os.path.join(ROOT, "corpus.json"), encoding="utf-8") as f:
        corpus = json.load(f)
    docs = [f"{item['title']}. {item['text']}" for item in corpus]
    order, scores = rank(
        session, docs, args.query, args.top,
        model_name=args.model, kind="articles",
    )
    for rank_no, idx in enumerate(order, 1):
        print(f"{rank_no}. [{corpus[idx]['id']}] {corpus[idx]['title']} — {scores[idx]:.3f}")
        print(f"   {corpus[idx]['article_file']}")
    if len(order) and scores[order[0]] < WEAK_MATCH:
        print("Weak match: the collection may not cover this question.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
