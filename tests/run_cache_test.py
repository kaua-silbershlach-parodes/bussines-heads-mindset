#!/usr/bin/env python3
"""Cache test: cached embeddings reproduce fresh rankings exactly.

The engine caches article/chunk embeddings keyed by the SHA-256 of
the exact texts. This test proves the cache changes nothing but the
wait: rankings from the cache equal rankings computed fresh, under
both models, and a changed text set misses the cache.

Run:  python tests/run_cache_test.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from engine import MODELS, embed, embed_cached, load_session  # noqa: E402

QUERIES = [
    "how to start a business without initial capital",
    "how to delegate authority correctly",
    "how to develop self-discipline",
]


def top10(vecs, qv):
    v = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    q = qv / np.linalg.norm(qv)
    return list(np.argsort(-(v @ q))[:10])


def main():
    corpus = json.load(open(os.path.join(ROOT, "corpus.json"), encoding="utf-8"))
    docs = [f"{c['title']}. {c['text']}" for c in corpus]
    passed = 0
    total = 0
    for model in ("minilm", "e5"):
        session = load_session(MODELS[model])
        fresh = embed(session, docs)
        cached = embed_cached(session, model, "articles", docs)
        total += 1
        same = np.array_equal(fresh, cached)
        passed += same
        print(f"{model} cached == fresh vectors: {'OK' if same else 'MISMATCH'}")
        for q in QUERIES:
            qv = embed(session, [q])[0]
            total += 1
            match = top10(fresh, qv) == top10(cached, qv)
            passed += match
            print(f"{model} top-10 cache == fresh for {q!r}: {'OK' if match else 'MISMATCH'}")
        changed = embed_cached(session, model, "articles", docs[:-1])
        total += 1
        ok = changed.shape[0] == len(docs) - 1
        passed += ok
        print(f"{model} changed text set recomputes: {'OK' if ok else 'MISMATCH'}")
        del session
    print(f"CACHE TEST: {passed}/{total}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
