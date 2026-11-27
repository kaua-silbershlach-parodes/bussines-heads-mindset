#!/usr/bin/env python3
"""Engine test: 8 reference questions must rank the expected article
first under both models (16 checks)."""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from engine import MODELS, embed, load_session  # noqa: E402

QUESTIONS = [
    ("how to start a business without initial capital", "bh-001"),
    ("how to delegate authority correctly", "bh-045"),
    ("how to find an idea for a business on the internet", "bh-046"),
    ("why a business could be losing money", "bh-053"),
    ("how to develop self-discipline", "bh-041"),
    ("how to present a project to an investor", "bh-044"),
    ("how to save money while running a small business", "bh-057"),
    ("signs you can become a successful entrepreneur", "bh-013"),
]


def main():
    with open(os.path.join(ROOT, "corpus.json"), encoding="utf-8") as f:
        corpus = json.load(f)
    docs = [f"{c['title']}. {c['text']}" for c in corpus]
    passed = 0
    total = 0
    for model in ("minilm", "e5"):
        session = load_session(MODELS[model])
        doc_vecs = embed(session, docs)
        doc_vecs = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
        for question, expected in QUESTIONS:
            query_vec = embed(session, [question])[0]
            query_vec = query_vec / np.linalg.norm(query_vec)
            scores = doc_vecs @ query_vec
            top = corpus[int(np.argmax(scores))]["id"]
            total += 1
            ok = top == expected
            passed += ok
            print(f"{model} {question!r}: top {top} (expected {expected}) {'OK' if ok else 'FAIL'}")
    print(f"ENGINE TEST: {passed}/{total}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
