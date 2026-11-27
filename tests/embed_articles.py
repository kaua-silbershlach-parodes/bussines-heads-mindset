"""Reference embedding generator for the real embedding test.

Runs Oracle's official augmented models (models/) with ONNX Runtime +
extensions, embeds every article (title + body) and every chunk, and
stores the vectors in the database. This is the outside-the-database
route Oracle's own material describes for platforms where loading the
model into the database is blocked (Windows Free, ORA-40216); on the
Linux lab the suite populates the same columns in-database with
VECTOR_EMBEDDING, and the precision test proves both routes agree.

Connection via environment: ORACLE_LAB_USER (default "mindset"),
ORACLE_LAB_PW (default: the workshop's published lab password),
ORACLE_LAB_DSN (default localhost:1521/FREEPDB1).
"""
import json
import os
import sys
from array import array

import numpy as np
import oracledb

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from engine import MODELS, embed, load_session  # noqa: E402

USER = os.environ.get("ORACLE_LAB_USER", "mindset")
PW = os.environ.get("ORACLE_LAB_PW", "Welcome_12345")
DSN = os.environ.get("ORACLE_LAB_DSN", "localhost:1521/FREEPDB1")

corpus = json.load(open(os.path.join(ROOT, "corpus.json"), encoding="utf-8"))
chunks = json.load(open(os.path.join(ROOT, "data", "article_chunks.json"), encoding="utf-8"))

conn = oracledb.connect(user=USER, password=PW, dsn=DSN)
cur = conn.cursor()

for model, col in (("minilm", "minilm_vector"), ("e5", "e5_vector")):
    session = load_session(MODELS[model])
    docs = [f"{c['title']}. {c['text']}" for c in corpus]
    vecs = embed(session, docs)
    done = 0
    for c, vec in zip(corpus, vecs):
        cur.execute(
            f"UPDATE articles SET {col} = :v WHERE id = :i",
            v=array("f", vec.tolist()), i=c["id"],
        )
        done += 1
    conn.commit()
    print(f"articles {model}: embedded {done}/{len(corpus)}")

    vecs = embed(session, [c["text"] for c in chunks])
    done = 0
    for c, vec in zip(chunks, vecs):
        cur.execute(
            f"UPDATE article_chunks SET {col} = :v WHERE id = :i",
            v=array("f", vec.tolist()), i=c["id"],
        )
        done += 1
    conn.commit()
    print(f"chunks {model}: embedded {done}/{len(chunks)}")

print("articles with both vectors:",
      cur.execute("SELECT COUNT(*) FROM articles WHERE e5_vector IS NOT NULL AND minilm_vector IS NOT NULL").fetchone()[0])
print("chunks with both vectors:",
      cur.execute("SELECT COUNT(*) FROM article_chunks WHERE e5_vector IS NOT NULL AND minilm_vector IS NOT NULL").fetchone()[0])
conn.close()
