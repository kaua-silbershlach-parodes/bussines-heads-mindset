# bussines-heads-guide
Entrepreneurs are people who not only have distinctive thinking but also perform unique actions. You can get closer to the mindset of an entrepreneur with us.

## The collection

- **82 articles** in `articles/` (`001`–`082`), 29,172 words of article body.
- **82 paired images** in `images/`, every one 3200×1800.
- `corpus.json` carries the 82 articles as search records (`bh-001`–`bh-082`):
  title, full body text, article and image paths, word/character counts, and a
  SHA-256 of each body.
- `data/article_chunks.json` carries **771 chunks** (40-word windows). Joining an
  article's chunks reconstructs its body 100% — proven by the structure test.

## The search engine

The same Oracle augmented ONNX engine as the intelligent-writing-engine, now
over this collection:

- `engine.py` — local semantic search over the 82 articles (and `--chunks`
  over the 771 chunks), with Oracle's two prebuilt augmented models in
  `models/` (all-MiniLM-L12-v2 and multilingual-e5-small, 384 dimensions,
  served through Git LFS).
- `db_engine.py` — the same search on Oracle AI Database 26ai: exact,
  approximate (HNSW/IVF), hybrid, provider, RAG over the chunks, and ORDS REST
  modes.
- `data/articles_schema.sql` — the `ARTICLES`, `ARTICLE_CHUNKS`, and
  `ARTICLE_IMAGES` tables with their vector and hybrid indexes.

## The data dump — the 82 articles 100%

`data/articles.dmp` (3,735,552 bytes) is an Oracle Data Pump export carrying
all three tables with both vector columns populated in-database:

- `ARTICLES`: **82/82** articles, **82/82** vector pairs.
- `ARTICLE_CHUNKS`: **771/771** chunks, **771/771** vector pairs.
- `ARTICLE_IMAGES`: **82/82** paired image records.

Round-trip proof: the dump was re-imported into a clean schema — 82 articles
with **82/82** body SHA-256 matches against `corpus.json`, and the chunks
reconstruct **82/82** article bodies.

## Tests

- `tests/run_engine_test.py` — **16/16**: 8 reference questions rank the
  expected article first under both models.
- `tests/run_structure_test.py` — **10/10**: corpus bodies equal the article
  files 100%, chunks reconstruct every article, images paired at 3200×1800,
  dump present.
- `tests/run_full_suite_linux.py` — the complete Oracle test suite from the
  official template, adapted to this content: **16 PASS + 2 BLOCKED**.
  Exact search agrees with the local engine **16/16** (top-1, both models)
  and the q1 MiniLM top-10 set matches exactly; ORDS REST, external table
  (82 rows), approximate search, five distance metrics, IVF and HNSW indexes
  (HNSW scan in the plan), hybrid search (top article `bh-001`), container
  embeddings, chunking, CLIP text encoder (512 dimensions), CLIP
  image-to-image over the paired images (nearest image is itself), and RAG
  over the chunks (771/771 embedded, retrieval returns `bh-001`) all pass.
  The 2 BLOCKED are text generation (T13 and the generation half of T14):
  the running Private AI container serves no LLM on this lab host, so the
  engine retrieves and guides; it does not generate new text.
- `tests/run_precision_test.py` — the real Oracle tests at maximum
  precision, with `tests/01-connection-test.sql`,
  `tests/02-real-vector-check.sql`, and `tests/embed_articles.py`:
  **PASS**. Connection and version (26ai, FREEPDB1 READ WRITE); real
  vector check (82/82 articles and 771/771 chunks with both vectors,
  384 dimensions); the parks-style exact top-10 — under MiniLM the
  database reproduces the local engine's top-10 **exactly on 8/8
  queries, same order and same distances to the 4th decimal**, the
  same standard as Oracle's published parks result; approximate search
  equals exact search **16/16**; provider embeddings equal in-database
  embeddings (top-1 8/8, top-10 sets 8/8); chunk search agrees between
  the database and the local engine **8/8**; the five distance metrics,
  ORDS pagination (82/82), and cross-model top-1 agreement (8/8) all
  pass. Two measured limits, stated plainly: the e5 graph executes
  with a small runtime difference between Oracle's in-database runtime
  and ONNX Runtime (vector cosine 0.9985, against 1.000000 for
  MiniLM), so cross-runtime e5 top-10 sets agree on 5/8 queries with a
  maximum distance delta of 0.0071 — while the outside-embedding
  route (local e5 vectors stored in the database, searched with
  `TO_VECTOR`) reproduces the top-10 **exactly on 8/8 queries, order
  and distances**; and at chunk level the expected article is in the
  top-5 chunks on 8/8 queries (top-1 on 5/8), with hybrid search
  ranking the expected article first on 7/8. Embeddings are computed
  one text per call everywhere — batching moves e5 distances by up to
  ~0.006 and breaks exact reproduction.

## Usage

```bash
pip install -r requirements.txt
python engine.py "how to start a business without capital"
python engine.py --chunks "finding an investor"
python tests/run_engine_test.py
python tests/run_structure_test.py
```

Copyright © A Elite Estrangeira.
