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

Install the local engine's requirements (the two models arrive through
Git LFS):

```bash
git lfs pull
pip install -r requirements.txt
```

### Use case 1 — ask the collection a question (local, no database)

```bash
python engine.py "how to start a business without capital" --model minilm --top 3
```

```text
1. [bh-001] 15 tips on how to start your business without initial capital — 0.850
2. [bh-032] Help for aspiring entrepreneurs — 0.533
3. [bh-010] Why Start a Business from the End — 0.491
```

### Use case 2 — ask in another language (multilingual e5)

The e5 model covers 100 languages, so the question does not need to be
in English:

```bash
python engine.py "como delegar autoridade corretamente" --model e5 --top 3
```

```text
1. [bh-045] How to delegate authority correctly — 0.845
2. [bh-066] Don't stand still. Motivation for every day — 0.798
3. [bh-059] How to convince a person: 10 simple tips — 0.796
```

### Use case 3 — find the exact passage, not just the article

`--chunks` searches the 771 article chunks and cites the article and
chunk that answer the question:

```bash
python engine.py --chunks "finding an investor" --top 3
```

```text
1. [bh-032 chunk 1] 0.880
2. [bh-032 chunk 6] 0.877
3. [bh-010 chunk 10] 0.877
```

### Use case 4 — search inside Oracle AI Database 26ai

With the lab schema prepared (models loaded in-database, tables
populated from `data/articles.dmp` or `data/articles_schema.sql`),
`db_engine.py` runs the same search with the database's own stack.
Exact search embeds the query inside the database:

```bash
python db_engine.py "how to start a business without capital" --model minilm --top 3
```

```text
1. [bh-001] 15 tips on how to start your business without initial capital — 0.8499
2. [bh-032] Help for aspiring entrepreneurs — 0.5330
3. [bh-010] Why Start a Business from the End — 0.4907
```

The same ranking as the local engine — the precision test proves the
top-10 identical, order and distances, on 8/8 reference queries.

### Use case 5 — approximate search over the vector indexes

```bash
python db_engine.py "how to delegate authority correctly" --mode approx --top 3
```

```text
1. [bh-045] How to delegate authority correctly — 0.9104
2. [bh-041] How to develop self-discipline — 0.8418
3. [bh-059] How to convince a person: 10 simple tips — 0.8418
```

### Use case 6 — hybrid search (vector + text fusion)

```bash
python db_engine.py "finding an investor" --mode hybrid --top 3
```

```text
1. [bh-032] Help for aspiring entrepreneurs — score 70.71 (vector 71.48, text 63.00)
   You can earn by searching for investors who can be useful for both business beginners...
2. [bh-001] 15 tips on how to start your business without initial capital — score 66.65 (vector 68.02, text 53.00)
   1. Find a partner who will invest. 2. Negotiate with partners about the share of profits...
3. [bh-011] What is the need for risk in business? — score 59.75 (vector 65.33, text 4.00)
```

### Use case 7 — provider embeddings from the Private AI container

```bash
python db_engine.py "how to save money while running a small business" --mode provider --model minilm --top 3
```

```text
1. [bh-057] How to save money while running a small business? — 0.7933
2. [bh-053] Why your business could be losing money — 0.5263
3. [bh-034] 7 millionaire tips — 0.5091
```

### Use case 8 — RAG retrieval over the chunks

```bash
python db_engine.py "how to start a business without capital" --mode rag
```

```text
Retrieved context:
  [bh-001 chunk 2] 15 tips on how to start your business without initial capital
  [bh-001 chunk 3] 15 tips on how to start your business without initial capital
Top article (15 tips on how to start your business without initial capital) split for the prompt: 18 chunks at a 10-word maximum (UTL_TO_CHUNKS).
```

Grounded generation additionally needs an LLM endpoint configured for
the lab host; retrieval works without one.

### Use case 9 — read the corpus over REST (no SQL connection)

```bash
python db_engine.py --mode rest
curl "http://localhost:8080/ords/mindset/articles/?limit=100"
```

The ORDS endpoint serves all 82 articles as JSON, with pagination
(`limit`/`offset`).

### Use case 10 — restore the database from the dump

```bash
impdp mindset/<password>@localhost:1521/FREEPDB1 \
  directory=DATA_PUMP_DIR dumpfile=articles.dmp \
  tables=ARTICLES,ARTICLE_CHUNKS,ARTICLE_IMAGES
```

The dump carries the 82 articles 100% — bodies, chunks, images, and
both vector columns populated (round-trip proof in the tests section).

### Use case 11 — rebuild the vectors outside the database

On platforms where loading the model into the database is blocked
(Windows Free, ORA-40216), generate the embeddings with the same model
files and store them — the route Oracle's own material describes, and
the one the precision test proves exact on 8/8 queries for e5:

```bash
python tests/embed_articles.py
```

### Use case 12 — verify everything

```bash
python tests/run_engine_test.py        # 16/16, both models
python tests/run_structure_test.py     # 10/10, corpus = articles 100%
python tests/run_full_suite_linux.py   # 16 PASS + 2 BLOCKED
python tests/run_precision_test.py     # parks-style exact top-10, PASS
```

Connection settings for the database modes come from the environment:
`ORACLE_LAB_USER` (default `mindset`), `ORACLE_LAB_PW`,
`ORACLE_LAB_DSN` (default `localhost:1521/FREEPDB1`), and `ORDS_URL`
(default `http://localhost:8080/ords/mindset/articles/`).

Copyright © A Elite Estrangeira.
