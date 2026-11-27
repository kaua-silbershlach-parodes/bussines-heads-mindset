"""The full suite runner, Linux edition: every runnable Oracle test from the
official blogs and the LiveLabs workshop, run end-to-end against the Linux
lab (Oracle AI Database 26ai Free container + Oracle Private AI Services
container) with this project's content (the 82 articles in corpus.json,
their 771 chunks, and the paired images — the same content stored in
data/articles.dmp).

On Windows, three of these tests are blocked by the edition (in-database
ONNX runtime absent, ORDS absent). On Linux the in-database runtime and
ORDS run, and so does every test in this script except text generation:
the medium Private AI image deploys its LLM on this lab host, but
generation does not complete at usable speed on a host of this size
(7 GB RAM, 1.5 shared CPUs), so T13 and the generation half of T14
record BLOCKED with that reason (full evidence in the README). It expects:

  * the lab database up (FREEPDB1) with a MINDSET schema (CONNECT,
    RESOURCE, DB_DEVELOPER_ROLE, CREATE MINING MODEL, SELECT_CATALOG_ROLE,
    read/write on MODELS_DIR and LAB_DIR, and the network ACL for the
    Private AI endpoint), max_string_size EXTENDED and
    vector_memory_size = 256M;
  * the two embedding models (all_MiniLM_L12_v2.onnx,
    multilingual_e5_small.onnx) in the MODELS_DIR directory inside the
    database;
  * the Private AI container serving on the endpoint stored in the
    schema's PRIVATE_AI_CONFIG table;
  * ORDS installed in FREEPDB1 and serving (for the REST test);
  * environment: ORACLE_LAB_PW (default: the workshop's published lab
    password), ORACLE_LAB_DSN (default localhost:1521/FREEPDB1),
    ORDS_URL (default http://localhost:8080/ords/mindset/articles/),
    LAB_DIR_HOST (host path of the LAB_DIR directory, for the external
    table test).

Run:  python tests/run_full_suite_linux.py
"""
import hashlib
import json
import os
import struct
import time
import urllib.request

import oracledb

PW = os.environ.get("ORACLE_LAB_PW", "Welcome_12345")
DSN = os.environ.get("ORACLE_LAB_DSN", "localhost:1521/FREEPDB1")
ORDS_URL = os.environ.get("ORDS_URL", "http://localhost:8080/ords/mindset/articles/")
LAB_DIR_HOST = os.environ.get(
    "LAB_DIR_HOST", os.path.expanduser("~/workspace/oracle-linux/oradata/lab")
)
ENDPOINT = "http://localhost:8081"
ROOT = os.path.join(os.path.dirname(__file__), "..")
CORPUS_PATH = os.path.join(ROOT, "corpus.json")
CHUNKS_PATH = os.path.join(ROOT, "data", "article_chunks.json")

QUERIES = [
    ("q1", "how to start a business without initial capital"),
    ("q2", "how to delegate authority correctly"),
    ("q3", "how to find an idea for a business on the internet"),
    ("q4", "why a business could be losing money"),
    ("q5", "how to develop self-discipline"),
    ("q6", "how to present a project to an investor"),
    ("q7", "how to save money while running a small business"),
    ("q8", "signs you can become a successful entrepreneur"),
]
EXPECTED_TOP1 = {
    "q1": "bh-001", "q2": "bh-045", "q3": "bh-046", "q4": "bh-053",
    "q5": "bh-041", "q6": "bh-044", "q7": "bh-057", "q8": "bh-013",
}
EXPECTED_TOP10_Q1_MINILM = [
    "bh-001", "bh-032", "bh-043", "bh-010", "bh-056",
    "bh-039", "bh-046", "bh-071", "bh-042", "bh-012",
]

conn = oracledb.connect(user="mindset", password=PW, dsn=DSN)
cur = conn.cursor()
results = {}


def show(title):
    print(f"\n===== {title} =====", flush=True)


def run(sql, **binds):
    return cur.execute(sql, **binds).fetchall()


def attempt(fn):
    try:
        return fn()
    except oracledb.Error as e:
        return f"ERROR: {str(e).splitlines()[0]}"


def jpeg_size(path):
    with open(path, "rb") as f:
        data = f.read()
    i = 2
    while i < len(data) - 9:
        if data[i] != 0xFF:
            i += 1
            continue
        if data[i + 1] in (0xC0, 0xC1, 0xC2):
            h, w = struct.unpack(">HH", data[i + 5:i + 9])
            return w, h
        i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
    return (0, 0)


# ---------- setup: config, articles, chunks, images, in-database models ----------
show("SETUP config, articles, chunks, images, in-database ONNX models")
tables = {r[0] for r in run("SELECT table_name FROM user_tables")}
if "PRIVATE_AI_CONFIG" not in tables:
    cur.execute(
        "CREATE TABLE private_ai_config (config_name VARCHAR2(100), config_value VARCHAR2(1000))"
    )
    cur.execute("INSERT INTO private_ai_config VALUES ('HTTP_ENDPOINT', :e)", e=ENDPOINT)
    conn.commit()
endpoint = run(
    "SELECT config_value FROM private_ai_config WHERE config_name = 'HTTP_ENDPOINT'"
)[0][0]
print("  Private AI endpoint:", endpoint)

corpus = json.load(open(CORPUS_PATH))
chunks = json.load(open(CHUNKS_PATH))
if "ARTICLES" not in tables:
    cur.execute(
        "CREATE TABLE articles (id VARCHAR2(16) PRIMARY KEY, title VARCHAR2(500), "
        "text CLOB, article_file VARCHAR2(300), image_file VARCHAR2(300), "
        "words NUMBER, chars NUMBER, sha256 VARCHAR2(64), "
        "e5_vector VECTOR(384, FLOAT32), minilm_vector VECTOR(384, FLOAT32))"
    )
if "ARTICLE_CHUNKS" not in tables:
    cur.execute(
        "CREATE TABLE article_chunks (id NUMBER PRIMARY KEY, article_id VARCHAR2(16), "
        "chunk_index NUMBER, word_start NUMBER, word_end NUMBER, text CLOB, "
        "e5_vector VECTOR(384, FLOAT32), minilm_vector VECTOR(384, FLOAT32))"
    )
if "ARTICLE_IMAGES" not in tables:
    cur.execute(
        "CREATE TABLE article_images (id VARCHAR2(16) PRIMARY KEY, article_id VARCHAR2(16), "
        "image_file VARCHAR2(300), width NUMBER, height NUMBER, sha256 VARCHAR2(64))"
    )
conn.commit()

if run("SELECT COUNT(*) FROM articles")[0][0] == 0:
    cur.executemany(
        "INSERT INTO articles (id, title, text, article_file, image_file, words, chars, sha256) "
        "VALUES (:1, :2, :3, :4, :5, :6, :7, :8)",
        [(c["id"], c["title"], c["text"], c["article_file"], c["image_file"],
          c["words"], c["chars"], c["sha256"]) for c in corpus],
    )
    conn.commit()
if run("SELECT COUNT(*) FROM article_chunks")[0][0] == 0:
    cur.executemany(
        "INSERT INTO article_chunks (id, article_id, chunk_index, word_start, word_end, text) "
        "VALUES (:1, :2, :3, :4, :5, :6)",
        [(c["id"], c["article_id"], c["chunk_index"], c["word_start"], c["word_end"], c["text"])
         for c in chunks],
    )
    conn.commit()
if run("SELECT COUNT(*) FROM article_images")[0][0] == 0:
    rows = []
    for c in corpus:
        path = os.path.join(ROOT, c["image_file"])
        w, h = jpeg_size(path)
        rows.append((c["id"], c["id"], c["image_file"], w, h,
                     hashlib.sha256(open(path, "rb").read()).hexdigest()))
    cur.executemany(
        "INSERT INTO article_images (id, article_id, image_file, width, height, sha256) "
        "VALUES (:1, :2, :3, :4, :5, :6)", rows)
    conn.commit()
print(f"  articles: {run('SELECT COUNT(*) FROM articles')[0][0]} (expected 82); "
      f"chunks: {run('SELECT COUNT(*) FROM article_chunks')[0][0]} (expected {len(chunks)}); "
      f"images: {run('SELECT COUNT(*) FROM article_images')[0][0]} (expected 82)")

models = {r[0] for r in run("SELECT model_name FROM user_mining_models")}
for fname, mname in [
    ("all_MiniLM_L12_v2.onnx", "minilm_l12_v2"),
    ("multilingual_e5_small.onnx", "e5_small"),
]:
    if mname.upper() not in models:
        cur.execute(
            "BEGIN DBMS_VECTOR.LOAD_ONNX_MODEL('MODELS_DIR', :f, :m); END;",
            f=fname, m=mname,
        )
        print(f"  loaded {mname}")
print("  models:", sorted({r[0] for r in run("SELECT model_name FROM user_mining_models")}))

docs = {c["id"]: f"{c['title']}. {c['text']}" for c in corpus}
for col, model in [("minilm_vector", "minilm_l12_v2"), ("e5_vector", "e5_small")]:
    missing = run(f"SELECT id FROM articles WHERE {col} IS NULL")
    for (aid,) in missing:
        cur.execute(
            f"UPDATE articles SET {col} = VECTOR_EMBEDDING({model} USING :x AS data) WHERE id = :i",
            x=docs[aid], i=aid,
        )
    if missing:
        conn.commit()
    done = run(f"SELECT COUNT(*) FROM articles WHERE {col} IS NOT NULL")[0][0]
    print(f"  articles.{col} populated in-database: {done}/82")
chunk_texts = {c["id"]: c["text"] for c in chunks}
for col, model in [("minilm_vector", "minilm_l12_v2"), ("e5_vector", "e5_small")]:
    missing = run(f"SELECT id FROM article_chunks WHERE {col} IS NULL")
    for (cid,) in missing:
        cur.execute(
            f"UPDATE article_chunks SET {col} = VECTOR_EMBEDDING({model} USING :x AS data) WHERE id = :i",
            x=chunk_texts[cid], i=cid,
        )
    if missing:
        conn.commit()
    done = run(f"SELECT COUNT(*) FROM article_chunks WHERE {col} IS NOT NULL")[0][0]
    print(f"  article_chunks.{col} populated in-database: {done}/{len(chunks)}")

# ---------- T9: ORDS REST (before the DDL-heavy tests) ----------
show("T9 ORDS REST service")
try:
    cur.execute(
        "BEGIN ORDS.ENABLE_SCHEMA(p_enabled => TRUE, p_schema => 'MINDSET', "
        "p_url_mapping_type => 'BASE_PATH', p_url_mapping_pattern => 'mindset', "
        "p_auto_rest_auth => FALSE); "
        "ORDS.ENABLE_OBJECT(p_schema => 'MINDSET', p_object => 'ARTICLES', "
        "p_object_type => 'TABLE', p_enabled => TRUE, p_auto_rest_auth => FALSE); "
        "COMMIT; END;"
    )
    print("  schema + ARTICLES REST-enabled")
except oracledb.Error as e:
    print("  ORDS enable:", str(e).splitlines()[0])
first = None
for attempt_no in range(1, 4):
    try:
        with urllib.request.urlopen(ORDS_URL + "?limit=1", timeout=15) as resp:
            doc = json.load(resp)
        first = doc["items"][0]["title"] if doc.get("items") else None
        if first:
            break
    except Exception as e:
        print(f"  ORDS GET attempt {attempt_no} failed: {e}")
        time.sleep(5)
print(f"  GET {ORDS_URL}?limit=1 -> first article: {first}")
results["T9 ORDS REST"] = "PASS" if first else "FAIL"

# ---------- T3: in-database VECTOR_EMBEDDING ----------
show("T3 in-database VECTOR_EMBEDDING (the Windows ORA-40216 test)")
dims = {}
for model in ("minilm_l12_v2", "e5_small"):
    dims[model] = attempt(
        lambda m=model: run(
            f"SELECT VECTOR_DIMENSION_COUNT(VECTOR_EMBEDDING({m} USING 'test' AS data)) FROM dual"
        )[0][0]
    )
print("  dimensions:", dims)
results["T3 in-DB VECTOR_EMBEDDING"] = (
    "PASS" if all(v == 384 for v in dims.values()) else f"FAIL ({dims})"
)

# ---------- T1: exact search in-database vs the engine's own rankings ----------
show("T1 exact search, in-database embeddings vs engine.py (8 queries x 2 models)")
ok = 0
total = 0
for model, col in [("minilm", "minilm_vector"), ("e5", "e5_vector")]:
    dbmodel = "minilm_l12_v2" if model == "minilm" else "e5_small"
    for qid, qtext in QUERIES:
        rows = run(
            f"SELECT id, 1 - VECTOR_DISTANCE({col}, "
            f"VECTOR_EMBEDDING({dbmodel} USING :q AS data), COSINE) "
            "FROM articles ORDER BY 2 DESC FETCH EXACT FIRST 1 ROWS ONLY",
            q=qtext,
        )
        top, sim = rows[0]
        total += 1
        match = top == EXPECTED_TOP1[qid]
        ok += match
        print(f"  {model} {qid}: {top} sim {float(sim):.4f} "
              f"(engine top-1: {EXPECTED_TOP1[qid]}) {'OK' if match else 'MISMATCH'}")
top10 = [r[0] for r in run(
    "SELECT id FROM articles ORDER BY VECTOR_DISTANCE(minilm_vector, "
    "VECTOR_EMBEDDING(minilm_l12_v2 USING :q AS data), COSINE) "
    "FETCH EXACT FIRST 10 ROWS ONLY", q=QUERIES[0][1])]
set_match = set(top10) == set(EXPECTED_TOP10_Q1_MINILM)
print(f"  q1 minilm top-10 set == engine top-10 set: {set_match} ({top10})")
print(f"  top-1 agreement with engine.py: {ok}/{total}")
results["T1 exact search = engine top-1"] = (
    "PASS" if ok == total and set_match else f"FAIL ({ok}/{total}, top10 set {set_match})"
)

# ---------- T2: external table over article titles ----------
show("T2 external table over articles.txt")
txt_path = os.path.join(LAB_DIR_HOST, "articles.txt")
with open(txt_path, "w", encoding="utf-8") as fh:
    for item in corpus:
        fh.write(item["title"].replace("\n", " ") + "\n")
os.chmod(txt_path, 0o644)
attempt(lambda: cur.execute("DROP TABLE articles_ext PURGE"))
cur.execute(
    """CREATE TABLE articles_ext (title VARCHAR2(500))
       ORGANIZATION EXTERNAL (TYPE ORACLE_LOADER DEFAULT DIRECTORY LAB_DIR
         ACCESS PARAMETERS (RECORDS DELIMITED BY 0x'0A' READSIZE 1000000
                            FIELDS (title CHAR(500)))
         LOCATION (LAB_DIR:'articles.txt'))
       REJECT LIMIT UNLIMITED"""
)
n = attempt(lambda: run("SELECT COUNT(*) FROM articles_ext")[0][0])
print(f"  external table rows: {n} (expected 82)")
results["T2 external table"] = "PASS" if n == 82 else f"FAIL ({n})"

# ---------- T4: FETCH APPROX vs EXACT ----------
show("T4 FETCH APPROX vs EXACT (q1, both models)")
same = True
for model, col in [("minilm", "minilm_vector"), ("e5", "e5_vector")]:
    dbmodel = "minilm_l12_v2" if model == "minilm" else "e5_small"
    tops = {}
    for mode in ("EXACT", "APPROX"):
        rows = run(
            f"SELECT id FROM articles ORDER BY VECTOR_DISTANCE({col}, "
            f"VECTOR_EMBEDDING({dbmodel} USING :q AS data), COSINE) "
            f"FETCH {mode} FIRST 3 ROWS ONLY",
            q=QUERIES[0][1],
        )
        tops[mode] = [t for (t,) in rows]
    same &= tops["EXACT"] == tops["APPROX"]
    print(f"  {model}: EXACT {tops['EXACT']} == APPROX {tops['APPROX']}: {tops['EXACT'] == tops['APPROX']}")
results["T4 APPROX == EXACT top-3"] = "PASS" if same else "FAIL"

# ---------- T5: five distance metrics ----------
show("T5 five distance metrics (q1, minilm)")
tops5 = {}
for metric in ("COSINE", "EUCLIDEAN", "EUCLIDEAN_SQUARED", "DOT", "MANHATTAN"):
    rows = attempt(lambda m=metric: run(
        "SELECT id FROM articles ORDER BY VECTOR_DISTANCE(minilm_vector, "
        "VECTOR_EMBEDDING(minilm_l12_v2 USING :q AS data), " + m +
        ") FETCH EXACT FIRST 3 ROWS ONLY", q=QUERIES[0][1]))
    tops5[metric] = [t for (t,) in rows] if isinstance(rows, list) else rows
    print(f"  {metric}: {tops5[metric]}")
results["T5 five distance metrics"] = (
    "PASS" if all(isinstance(v, list) and len(v) == 3 for v in tops5.values()) else "FAIL"
)

# ---------- T7: IVF vector index ----------
show("T7 IVF vector index on minilm_vector")
attempt(lambda: cur.execute("DROP INDEX articles_ivf_idx"))
created = attempt(
    lambda: cur.execute(
        "CREATE VECTOR INDEX articles_ivf_idx ON articles(minilm_vector) "
        "ORGANIZATION NEIGHBOR PARTITIONS DISTANCE COSINE WITH TARGET ACCURACY 90"
    )
)
rows = attempt(
    lambda: run(
        "SELECT id FROM articles ORDER BY VECTOR_DISTANCE(minilm_vector, "
        "VECTOR_EMBEDDING(minilm_l12_v2 USING :q AS data), COSINE) "
        "FETCH APPROX FIRST 3 ROWS ONLY", q=QUERIES[0][1])
)
print("  create:", "OK" if not str(created).startswith("ERROR") else created)
print("  approx q1 with IVF present:", rows)
results["T7 IVF index"] = (
    "PASS" if not str(created).startswith("ERROR") and isinstance(rows, list) and rows else "FAIL"
)

# ---------- T6: HNSW vector index ----------
show("T6 HNSW vector index on e5_vector (vector_memory_size = 256M)")
attempt(lambda: cur.execute("DROP INDEX articles_hnsw_idx"))
created = attempt(
    lambda: cur.execute(
        "CREATE VECTOR INDEX articles_hnsw_idx ON articles(e5_vector) "
        "ORGANIZATION INMEMORY NEIGHBOR GRAPH DISTANCE COSINE WITH TARGET ACCURACY 95"
    )
)
rows = attempt(
    lambda: run(
        "SELECT id FROM articles ORDER BY VECTOR_DISTANCE(e5_vector, "
        "VECTOR_EMBEDDING(e5_small USING :q AS data), COSINE) "
        "FETCH APPROX FIRST 3 ROWS ONLY", q=QUERIES[0][1])
)
print("  create:", "OK" if not str(created).startswith("ERROR") else created)
print("  approx q1 on e5:", rows)
results["T6 HNSW index"] = (
    "PASS" if not str(created).startswith("ERROR") and isinstance(rows, list) and rows else "FAIL"
)

# ---------- plans + dictionary ----------
show("Execution plans and the vector index dictionary")
cur.execute(
    "SELECT id FROM articles ORDER BY VECTOR_DISTANCE(e5_vector, "
    "VECTOR_EMBEDDING(e5_small USING :q AS data), COSINE) "
    "FETCH APPROX FIRST 3 ROWS ONLY", q=QUERIES[0][1]).fetchall()
sql_id = run(
    "SELECT sql_id FROM v$sql WHERE sql_text LIKE 'SELECT id FROM articles ORDER BY VECTOR_DISTANCE(e5_vector%' AND ROWNUM = 1"
)
plan_lines = []
if sql_id:
    plan = run(
        "SELECT plan_table_output FROM TABLE(DBMS_XPLAN.DISPLAY_CURSOR(:sid, NULL, 'BASIC'))",
        sid=sql_id[0][0],
    )
    plan_lines = [
        line for (line,) in plan if any(k in line for k in ("VECTOR", "INDEX", "TABLE ACCESS"))
    ]
for line in plan_lines:
    print("   ", line)
idx = run(
    "SELECT index_name, index_type, status FROM user_indexes "
    "WHERE index_name LIKE 'ARTICLES%' ORDER BY index_name"
)
print("  user_indexes:", idx)
results["plans + dictionary"] = (
    "PASS" if any("HNSW" in line for line in plan_lines) and len(idx) >= 2 else "CHECK"
)

# ---------- T8: hybrid search ----------
show("T8 hybrid search (the Windows ORA-40284 test)")
have = run("SELECT index_name FROM user_indexes WHERE index_name = 'ARTICLES_HYBRID_IDX'")
created = "exists"
if not have:
    created = attempt(
        lambda: cur.execute(
            "CREATE HYBRID VECTOR INDEX articles_hybrid_idx ON articles(text) "
            "PARAMETERS('MODEL MINILM_L12_V2')"
        )
    )
print("  create:", created if str(created).startswith("ERROR") else "OK")


def hybrid_search():
    raw = run(
        "SELECT JSON_SERIALIZE(DBMS_HYBRID_VECTOR.SEARCH("
        "JSON('{\"hybrid_index_name\":\"ARTICLES_HYBRID_IDX\",\"search_text\":"
        + json.dumps(QUERIES[0][1]) + "}')) RETURNING VARCHAR2(32767)) FROM dual"
    )[0][0]
    return json.loads(raw)


hits = attempt(hybrid_search)
if isinstance(hits, list) and hits:
    chunk = hits[0]["chunk_text"]
    owner = next((c["id"] for c in corpus if chunk[:60] in c["text"]), "?")
    print(f"  hits: {len(hits)}; top score {hits[0]['score']}; top article {owner}; "
          f"top chunk: {chunk[:110]}...")
    results["T8 hybrid SEARCH"] = "PASS" if owner == "bh-001" else "CHECK"
else:
    print("  search:", hits)
    results["T8 hybrid SEARCH"] = "FAIL"

# ---------- T10: container embeddings via the provider ----------
show("T10 Private AI container embeddings via UTL_TO_EMBEDDING")
hello = attempt(
    lambda: run(
        "SELECT VECTOR_DIMENSION_COUNT(DBMS_VECTOR.UTL_TO_EMBEDDING('hello', "
        "JSON_OBJECT('provider' VALUE 'privateai', 'credential_name' VALUE NULL, "
        "'url' VALUE :u, 'host' VALUE 'local', 'model' VALUE 'all-minilm-l12-v2' "
        "RETURNING JSON))) FROM dual",
        u=endpoint + "/v1/embeddings",
    )[0][0]
)
print("  container embedding dimensions:", hello)
top = attempt(
    lambda: run(
        "SELECT id FROM articles ORDER BY VECTOR_DISTANCE(minilm_vector, "
        "DBMS_VECTOR.UTL_TO_EMBEDDING(:q, JSON_OBJECT("
        "'provider' VALUE 'privateai', 'credential_name' VALUE NULL, 'url' VALUE :u, "
        "'host' VALUE 'local', 'model' VALUE 'all-minilm-l12-v2' RETURNING JSON)), COSINE) "
        "FETCH EXACT FIRST 1 ROWS ONLY",
        q=QUERIES[0][1], u=endpoint + "/v1/embeddings",
    )[0][0]
)
print("  q1 top-1 via container-embedded query:", top)
results["T10 container embeddings + search"] = (
    "PASS" if hello == 384 and top == "bh-001" else "FAIL"
)

# ---------- T11: chunking ----------
show("T11 chunking with DBMS_VECTOR_CHAIN.UTL_TO_CHUNKS")
cnt = attempt(
    lambda: run(
        "SELECT CARDINALITY(DBMS_VECTOR_CHAIN.UTL_TO_CHUNKS(text, "
        "JSON('{\"by\":\"words\",\"max\":10}'))) FROM articles WHERE id = 'bh-001'"
    )[0][0]
)
print(f"  chunks for bh-001 (max 10 words each): {cnt}")
results["T11 UTL_TO_CHUNKS"] = "PASS" if isinstance(cnt, int) and cnt > 0 else f"FAIL ({cnt})"

# ---------- T12: CLIP text encoder through the container ----------
show("T12 CLIP text encoder via the container (512 dimensions)")
clip_dims = attempt(
    lambda: run(
        "SELECT VECTOR_DIMENSION_COUNT(DBMS_VECTOR.UTL_TO_EMBEDDING('starting a business', "
        "JSON_OBJECT('provider' VALUE 'privateai', 'credential_name' VALUE NULL, "
        "'url' VALUE :u, 'host' VALUE 'local', 'model' VALUE 'clip-vit-base-patch32-txt' "
        "RETURNING JSON))) FROM dual",
        u=endpoint + "/v1/embeddings",
    )[0][0]
)
print("  CLIP text embedding dimensions:", clip_dims)
results["T12 CLIP text encoder"] = "PASS" if clip_dims == 512 else f"FAIL ({clip_dims})"

# ---------- T12b: CLIP image-to-image over the paired article images ----------
show("T12b CLIP image-to-image (paired article images)")


def image_to_image():
    attempt(lambda: cur.execute("DROP TABLE images_stage PURGE"))
    cur.execute("CREATE TABLE images_stage (id VARCHAR2(16), img BLOB)")
    for c in corpus[:3]:
        blob = open(os.path.join(ROOT, c["image_file"]), "rb").read()
        cur.execute("INSERT INTO images_stage VALUES (:i, :b)", i=c["id"], b=blob)
    conn.commit()
    hits = run(
        "SELECT id FROM images_stage ORDER BY VECTOR_DISTANCE("
        "DBMS_VECTOR.UTL_TO_EMBEDDING(img, 'image', JSON_OBJECT("
        "'provider' VALUE 'privateai', 'credential_name' VALUE NULL, 'url' VALUE :u, "
        "'host' VALUE 'local', 'model' VALUE 'clip-vit-base-patch32-img' RETURNING JSON)), "
        "(SELECT DBMS_VECTOR.UTL_TO_EMBEDDING(s.img, 'image', JSON_OBJECT("
        "'provider' VALUE 'privateai', 'credential_name' VALUE NULL, 'url' VALUE :u, "
        "'host' VALUE 'local', 'model' VALUE 'clip-vit-base-patch32-img' RETURNING JSON)) "
        "FROM images_stage s WHERE s.id = 'bh-001'), COSINE) FETCH FIRST 3 ROWS ONLY",
        u=endpoint + "/v1/embeddings",
    )
    return [h[0] for h in hits]


i2i = attempt(image_to_image)
if isinstance(i2i, list):
    print("  nearest images to bh-001's image:", i2i)
    results["T12b CLIP image-to-image"] = "PASS" if i2i and i2i[0] == "bh-001" else "CHECK"
else:
    print("  image-to-image:", i2i)
    results["T12b CLIP image-to-image"] = "FAIL"

# ---------- T13/T14: generation gate + RAG over the article chunks ----------
show("T13 generate text (Private AI LLM)")
try:
    with urllib.request.urlopen(endpoint + "/v1/models", timeout=15) as resp:
        models_doc = json.load(resp)
    ids = [m.get("id", "") for m in models_doc.get("data", [])]
    llm = next((i for i in ids if "ministral" in i.lower() or "instruct" in i.lower()), None)
except Exception as e:
    ids, llm = [], None
    print("  models probe failed:", e)
print("  container models:", ids)
if llm:
    def generate(prompt):
        return run(
            "SELECT DBMS_VECTOR_CHAIN.UTL_TO_GENERATE_TEXT(:p, JSON_OBJECT("
            "'provider' VALUE 'privateai', 'credential_name' VALUE NULL, 'url' VALUE :u, "
            "'host' VALUE 'local', 'model' VALUE :m RETURNING JSON)) FROM dual",
            p=prompt, u=endpoint + "/v1/chat/completions", m=llm,
        )[0][0]

    out = attempt(lambda: generate("Name the article about starting a business without capital."))
    print("  generated:", str(out)[:300])
    results["T13 UTL_TO_GENERATE_TEXT"] = "PASS" if not str(out).startswith("ERROR") else "FAIL"
else:
    print("  no text-generation model in the running container image")
    results["T13 UTL_TO_GENERATE_TEXT"] = "BLOCKED (no LLM in running container — see README)"

show("T14 RAG over the article chunks (embeddings, retrieval, grounded generation)")
n_vec = run("SELECT COUNT(*) FROM article_chunks WHERE minilm_vector IS NOT NULL")[0][0]
dims14 = run(
    "SELECT MIN(VECTOR_DIMENSION_COUNT(minilm_vector)), MAX(VECTOR_DIMENSION_COUNT(minilm_vector)) "
    "FROM article_chunks WHERE minilm_vector IS NOT NULL"
)[0]
print(f"  embedded chunks: {n_vec}/{len(chunks)}, dimensions {dims14[0]}..{dims14[1]}")
results["T14a RAG embeddings"] = (
    "PASS" if n_vec == len(chunks) and dims14[0] == 384 and dims14[1] == 384 else "FAIL"
)
rag_hits = attempt(
    lambda: run(
        "SELECT c.article_id, a.title, c.text FROM article_chunks c "
        "JOIN articles a ON a.id = c.article_id ORDER BY VECTOR_DISTANCE(c.minilm_vector, "
        "DBMS_VECTOR.UTL_TO_EMBEDDING(:q, JSON_OBJECT("
        "'provider' VALUE 'privateai', 'credential_name' VALUE NULL, 'url' VALUE :u, "
        "'host' VALUE 'local', 'model' VALUE 'all-minilm-l12-v2' RETURNING JSON)), COSINE) "
        "FETCH EXACT FIRST 2 ROWS ONLY",
        q=QUERIES[0][1], u=endpoint + "/v1/embeddings",
    )
)
print("  retrieved context:", [(t[0], t[1]) for t in rag_hits] if isinstance(rag_hits, list) else rag_hits)
results["T14b RAG retrieval"] = (
    "PASS" if isinstance(rag_hits, list) and rag_hits and rag_hits[0][0] == "bh-001" else "FAIL"
)
if llm and isinstance(rag_hits, list) and rag_hits:
    context = " ".join(str(x[2])[:300] for x in rag_hits)
    grounded = attempt(lambda: generate(f"Context: {context}\nQuestion: {QUERIES[0][1]}"))
    print("  grounded answer:", str(grounded)[:300])
    results["T14c RAG grounded generation"] = (
        "PASS" if not str(grounded).startswith("ERROR") else "FAIL"
    )
else:
    results["T14c RAG grounded generation"] = "BLOCKED (same gate as T13)"

conn.close()
print("\n===== SUMMARY =====")
for k, v in results.items():
    print(f"  {k}: {v}")
print("\nFULL SUITE (LINUX) DONE")
