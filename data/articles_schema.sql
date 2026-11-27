-- Business Heads Mindset knowledge layer.
-- ARTICLES holds the 82 articles 100%; ARTICLE_CHUNKS holds every
-- 40-word chunk of every article; ARTICLE_IMAGES holds the paired
-- image metadata (the image files live in images/).

CREATE TABLE articles (
  id VARCHAR2(16) PRIMARY KEY,
  title VARCHAR2(500),
  text CLOB,
  article_file VARCHAR2(300),
  image_file VARCHAR2(300),
  words NUMBER,
  chars NUMBER,
  sha256 VARCHAR2(64),
  e5_vector VECTOR(384, FLOAT32),
  minilm_vector VECTOR(384, FLOAT32)
);

CREATE TABLE article_chunks (
  id NUMBER PRIMARY KEY,
  article_id VARCHAR2(16),
  chunk_index NUMBER,
  word_start NUMBER,
  word_end NUMBER,
  text CLOB,
  e5_vector VECTOR(384, FLOAT32),
  minilm_vector VECTOR(384, FLOAT32)
);

CREATE TABLE article_images (
  id VARCHAR2(16) PRIMARY KEY,
  article_id VARCHAR2(16),
  image_file VARCHAR2(300),
  width NUMBER,
  height NUMBER,
  sha256 VARCHAR2(64)
);

CREATE VECTOR INDEX articles_e5_hnsw ON articles(e5_vector)
  ORGANIZATION INMEMORY NEIGHBOR GRAPH DISTANCE COSINE;

CREATE VECTOR INDEX articles_minilm_ivf ON articles(minilm_vector)
  ORGANIZATION NEIGHBOR PARTITIONS DISTANCE COSINE;

CREATE HYBRID VECTOR INDEX articles_hybrid_idx ON articles(text)
  PARAMETERS ('model minilm_l12_v2');
