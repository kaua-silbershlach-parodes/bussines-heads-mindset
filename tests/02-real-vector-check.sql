-- 02-real-vector-check.sql - Verification for the real embedding test
-- (tests/run_precision_test.py). Run as the mindset user:
--   sqlplus mindset@localhost:1521/FREEPDB1 @02-real-vector-check.sql
-- Expected, in order: 82 articles, 82 with an e5 vector, 82 with a
-- MiniLM vector, 384 dimensions, 771 chunks, 771 with both vectors —
-- the counts of this collection and the two models, with nothing
-- missing. (COUNT(<vector column>) itself is not supported —
-- ORA-22849 — so the populated counts are filtered COUNT(*).)
SET PAGESIZE 50 LINESIZE 200 FEEDBACK OFF
SELECT COUNT(*) AS articles FROM articles;
SELECT COUNT(*) AS with_e5_vector FROM articles WHERE e5_vector IS NOT NULL;
SELECT COUNT(*) AS with_minilm_vector FROM articles WHERE minilm_vector IS NOT NULL;
SELECT VECTOR_DIMENSION_COUNT(e5_vector) AS dims FROM articles WHERE e5_vector IS NOT NULL AND ROWNUM = 1;
SELECT COUNT(*) AS chunks FROM article_chunks;
SELECT COUNT(*) AS chunks_with_both_vectors FROM article_chunks WHERE e5_vector IS NOT NULL AND minilm_vector IS NOT NULL;
