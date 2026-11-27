-- 01-connection-test.sql - Smoke test 1: connection and version.
--
-- Run through the listener, as the mindset user:
--   sqlplus mindset@localhost:1521/FREEPDB1 @01-connection-test.sql
--
-- Expected:
--   BANNER_FULL = Oracle AI Database 26ai Free Release 23.26.4.0.0 ...
--   FREEPDB1    = READ WRITE

SELECT banner_full FROM v$version;

SELECT name, open_mode FROM v$pdbs;

EXIT
