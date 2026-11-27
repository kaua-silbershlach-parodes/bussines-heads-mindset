#!/usr/bin/env python3
"""Structure test: the 82 articles are present 100% in the corpus and
chunks, every image is paired, and the dump/schema artifacts exist."""
import hashlib
import json
import os
import re
import struct
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
checks = []


def check(name, ok, detail=""):
    checks.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip())


def jpeg_size(path):
    with open(path, "rb") as f:
        data = f.read()
    i = 2
    while i < len(data) - 9:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2):
            h, w = struct.unpack(">HH", data[i + 5:i + 9])
            return w, h
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        i += 2 + length
    return None


def main():
    with open(os.path.join(ROOT, "corpus.json"), encoding="utf-8") as f:
        corpus = json.load(f)
    with open(os.path.join(ROOT, "data", "article_chunks.json"), encoding="utf-8") as f:
        chunks = json.load(f)

    article_files = sorted(os.listdir(os.path.join(ROOT, "articles")))
    image_files = sorted(os.listdir(os.path.join(ROOT, "images")))
    check("82 article files", len(article_files) == 82, f"({len(article_files)})")
    check("82 image files", len(image_files) == 82, f"({len(image_files)})")
    check("corpus has 82 articles", len(corpus) == 82, f"({len(corpus)})")
    check("corpus ids bh-001..bh-082", [c["id"] for c in corpus] == [f"bh-{i:03d}" for i in range(1, 83)])

    bodies_ok = True
    images_ok = True
    for c in corpus:
        raw = open(os.path.join(ROOT, c["article_file"]), encoding="utf-8").read()
        body = re.sub(r"^#.*$", "", raw, flags=re.M)
        body = re.sub(r"!\[.*?\]\(.*?\)", "", body)
        body = " ".join(body.split())
        if body != c["text"] or hashlib.sha256(body.encode()).hexdigest() != c["sha256"]:
            bodies_ok = False
        if jpeg_size(os.path.join(ROOT, c["image_file"])) != (3200, 1800):
            images_ok = False
    check("corpus text = article bodies 100%", bodies_ok)
    check("all images paired at 3200x1800", images_ok)

    by_article = {}
    for ch in chunks:
        by_article.setdefault(ch["article_id"], []).append(ch)
    coverage_ok = len(by_article) == 82
    for c in corpus:
        parts = sorted(by_article.get(c["id"], []), key=lambda x: x["chunk_index"])
        if " ".join(p["text"] for p in parts) != c["text"]:
            coverage_ok = False
        if any(len(p["text"].split()) > 40 for p in parts):
            coverage_ok = False
    check("chunks reconstruct every article 100%", coverage_ok, f"({len(chunks)} chunks)")
    check("chunk ids sequential", [ch["id"] for ch in chunks] == list(range(1, len(chunks) + 1)))

    schema = open(os.path.join(ROOT, "data", "articles_schema.sql"), encoding="utf-8").read()
    check("schema defines articles + chunks + images", all(s in schema for s in ("CREATE TABLE articles", "CREATE TABLE article_chunks", "CREATE TABLE article_images")))
    dump = os.path.join(ROOT, "data", "articles.dmp")
    check("data dump present", os.path.isfile(dump) and os.path.getsize(dump) > 0,
          f"({os.path.getsize(dump) if os.path.isfile(dump) else 0} bytes)")

    passed = sum(checks)
    print(f"STRUCTURE TEST: {passed}/{len(checks)}")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
