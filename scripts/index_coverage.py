"""Which books in the bucket are in the index, and which are not.

    python -m scripts.index_coverage [PREFIX ...]

A book's lessons are filed under an id made from the book's own bytes, and the
index stores that id rather than where the file came from. So a book is looked
up the way ingestion names it: fetched, fingerprinted, and every lesson id it
could have searched for. A file renamed or moved in the bucket is still found;
a file whose bytes changed is reported missing, because it is a different book.

    PREFIX  bucket prefixes to check (default: books/ and ai-artifacts/books/)

Lessons cut from a book (``.../lessons/``) are not books and are skipped.
"""

from __future__ import annotations

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, Sequence

from ingestion.books import lesson_id_for
from ingestion.lesson import book_id_for
from shared.config import get_config
from shared.source import materialise
from shared.storage import _s3_client
from shared.vector_store import query_client

logger = logging.getLogger("index_coverage")

DEFAULT_PREFIXES = ("books/", "ai-artifacts/books/")
MOST_LESSONS = 1000
WORKERS = 6


@dataclass(frozen=True)
class Coverage:
    key: str
    book_id: str
    lessons: int
    chunks: int


def candidate_ids(book_id: str) -> list[str]:
    return [book_id] + [lesson_id_for(book_id, ordinal) for ordinal in range(MOST_LESSONS)]


def book_keys(client, bucket: str, prefixes: Iterable[str]) -> list[str]:
    keys = []
    for prefix in prefixes:
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            keys += [o["Key"] for o in page.get("Contents", [])
                     if o["Key"].lower().endswith(".pdf") and "/lessons/" not in o["Key"]]
    return sorted(set(keys))


def chunk_counts(qdrant, collection: str, ids: Sequence[str]) -> dict[str, int]:
    from qdrant_client import models

    counts: dict[str, int] = {}
    offset = None
    flt = models.Filter(must=[models.FieldCondition(key="lecture_id", match=models.MatchAny(any=list(ids)))])
    while True:
        points, offset = qdrant.scroll(collection, scroll_filter=flt, limit=1000, offset=offset,
                                       with_payload=["lecture_id"], with_vectors=False)
        for point in points:
            lecture = point.payload["lecture_id"]
            counts[lecture] = counts.get(lecture, 0) + 1
        if offset is None:
            return counts


def check(key: str, *, bucket: str, config, qdrant, collection: str) -> Coverage:
    with materialise(f"s3://{bucket}/{key}", config) as path:
        book_id = book_id_for(path)
    counts = chunk_counts(qdrant, collection, candidate_ids(book_id))
    return Coverage(key, book_id, len(counts), sum(counts.values()))


def main(prefixes: Sequence[str]) -> int:
    config = get_config()
    bucket = os.environ["S3_BUCKET"]
    keys = book_keys(_s3_client(os.environ.get("S3_REGION"), None), bucket, prefixes)
    qdrant = query_client(config, timeout=120)
    collection = config.get("vector_store.collection")
    logger.info("%d books under %s", len(keys), ", ".join(prefixes))
    found, missing, failed = [], [], []
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = {pool.submit(check, key, bucket=bucket, config=config, qdrant=qdrant,
                               collection=collection): key for key in keys}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception:
                logger.exception("could not check %s", futures[future])
                failed.append(futures[future])
                continue
            (found if result.lessons else missing).append(result)
            logger.info("%s %d lessons %d chunks %s", "in index" if result.lessons else "MISSING",
                        result.lessons, result.chunks, result.key)
    logger.info("%d in the index, %d missing, %d could not be checked",
                len(found), len(missing), len(failed))
    for result in sorted(missing, key=lambda r: r.key):
        logger.info("missing: s3://%s/%s", bucket, result.key)
    for key in sorted(failed):
        logger.error("unchecked: s3://%s/%s", bucket, key)
    return 1 if failed else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main(sys.argv[1:] or DEFAULT_PREFIXES))
