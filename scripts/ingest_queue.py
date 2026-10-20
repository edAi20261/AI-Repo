"""Ingest a list of books, one after another, each on this one machine.

For a host with no batch jobs to fan out across, such as Railway: every book
goes through the same four steps the Cloud Run jobs run, as the only share of
each, in a folder of its own that is removed once the book is done.

    BOOKS   one book per line: SOURCE|SUBJECT_ID|ministry or external

A book that fails is reported and the next one starts; the exit status says
whether any failed.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable, Sequence

logger = logging.getLogger("ingest_queue")

STEPS = ("read", "prepare", "embed", "finish")
KINDS = frozenset({"ministry", "external"})


@dataclass(frozen=True)
class Book:
    source: str
    subject_id: str
    kind: str


def parse_books(text: str) -> list[Book]:
    books = []
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"line {number} is not SOURCE|SUBJECT_ID|KIND: {line!r}")
        if parts[2] not in KINDS:
            raise ValueError(f"line {number}: the kind is ministry or external, not {parts[2]!r}")
        books.append(Book(*parts))
    if not books:
        raise ValueError("BOOKS lists no books")
    return books


def _run_step(step: str, env: dict[str, str]) -> int:
    return subprocess.run(["bash", "scripts/cloud_run_step.sh", step], env=env).returncode


def ingest(book: Book, *, run_step: Callable[[str, dict[str, str]], int] = _run_step) -> bool:
    run_dir = tempfile.mkdtemp(prefix="run-")
    env = {**os.environ, "SOURCE": book.source, "SUBJECT_ID": book.subject_id,
           "BOOK_KIND": book.kind, "RUN_DIR": run_dir,
           "CLOUD_RUN_TASK_INDEX": "0", "CLOUD_RUN_TASK_COUNT": "1"}
    try:
        for step in STEPS:
            logger.info("%s: %s", step, book.source)
            if run_step(step, env) != 0:
                logger.error("%s failed: %s", step, book.source)
                return False
        return True
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def main(books: Sequence[Book], *, run_step: Callable[[str, dict[str, str]], int] = _run_step) -> int:
    failed = [book.source for book in books if not ingest(book, run_step=run_step)]
    logger.info("%d of %d books ingested", len(books) - len(failed), len(books))
    for source in failed:
        logger.error("not ingested: %s", source)
    return 1 if failed else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main(parse_books(os.environ.get("BOOKS", ""))))
