"""What happened to the books in a manifest, read back from the database.

A run prints its own stats, but a window that was closed takes them with it, and
the operator running a shelf of books is not the person who will be asked
whether they are all in. This reads the same tables the service reads, so it is
the truth rather than a transcript.

    python -m scripts.ingest_status books.tsv

Prints one line per book: how much of it has been read, whether it was indexed
or held back, and how many chunks it produced.
"""

from __future__ import annotations

import sys
from pathlib import Path

from shared.config import get_config
from shared.postgres import connect

CONNECT_TIMEOUT = 15
STATEMENT_TIMEOUT_MS = 30000

_COUNTS = """
SELECT
  (SELECT count(DISTINCT page) FROM ai.page_transcriptions WHERE lecture_id = %s::uuid),
  (SELECT count(*) FROM ai.chunks WHERE lecture_id = %s::uuid),
  (SELECT status FROM ai.lecture_ingestion WHERE lecture_id = %s::uuid)
"""

# What a status means to someone who did not write the pipeline.
PLAIN = {
    "ready": "done, students can use it",
    "needs_review": "HELD BACK, too many pages could not be read",
    None: "not finished",
}


def _books(manifest: Path):
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 4:
            yield parts[1].strip(), parts[3].strip()


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m scripts.ingest_status <books.tsv>", file=sys.stderr)
        return 2
    manifest = Path(sys.argv[1])
    if not manifest.is_file():
        print(f"no such file: {manifest}", file=sys.stderr)
        return 2

    config = get_config()
    with connect(config, connect_timeout=CONNECT_TIMEOUT,
                 statement_timeout_ms=STATEMENT_TIMEOUT_MS) as conn:
        with conn.cursor() as cur:
            unfinished = 0
            for lecture_id, title in _books(manifest):
                cur.execute(_COUNTS, (lecture_id, lecture_id, lecture_id))
                pages, chunks, status = cur.fetchone()
                if status != "ready":
                    unfinished += 1
                print(f"{title}\n"
                      f"    pages read   {pages}\n"
                      f"    chunks       {chunks}\n"
                      f"    status       {PLAIN.get(status, status)}\n")
    if unfinished:
        print(f"{unfinished} book(s) not finished. Running the same manifest again "
              f"continues them; a page already read is not read a second time.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
