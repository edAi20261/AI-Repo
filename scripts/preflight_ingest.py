"""Prove this machine can finish a book before it starts one.

A book is hours of paid model calls, and every one of them is wasted if the
thing that writes the result is unreachable. The failures this catches are all
ones that surface late and look like something else: a database the laptop's
network cannot see, a Qdrant key that reads but cannot write, a service account
that was never granted Vertex, an S3 bucket that accepts no figure crops.

Checks what a run writes, not only what it reads. A credential that can read is
the one that wastes the whole book.
"""

from __future__ import annotations

import sys
import uuid

from shared.config import get_config
from shared.logging import setup_logging

PROBE = f"preflight/{uuid.uuid4()}"


def _database(config) -> str:
    from shared.postgres import connect_once

    with connect_once(config) as conn:
        with conn.cursor() as cur:
            cur.execute("select current_user, current_database()")
            who, where = cur.fetchone()
            # Writing is the half that matters: a read-only grant looks healthy
            # on a select and loses the book at the end. Rolled back rather than
            # deleted, so a check that fails halfway cannot leave a row behind —
            # this runs against the live database.
            cur.execute(
                "insert into ai.page_transcriptions"
                " (lecture_id, page, model, prompt_version, raw_output)"
                " values (%s::uuid, 0, 'preflight', %s, '{}'::jsonb)",
                (str(uuid.UUID(int=0)), PROBE))
        conn.rollback()
    return f"{who}@{where}, ai.* readable and writable"


def _vectors(config) -> str:
    from ingestion.stores import get_vector_store

    store = get_vector_store(config)
    name = config.get("vector_store.collection")
    count = store.count(name) if hasattr(store, "count") else None
    return f"{name}" + (f", {count} points" if count is not None else "")


def _vision(config) -> str:
    provider = config.get("vision.provider")
    if provider != "vertex":
        from shared.vision import _api_key

        return f"{provider}, key present" if _api_key(config) else f"{provider}, NO KEY"
    from shared.gcp import access_token, credential_source, project_of

    source = credential_source(config)
    if not source:
        raise RuntimeError(
            f"{config.get('vision.credentials_env')} is empty; Vertex needs a service account")
    token = access_token(source)
    return f"vertex, project {project_of(source)}, token {len(token)} chars"


def _storage(config) -> str:
    from shared.storage import get_uploader

    uploader = get_uploader(config)
    # Inside a folder of its own: removing it is a delete by prefix, which S3
    # does for any prefix and a local disk only for a directory.
    uri = uploader.upload(b"preflight", f"{PROBE}/probe.txt", content_type="text/plain")
    # Removing it is part of the check: a run that can write crops but not clear
    # the previous ones logs a failure on every book.
    uploader.delete_prefix(PROBE)
    return f"{type(uploader).__name__}, wrote and removed {uri}"


def _embedder(config) -> str:
    from shared.embeddings import get_embedder

    embedder = get_embedder(config)
    vector = embedder.embed(["preflight"])[0]
    return f"{config.get('embeddings.model')}, {len(vector)} dimensions"


CHECKS = (
    ("database", _database),
    ("vectors", _vectors),
    ("vision", _vision),
    ("storage", _storage),
    ("embedder", _embedder),
)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m scripts.preflight_ingest")
    parser.add_argument("--skip", action="append", default=[],
                        choices=[name for name, _ in CHECKS],
                        help="a check this machine does not need, e.g. the embedder "
                             "on a machine that only reads pages")
    args = parser.parse_args(argv)
    setup_logging()
    config = get_config()
    failures = 0
    for name, check in CHECKS:
        if name in args.skip:
            print(f"  skip  {name}", flush=True)
            continue
        try:
            print(f"  ok    {name:9} {check(config)}", flush=True)
        except Exception as error:      # noqa: BLE001 — every failure is reportable
            failures += 1
            print(f"  FAIL  {name:9} {type(error).__name__}: {error}", flush=True)
    if failures:
        print(f"\n{failures} check(s) failed. Fix these before starting a book: a run that "
              f"cannot write is hours of model calls thrown away.", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
