"""Command-line entrypoint for the offline ingestion pipeline.

Three modes:

  # direct (offline) — you already have the file and its subject facts
  python -m ingestion <lesson.pdf> --lecture-id <uuid> --subject-id <uuid> --lesson-title "…"
      [--lesson-order N] [--grade g8] [--subject-is-math]

  # resolve — turn a lesson_id into source + subject via the backend, then run
  python -m ingestion --resolve <lesson_id> [--fixtures manifest.yaml]

  # backfill — every lesson pending ingestion, resolved and run in turn
  python -m ingestion --backfill [--fixtures manifest.yaml] [--subject-id <uuid>]

Resolve/backfill use the real backend when ``BACKEND_API_BASE_URL`` is set, else a
local fixtures manifest. With no DATABASE_URL / QDRANT_URL the run uses dry-run stores.
"""

from __future__ import annotations

import argparse
import logging

from shared.logging import setup_logging
from shared.source import materialise

from . import get_backend_client, get_config, new_state, resolve_lesson, run_ingestion
from .backend import BackendClient, LessonSource
from .context import PipelineContext
from .lesson import add_lesson_arguments, apply_lesson_overrides, lesson_state
from .lesson import report as _report
from .read_ahead import ReadAheadUnavailable, read_ahead

logger = logging.getLogger(__name__)


def _ingest(state, config, *, ctx: PipelineContext | None = None,
            read_ahead: bool = True) -> dict:
    """The whole pipeline, with the book's pages read ahead when that can help."""
    ctx = ctx or PipelineContext.build(config)
    if read_ahead and config.get("ingestion.ocr.read_ahead.enabled") \
            and not state.get("source_text"):
        _read_ahead_whole_book(state, ctx, config)
    return run_ingestion(state, ctx=ctx, config=config)


def _read_ahead_whole_book(state, ctx: PipelineContext, config) -> None:
    """Every page into the cache, so the extract stage finds them read.

    A remote source is fetched for this and again by the extract stage: one more
    download of a file that takes seconds, against renders that take minutes.
    Nothing read ahead is wasted if this fails part way — the cache keeps it and
    the pipeline reads what is missing.
    """
    try:
        with materialise(state["source_uri"], config) as path:
            report = read_ahead(path, state, ctx)
    except ReadAheadUnavailable as reason:
        logger.warning("pages not read ahead; the pipeline reads them itself: %s", reason,
                       extra={"lecture_id": state.get("lecture_id")})
        return
    logger.info("pages read ahead", extra={"lecture_id": state.get("lecture_id"), **report})


def _run_direct(args, config) -> int:
    state = lesson_state(args, config)
    _report(args.lecture_id, _ingest(state, config, read_ahead=args.read_ahead))
    return 0


def _run_one_resolved(
    client: BackendClient, lesson_id: str, config, ctx: PipelineContext,
    *, read_ahead: bool = True,
) -> None:
    src: LessonSource = resolve_lesson(client, lesson_id)
    state = new_state(
        lecture_id=src.lesson_id,
        subject_id=src.subject_id,
        source_uri=src.source_uri,
        source_text=src.text_content,
        grade=src.grade,
        subject_name=src.subject_name,
        lesson_title=src.lesson_title,
        lesson_order=src.lesson_order,
        language_hint=src.language,
        subject_is_math=src.subject_is_math,
    )
    _report(lesson_id, _ingest(state, config, ctx=ctx, read_ahead=read_ahead))


def _run_resolve(args, config) -> int:
    client = get_backend_client(config, fixtures_path=args.fixtures)
    ctx = PipelineContext.build(config)
    _run_one_resolved(client, args.resolve, config, ctx, read_ahead=args.read_ahead)
    return 0


def _run_backfill(args, config) -> int:
    client = get_backend_client(config, fixtures_path=args.fixtures)
    pending = client.list_pending_lesson_ids(subject_id=args.subject_id)
    if not pending:
        print("no lessons pending ingestion")
        return 0
    print(f"backfilling {len(pending)} lesson(s)")
    # Build the context (and its embedder) ONCE and reuse it across every lesson —
    # rebuilding per lesson reloads the embedding model each time.
    ctx = PipelineContext.build(config)
    for lesson_id in pending:
        _run_one_resolved(client, lesson_id, config, ctx, read_ahead=args.read_ahead)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ingestion",
        description="Ingest lesson files into chunks, topics and the vector index.",
    )
    parser.add_argument("source", nargs="?", help="direct mode: local path / file:// / s3:// URI")
    add_lesson_arguments(parser)

    parser.add_argument("--resolve", metavar="LESSON_ID", help="resolve one lesson id and run")
    parser.add_argument("--backfill", action="store_true", help="run every pending lesson")
    parser.add_argument("--fixtures", default=None, help="fixtures manifest for offline resolve")
    parser.add_argument("--no-read-ahead", dest="read_ahead", action="store_false",
                        help="let the pipeline read each page itself, one pass at a time")

    args = parser.parse_args(argv)
    config = get_config()
    apply_lesson_overrides(args, config)
    setup_logging(config)

    if args.backfill:
        return _run_backfill(args, config)
    if args.resolve:
        return _run_resolve(args, config)
    if args.source:
        return _run_direct(args, config)

    parser.error(
        "choose one: <source> (direct), "
        "--resolve <lesson_id>, or --backfill"
    )
    return 2  # unreachable; parser.error exits


if __name__ == "__main__":
    raise SystemExit(main())
