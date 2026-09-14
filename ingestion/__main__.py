"""Command-line entrypoint for the offline ingestion pipeline.

Three modes:

  # direct (offline) — you already have the file and its subject facts
  python -m ingestion <lesson.pdf> --lecture-id <uuid> --subject-id <uuid> [--grade g8] [--subject-is-math]

  # resolve — turn a lesson_id into source + subject via the backend, then run
  python -m ingestion --resolve <lesson_id> [--fixtures manifest.yaml]

  # backfill — every lesson pending ingestion, resolved and run in turn
  python -m ingestion --backfill [--fixtures manifest.yaml] [--subject-id <uuid>]

Resolve/backfill use the real backend when ``BACKEND_API_BASE_URL`` is set, else a
local fixtures manifest. With no DATABASE_URL / QDRANT_URL the run uses dry-run stores.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from shared.logging import setup_logging

from . import get_backend_client, get_config, new_state, resolve_lesson, run_ingestion
from .backend import BackendClient, LessonSource
from .context import PipelineContext

logger = logging.getLogger(__name__)


def _to_uri(source: str) -> str:
    """Accept a local path or a file:///s3:// URI; normalise a bare path."""
    if "://" in source:
        return source
    return Path(source).resolve().as_uri()


def _report(lesson_id: str, result: dict) -> None:
    # The stats block is deliberate CLI output the operator reads; warnings go to
    # the logger so they are captured in production the same as everywhere else.
    print(f"# {lesson_id}")
    print(json.dumps(result.get("stats", {}), ensure_ascii=False, indent=2))
    if result.get("ingestion_status"):
        print(f"ingestion_status: {result['ingestion_status']}")
    for err in result.get("errors", []):
        logger.warning(err, extra={"lecture_id": lesson_id})


def _run_direct(args, config) -> int:
    state = new_state(
        lecture_id=args.lecture_id,
        subject_id=args.subject_id,
        source_uri=_to_uri(args.source),
        grade=args.grade,
        namespace=args.namespace,
        version=args.version,
        subject_is_math=args.subject_is_math,
    )
    _report(args.lecture_id, run_ingestion(state, config=config))
    return 0


def _run_one_resolved(
    client: BackendClient, lesson_id: str, config, ctx: PipelineContext
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
    _report(lesson_id, run_ingestion(state, ctx=ctx, config=config))


def _run_resolve(args, config) -> int:
    client = get_backend_client(config, fixtures_path=args.fixtures)
    ctx = PipelineContext.build(config)
    _run_one_resolved(client, args.resolve, config, ctx)
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
        _run_one_resolved(client, lesson_id, config, ctx)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ingestion",
        description="Ingest lesson files into chunks, topics and the vector index.",
    )
    parser.add_argument("source", nargs="?", help="direct mode: local path / file:// / s3:// URI")
    parser.add_argument("--lecture-id", help="direct mode: content.lessons.id (uuid)")
    parser.add_argument("--subject-id", help="direct mode subject id, or backfill filter")
    parser.add_argument("--grade", default=None, help="grade level of the subject")
    parser.add_argument(
        "--subject-is-math",
        action="store_true",
        help="force every page down the vision route (maths subject)",
    )
    parser.add_argument("--namespace", default="syllabus")
    parser.add_argument("--version", type=int, default=1, help="re-ingestion version")

    parser.add_argument("--resolve", metavar="LESSON_ID", help="resolve one lesson id and run")
    parser.add_argument("--backfill", action="store_true", help="run every pending lesson")
    parser.add_argument("--fixtures", default=None, help="fixtures manifest for offline resolve")

    args = parser.parse_args(argv)
    config = get_config()
    setup_logging(config)

    if args.backfill:
        return _run_backfill(args, config)
    if args.resolve:
        return _run_resolve(args, config)
    if args.source and args.lecture_id and args.subject_id:
        return _run_direct(args, config)

    parser.error(
        "choose one: <source> --lecture-id --subject-id (direct), "
        "--resolve <lesson_id>, or --backfill"
    )
    return 2  # unreachable; parser.error exits


if __name__ == "__main__":
    raise SystemExit(main())
