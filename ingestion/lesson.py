"""One lesson, as every ingestion command describes and starts it.

Shared by ``python -m ingestion`` and by each machine of a run split across
several (``ingestion.distributed``). The language a page is read in and the
subject a chunk is labelled with are decided here; two commands that each
decided them would read one book two ways.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from . import get_backend_client
from .state import new_state
from .utils.grades import to_platform_grade

logger = logging.getLogger(__name__)


def to_uri(source: str) -> str:
    """Accept a local path or a file:///s3:// URI; normalise a bare path."""
    if "://" in source:
        return source
    return Path(source).resolve().as_uri()


def report(lesson_id: str, result: dict) -> None:
    # The stats block is deliberate CLI output the operator reads; warnings go to
    # the logger so they are captured in production the same as everywhere else.
    print(f"# {lesson_id}")
    print(json.dumps(result.get("stats", {}), ensure_ascii=False, indent=2))
    if result.get("ingestion_status"):
        print(f"ingestion_status: {result['ingestion_status']}")
    for err in result.get("errors", []):
        logger.warning(err, extra={"lecture_id": lesson_id})


def _subject_of(config, subject_id: str) -> dict:
    """The subject's name and language, so a direct run labels its chunks the way a
    resolved run does. A citation without them is labelled by whatever the layout
    parser took for a heading."""
    try:
        return get_backend_client(config).get_subject(subject_id) or {}
    except Exception:
        logger.warning("subject facts unavailable; chunks will carry no subject name "
                       "or language", extra={"subject_id": subject_id}, exc_info=True)
        return {}


def lesson_state(args, config):
    """The state a direct run starts from.

    One function for every way a direct run is started — here, and each machine
    of a run split across several — because the language a page is read in is
    part of this state, and two machines that disagreed about it would read the
    same book two ways.
    """
    subject = _subject_of(config, args.subject_id)
    if not args.lesson_title:
        logger.warning("no --lesson-title: citations from this lecture will be labelled "
                       "by section headings only", extra={"lecture_id": args.lecture_id})
    return new_state(
        lecture_id=args.lecture_id,
        subject_id=args.subject_id,
        source_uri=to_uri(args.source),
        grade=to_platform_grade(args.grade),
        subject_name=args.subject_name or subject.get("name"),
        lesson_title=args.lesson_title,
        lesson_order=args.lesson_order,
        namespace=args.namespace,
        version=args.version,
        language_hint=args.language or subject.get("language"),
        subject_is_math=args.subject_is_math,
    )


def add_lesson_arguments(parser: argparse.ArgumentParser) -> None:
    """The arguments that describe one lesson, for every command that takes one."""
    parser.add_argument("--lecture-id", help="direct mode: content.lessons.id (uuid)")
    parser.add_argument("--subject-id", help="direct mode subject id, or backfill filter")
    parser.add_argument("--grade", default=None, help="grade level of the subject")
    parser.add_argument("--lesson-title", default=None, help="direct mode: the lesson's title")
    parser.add_argument("--lesson-order", type=int, default=None,
                        help="direct mode: the lesson's position in its subject")
    parser.add_argument(
        "--subject-is-math",
        action="store_true",
        help="force every page down the vision route (maths subject)",
    )
    parser.add_argument("--namespace", default="syllabus")
    parser.add_argument("--version", type=int, default=1, help="re-ingestion version")
    parser.add_argument("--language", default=None,
                        help="the book's language (ar, en, fr, de) when the backend "
                             "cannot be asked")
    parser.add_argument("--subject-name", default=None,
                        help="the subject's name when the backend cannot be asked")
    parser.add_argument("--layout-model", dest="layout_model", action="store_true", default=None,
                        help="read magazine-style pages with the document layout model")
    parser.add_argument("--no-layout-model", dest="layout_model", action="store_false",
                        help="read every page with the text layer or the vision model")


def apply_lesson_overrides(args, config) -> None:
    if args.layout_model is not None:
        config.override("ingestion.layout.tier2.enabled", args.layout_model)
