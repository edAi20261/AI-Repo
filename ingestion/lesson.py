"""One lesson, as every ingestion command describes and starts it.

Shared by ``python -m ingestion`` and by each machine of a run split across
several (``ingestion.distributed``). The language a page is read in and the
subject a chunk is labelled with are decided here; two commands that each
decided them would read one book two ways.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import unicodedata
import uuid
from pathlib import Path

from . import get_backend_client
from .state import new_state
from .utils.grades import to_platform_grade

logger = logging.getLogger(__name__)

# Fixed for good. Every generated id is derived from it, so changing it gives
# every book already indexed a new identity, and the next upload of one lands
# beside its old copy instead of replacing it.
ID_NAMESPACE = uuid.UUID("5b1f6c0e-2a7d-5e43-9c1b-7d0a4e8f2c61")
# What fetch_book leaves beside a downloaded book: the name it had where it came
# from, since the download itself is saved under a fixed name.
NAME_SUFFIX = ".name"
UNTITLED = "Untitled book"


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


def fill_in_identity(args) -> None:
    """Give a book the ids and title nobody typed, the same way on every machine.

    The ids used to come from the platform backend, which created the lesson
    first. Without it, a person uploading a book has nothing to type, so:

      - the lecture id is derived from the book's own bytes: the same file is
        always the same lecture, so uploading it again replaces it rather than
        indexing a second copy;
      - the subject id from the subject's name, so books named under one
        subject share one topic catalogue — or from the title when no subject
        is named, which makes the book its own subject;
      - the title from the PDF, else the name the file had when it was fetched.

    Derived, never random: each machine of a split run works this out for
    itself, and a random id would give every machine a different book.
    """
    if not args.lesson_title:
        args.lesson_title = _title_of(args.source)
    if not args.lecture_id:
        args.lecture_id = str(uuid.uuid5(ID_NAMESPACE, "book:" + _fingerprint(args.source)))
    if not args.subject_id:
        named = args.subject_name or args.lesson_title
        args.subject_id = str(uuid.uuid5(ID_NAMESPACE, "subject:" + _key(named)))


def _fingerprint(source: str) -> str:
    path = Path(source[len("file://"):] if source.startswith("file://") else source)
    if not path.is_file():
        # A remote source is fetched later; its address is what is known now.
        return "uri:" + source
    digest = hashlib.sha256()
    with open(path, "rb") as book:
        for block in iter(lambda: book.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _title_of(source: str) -> str:
    path = Path(source)
    if path.is_file():
        try:
            from .utils import pdf

            doc = pdf.open_document(str(path))
            try:
                title = _clean((doc.metadata or {}).get("title") or "")
            finally:
                doc.close()
            if title:
                return title
        except Exception:      # noqa: BLE001 — a title is a nicety, not a reason to fail
            logger.warning("could not read the PDF's own title", exc_info=True)
        sidecar = Path(str(path) + NAME_SUFFIX)
        if sidecar.is_file():
            title = _clean(Path(sidecar.read_text(encoding="utf-8")).stem.replace("_", " "))
            if title:
                return title
        if path.stem.lower() not in ("book", "lesson", "document"):
            return _clean(path.stem.replace("_", " ")) or UNTITLED
    return UNTITLED


def _clean(text: str) -> str:
    """Without the invisible direction marks a right-to-left file name carries."""
    kept = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    return " ".join(kept.split())


def _key(name: str) -> str:
    return unicodedata.normalize("NFKC", _clean(name)).casefold()


def lesson_state(args, config):
    """The state a direct run starts from.

    One function for every way a direct run is started — here, and each machine
    of a run split across several — because the language a page is read in is
    part of this state, and two machines that disagreed about it would read the
    same book two ways.
    """
    fill_in_identity(args)
    logger.info("book identity", extra={"lecture_id": args.lecture_id,
                                        "subject_id": args.subject_id,
                                        "lesson_title": args.lesson_title})
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
    parser.add_argument("--lecture-id",
                        help="content.lessons.id; derived from the book's bytes when left out")
    parser.add_argument("--subject-id",
                        help="the subject; derived from --subject-name (or the title) when "
                             "left out. In backfill mode, a filter")
    parser.add_argument("--grade", default=None, help="grade level of the subject")
    parser.add_argument("--lesson-title", default=None,
                        help="the book's title; from the PDF or its file name when left out")
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
