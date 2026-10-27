"""A lesson's own pages of its book, as the PDF the lesson page shows.

The lesson page shows a lesson's PDF attachment, and a lesson split out of a
book had none: a student opened it to an empty page. Ingestion cuts each shown
lesson's pages out of the book while it has the book (``prepare``), stores them,
and attaches them before the lesson is published (``finish``);
``scripts.attach_lesson_pages`` does the same for lessons ingested before, with
these same functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlparse

from shared.config import Config

CONTENT_TYPE = "application/pdf"
# content.media_assets.uploaded_by is a required user. A lesson cut from a book
# has no person who uploaded it, so the platform names the account it is stored as.
UPLOADER = "LESSON_PAGES_UPLOADER"


class PagesNotAttached(RuntimeError):
    pass


@dataclass(frozen=True)
class StoredPages:
    bucket: str
    key: str
    file_name: str
    size_bytes: int
    first_page: int
    last_page: int


def file_name(title: str, lesson_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in " -_" else " " for ch in title)
    return f"{' '.join(safe.split())[:120] or lesson_id}.pdf"


def storage_key(subject_id: str, lesson_id: str) -> str:
    return f"lessons/{subject_id}/{lesson_id}.pdf"


def cut(book: Any, first_page: int, last_page: int) -> bytes:
    import fitz

    if not 1 <= first_page <= last_page <= book.page_count:
        raise PagesNotAttached(f"pages {first_page}-{last_page} are not all in a book of "
                               f"{book.page_count} pages")
    part = fitz.open()
    try:
        part.insert_pdf(book, from_page=first_page - 1, to_page=last_page - 1)
        return part.tobytes(garbage=3, deflate=True)
    finally:
        part.close()


def store(uploader: Any, book: Any, *, subject_id: str, lesson_id: str, title: str,
          first_page: int, last_page: int) -> StoredPages:
    data = cut(book, first_page, last_page)
    uri = uploader.upload(data, storage_key(subject_id, lesson_id), content_type=CONTENT_TYPE)
    location = urlparse(uri)
    if location.scheme != "s3":
        raise PagesNotAttached(f"the pages were stored at {uri}, where the app cannot open "
                               "them; the app serves a lesson's PDF from S3 (set S3_BUCKET)")
    return StoredPages(bucket=location.netloc, key=location.path.lstrip("/"),
                       file_name=file_name(title, lesson_id), size_bytes=len(data),
                       first_page=first_page, last_page=last_page)


def uploader_id(config: Config) -> str:
    user = config.secret(UPLOADER)
    if not user:
        raise PagesNotAttached(f"{UPLOADER} names no user to store a lesson's pages as")
    return user


# A lesson already attached is left alone: a re-ingest overwrites the same
# object, so the attachment it has still shows the lesson's pages.
_ATTACH = """
WITH media AS (
    INSERT INTO content.media_assets
        (uploaded_by, s3_bucket, s3_key, file_name, mime_type, size_bytes)
    SELECT %(uploader)s, %(bucket)s, %(key)s, %(file_name)s, %(mime_type)s, %(size)s
    WHERE NOT EXISTS (SELECT 1 FROM content.lesson_attachments WHERE lesson_id = %(lesson)s)
    RETURNING id
)
INSERT INTO content.lesson_attachments (lesson_id, media_id, description, order_index)
SELECT %(lesson)s, id, 'Lesson PDF', 0 FROM media
"""


def record(cur: Any, pages: Mapping[str, Any], *, lesson_id: str, uploader: str) -> None:
    cur.execute(_ATTACH, {"uploader": uploader, "bucket": pages["bucket"],
                          "key": pages["key"], "file_name": pages["file_name"],
                          "mime_type": CONTENT_TYPE, "size": pages["size_bytes"],
                          "lesson": lesson_id})
