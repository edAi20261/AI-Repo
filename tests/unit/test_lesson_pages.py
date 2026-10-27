from __future__ import annotations

import pytest

from shared.namespaces import BACKGROUND, SYLLABUS

BOOK = "0743ec4c-b00d-5e17-b573-9b60b5593a26"
SUBJECT = "8c2f0a10-0001-4a00-9000-00000000000a"
UPLOADER = "11111111-2222-4333-8444-555555555555"


class FakeS3Uploader:
    def __init__(self, bucket: str = "media", *, explode: bool = False) -> None:
        self.bucket = bucket
        self.explode = explode
        self.objects: dict[str, tuple[bytes, str]] = {}

    def upload(self, data: bytes, key: str, content_type: str = "image/png") -> str:
        if self.explode:
            raise RuntimeError("AccessDenied")
        stored = f"ai-artifacts/{key}"
        self.objects[stored] = (data, content_type)
        return f"s3://{self.bucket}/{stored}"


class FakeLocalUploader(FakeS3Uploader):
    def upload(self, data: bytes, key: str, content_type: str = "image/png") -> str:
        return f"file:///tmp/artifacts/{key}"


def _book(path, pages: int) -> str:
    import fitz

    doc = fitz.open()
    for number in range(1, pages + 1):
        doc.new_page().insert_text((72, 72), f"page {number}")
    doc.save(str(path))
    doc.close()
    return str(path)


def _texts(data: bytes) -> list[str]:
    import fitz

    with fitz.open(stream=data, filetype="pdf") as doc:
        return [page.get_text().strip() for page in doc]


def _lessons(*spans: tuple[int, int], sections: tuple[int, ...] = ()) -> list[dict]:
    from ingestion.books import lesson_id_for

    return [{"lesson_id": lesson_id_for(BOOK, n), "ordinal": n, "title": f"Lesson {n}",
             "label": f"Lesson {n}", "unit_number": 1, "unit_title": "", "page_from": first,
             "page_to": last, "section": n in sections}
            for n, (first, last) in enumerate(spans)]


def test_each_ministry_lesson_carries_exactly_its_own_pages_of_the_book(tmp_path):
    """A lesson split out of a book was published with no PDF, and the lesson page
    opened empty for 1,507 of 1,814 ministry lessons."""
    from ingestion.books import store_lesson_pages
    from shared.config import get_config

    uploader = FakeS3Uploader()
    state = {"lecture_id": BOOK, "subject_id": SUBJECT, "namespace": SYLLABUS,
             "lessons": _lessons((1, 3), (4, 6), (7, 8), sections=(2,))}
    store_lesson_pages(state, _book(tmp_path / "book.pdf", 8), get_config(), uploader=uploader)

    first, second, index = state["lessons"]
    assert first["pdf"]["bucket"] == "media"
    assert first["pdf"]["key"] == f"ai-artifacts/lessons/{SUBJECT}/{first['lesson_id']}.pdf"
    assert first["pdf"]["file_name"] == "Lesson 0.pdf"
    data, content_type = uploader.objects[first["pdf"]["key"]]
    assert content_type == "application/pdf"
    assert first["pdf"]["size_bytes"] == len(data)
    assert _texts(data) == ["page 1", "page 2", "page 3"]
    assert _texts(uploader.objects[second["pdf"]["key"]][0]) == ["page 4", "page 5", "page 6"]
    assert "pdf" not in index
    assert len(uploader.objects) == 2


def test_a_background_books_lessons_are_given_no_pages(tmp_path):
    from ingestion.books import store_lesson_pages
    from shared.config import get_config

    uploader = FakeS3Uploader()
    state = {"lecture_id": BOOK, "subject_id": SUBJECT, "namespace": BACKGROUND,
             "lessons": _lessons((1, 4))}
    store_lesson_pages(state, _book(tmp_path / "book.pdf", 4), get_config(), uploader=uploader)
    assert uploader.objects == {}
    assert "pdf" not in state["lessons"][0]


@pytest.mark.parametrize("uploader", [FakeS3Uploader(explode=True), FakeLocalUploader()],
                         ids=["storage refused", "no bucket configured"])
def test_pages_that_could_not_be_stored_are_recorded_against_the_lesson(tmp_path, uploader):
    """A local file:// copy is no attachment the app can open, and a refused
    upload is not a lesson's pages: either must stop the lesson being shown and
    say why."""
    from ingestion.books import store_lesson_pages
    from shared.config import get_config

    state = {"lecture_id": BOOK, "subject_id": SUBJECT, "namespace": SYLLABUS,
             "lessons": _lessons((1, 2))}
    store_lesson_pages(state, _book(tmp_path / "book.pdf", 2), get_config(), uploader=uploader)
    lesson = state["lessons"][0]
    assert "pdf" not in lesson
    assert lesson["pdf_error"]


def test_attaching_records_the_pages_as_the_lessons_pdf(fake_postgres, monkeypatch):
    from ingestion.books import LessonCatalog, lesson_id_for
    from shared.config import get_config

    monkeypatch.setenv("LESSON_PAGES_UPLOADER", UPLOADER)
    db = fake_postgres()
    lesson_id = lesson_id_for(BOOK, 0)
    pdf = {"bucket": "media", "key": "ai-artifacts/lessons/s/l.pdf", "file_name": "One.pdf",
           "size_bytes": 1234, "first_page": 1, "last_page": 3}
    LessonCatalog(get_config()).attach(lesson_id, pdf=pdf, reason=None)
    (sql, args), = [(sql, args) for sql, args in db.statements if "media_assets" in sql]
    assert "content.lesson_attachments" in sql
    assert args == {"uploader": UPLOADER, "bucket": "media", "key": pdf["key"],
                    "file_name": "One.pdf", "mime_type": "application/pdf", "size": 1234,
                    "lesson": lesson_id}
    assert db.committed


def test_a_lesson_whose_pages_were_not_stored_cannot_be_attached(fake_postgres, monkeypatch):
    from ingestion.books import LessonCatalog, lesson_id_for
    from ingestion.lesson_pages import PagesNotAttached
    from shared.config import get_config

    monkeypatch.setenv("LESSON_PAGES_UPLOADER", UPLOADER)
    db = fake_postgres()
    catalog = LessonCatalog(get_config())
    with pytest.raises(PagesNotAttached, match="AccessDenied"):
        catalog.attach(lesson_id_for(BOOK, 0), pdf=None, reason="AccessDenied")
    monkeypatch.delenv("LESSON_PAGES_UPLOADER")
    with pytest.raises(PagesNotAttached, match="LESSON_PAGES_UPLOADER"):
        catalog.attach(lesson_id_for(BOOK, 0), pdf={"bucket": "b", "key": "k",
                                                    "file_name": "f", "size_bytes": 1,
                                                    "first_page": 1, "last_page": 1},
                       reason=None)
    assert db.statements == []


def test_a_ministry_lesson_is_only_published_once_its_pages_are_attached(fake_postgres):
    """Publishing checked status and source only, so every lesson split out of a
    book was published to an empty page; a lesson filed by any other path
    (split_whole_books) is held to the same rule."""
    from ingestion.books import LessonCatalog, lesson_id_for
    from shared.config import get_config

    db = fake_postgres()
    LessonCatalog(get_config()).finished(lesson_id_for(BOOK, 0), status="ready",
                                         errors=[], version=1)
    (sql, _), = [(sql, args) for sql, args in db.statements if "UPDATE content.lessons" in sql]
    published = sql.split("status = CASE WHEN", 1)[1].split("THEN 'published'", 1)[0]
    assert "content.lesson_attachments" in published


class _Catalog:
    def __init__(self, *, refuse: bool = False) -> None:
        self.refuse = refuse
        self.calls: list[tuple] = []

    def attach(self, lesson_id, *, pdf, reason):
        from ingestion.lesson_pages import PagesNotAttached

        self.calls.append(("attach", lesson_id))
        if self.refuse or pdf is None:
            raise PagesNotAttached(reason or "refused")

    def finished(self, lesson_id, *, status, errors, version):
        self.calls.append(("finished", lesson_id, status, tuple(errors)))


def _part(*, shown: bool, pdf: dict | None, reason: str | None = None) -> dict:
    return {"lecture_id": "lesson-1", "shown": shown, "lesson_pdf": pdf,
            "lesson_pdf_error": reason, "version": 1}


def test_a_shown_lesson_is_attached_before_it_is_published(monkeypatch):
    import ingestion.distributed as distributed

    monkeypatch.setattr(distributed, "run_stages", lambda part, ctx, stages: part)
    catalog = _Catalog()
    distributed._finish_lesson(_part(shown=True, pdf={"key": "k"}), None, catalog)
    assert catalog.calls == [("attach", "lesson-1"), ("finished", "lesson-1", "ready", ())]


def test_a_lesson_whose_pages_are_missing_is_held_back_and_says_why(monkeypatch, caplog):
    """A lesson with no pages was published anyway; a student opened it to an
    empty page and nobody was told."""
    import ingestion.distributed as distributed

    monkeypatch.setattr(distributed, "run_stages", lambda part, ctx, stages: part)
    catalog = _Catalog()
    _, _, status, errors = distributed._finish_lesson(
        _part(shown=True, pdf=None, reason="AccessDenied"), None, catalog)
    assert status == "needs_review"
    assert any("AccessDenied" in error for error in errors)
    assert catalog.calls[-1] == ("finished", "lesson-1", "needs_review", tuple(errors))
    assert any(r.levelname == "ERROR" and "AccessDenied" in r.getMessage()
               for r in caplog.records)


def test_a_lesson_no_student_is_shown_needs_no_pages(monkeypatch):
    import ingestion.distributed as distributed

    monkeypatch.setattr(distributed, "run_stages", lambda part, ctx, stages: part)
    catalog = _Catalog(refuse=True)
    _, _, status, _ = distributed._finish_lesson(_part(shown=False, pdf=None), None, catalog)
    assert status == "ready"
    assert catalog.calls == [("finished", "lesson-1", "ready", ())]
