from __future__ import annotations

import pytest

from ingestion.books import (
    LessonCatalog,
    _validator,
    detect_lessons,
    file_under_lessons,
    lesson_id_for,
    lesson_states,
    lessons_from,
)
from ingestion.layout.blocks import Block, PageMeta, ParsedPage
from ingestion.schemas import LessonBreakdown, LessonItem
from ingestion.state import Chunk, Figure

BOOK = "0743ec4c-b00d-5e17-b573-9b60b5593a26"
SUBJECT = "8c2f0a10-0001-4a00-9000-00000000000a"


def _breakdown(*starts: int) -> LessonBreakdown:
    return LessonBreakdown(lessons=[
        LessonItem(title=f"Lesson {n}", unit_number=1, start_page=s)
        for n, s in enumerate(starts, 1)])


def _chunk(page: int) -> Chunk:
    return Chunk(lecture_id=BOOK, text=f"page {page}", heading_path="", page_from=page,
                 page_to=page, language="en", order_index=page)


def _page(number: int, heading: str, body: str) -> dict:
    return ParsedPage(
        meta=PageMeta(doc_id=BOOK, page=number, width=600, height=800),
        blocks=[Block(text=heading, type="heading", bbox=(0, 0, 1, 1)),
                Block(text=body, type="body", bbox=(0, 2, 1, 3))],
    ).model_dump()


def test_every_page_of_the_book_belongs_to_exactly_one_lesson():
    lessons = lessons_from(_breakdown(5, 20, 40), book_id=BOOK, first_page=1, last_page=60)
    assert [(lsn.page_from, lsn.page_to) for lsn in lessons] == [(1, 19), (20, 39), (40, 60)]


def test_a_lesson_keeps_its_id_when_the_same_book_is_ingested_again():
    first = lessons_from(_breakdown(3, 9), book_id=BOOK, first_page=1, last_page=12)
    again = lessons_from(_breakdown(3, 10), book_id=BOOK, first_page=1, last_page=12)
    assert [lsn.lesson_id for lsn in first] == [lsn.lesson_id for lsn in again]
    other = lessons_from(_breakdown(3, 9), book_id="another-book", first_page=1, last_page=12)
    assert first[0].lesson_id != other[0].lesson_id


def test_chunks_and_figures_are_filed_under_the_lesson_their_page_is_in():
    lessons = lessons_from(_breakdown(1, 10), book_id=BOOK, first_page=1, last_page=20)
    figure = Figure(lecture_id=BOOK, page=15, s3_uri="s3://b/f.png", bbox=(0, 0, 1, 1))
    state = {"lecture_id": BOOK, "book_id": BOOK, "subject_id": SUBJECT,
             "chunks": [_chunk(2), _chunk(12), _chunk(20)], "figures": [figure]}
    file_under_lessons(state, lessons)
    owners = [c.lecture_id for c in state["chunks"]]
    assert owners == [lessons[0].lesson_id, lessons[1].lesson_id, lessons[1].lesson_id]
    assert state["chunks"][1].lesson_title == "Lesson 2"
    assert figure.lecture_id == lessons[1].lesson_id

    parts = lesson_states(state)
    assert [p["lecture_id"] for p in parts] == [lsn.lesson_id for lsn in lessons]
    assert [len(p["chunks"]) for p in parts] == [1, 2]
    assert parts[1]["figures"] == [figure]
    assert parts[1]["subject_id"] == SUBJECT
    assert "lessons" not in parts[0]


def test_a_breakdown_that_does_not_fit_the_book_is_sent_back():
    check = _validator(1, 50)
    check(_breakdown(1, 10, 30))
    for bad in (_breakdown(), _breakdown(10, 5), _breakdown(1, 70), _breakdown(3, 3)):
        with pytest.raises(ValueError):
            check(bad)


class _Model:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def invoke(self, prompt):
        self.prompts.append(str(prompt))

        class _Message:
            content = self.reply

        return _Message()


def test_the_lessons_are_read_from_the_book_by_the_model():
    from shared.config import get_config

    pages = [_page(n, f"Unit 1 Lesson {1 + n // 4}", f"text on page {n}") for n in range(1, 9)]
    state = {"lecture_id": BOOK, "parsed_pages": pages}
    model = _Model('{"lessons": [{"title": "Unit 1 Lesson 1", "unit_number": 1, '
                   '"start_page": 1}, {"title": "Unit 1 Lesson 2", "unit_number": 1, '
                   '"start_page": 4}]}')
    lessons = detect_lessons(state, get_config(), llm=model)
    assert [(lsn.title, lsn.page_from, lsn.page_to) for lsn in lessons] == [
        ("Unit 1 Lesson 1", 1, 3), ("Unit 1 Lesson 2", 4, 8)]
    assert "[4] Unit 1 Lesson 2" in model.prompts[0]


def test_lessons_are_registered_under_the_subject_after_the_lessons_already_there(
        fake_postgres):
    from shared.config import get_config

    db = fake_postgres(fetchone=[(7,)])
    lessons = [{"lesson_id": lesson_id_for(BOOK, n), "ordinal": n, "title": f"Lesson {n}",
                "unit_number": 1} for n in range(2)]
    LessonCatalog(get_config()).register(subject_id=SUBJECT, book_title="Gem",
                                         lessons=lessons)
    inserts = [args for sql, args in db.statements if "INSERT INTO content.lessons" in sql]
    assert [(a[0], a[1], a[3], a[6]) for a in inserts] == [
        (lessons[0]["lesson_id"], SUBJECT, "Lesson 0", 7),
        (lessons[1]["lesson_id"], SUBJECT, "Lesson 1", 8)]
    assert db.committed
