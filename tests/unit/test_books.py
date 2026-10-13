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
from ingestion.schemas import LessonBreakdown, LessonItem, SectionItem, UnitItem
from ingestion.state import Chunk, Figure

BOOK = "0743ec4c-b00d-5e17-b573-9b60b5593a26"
SUBJECT = "8c2f0a10-0001-4a00-9000-00000000000a"


def _breakdown(*starts: int) -> LessonBreakdown:
    return LessonBreakdown(units=[UnitItem(title="Unit 1", unit_number=1, lessons=[
        LessonItem(title=f"Lesson {n}", start_page=s) for n, s in enumerate(starts, 1)])])


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
    assert state["chunks"][1].lesson_title == "Unit 1 — Lesson 2"
    assert [f.lecture_id for f in state["figures"]] == [lessons[1].lesson_id]

    parts = lesson_states(state)
    assert [p["lecture_id"] for p in parts] == [lsn.lesson_id for lsn in lessons]
    assert [len(p["chunks"]) for p in parts] == [1, 2]
    assert parts[1]["figures"] == state["figures"]
    assert parts[1]["subject_id"] == SUBJECT
    assert "lessons" not in parts[0]


def test_a_breakdown_that_does_not_fit_the_book_is_sent_back():
    check = _validator(1, 50, 3.5)
    check(_breakdown(1, 10, 30))
    check(_breakdown(1, 10, 10))
    for bad in (_breakdown(), _breakdown(10, 5), _breakdown(1, 70)):
        with pytest.raises(ValueError):
            check(bad)


def test_two_lessons_taught_on_the_same_pages_each_get_those_pages():
    lessons = lessons_from(_breakdown(1, 10, 20, 20), book_id=BOOK, first_page=1,
                           last_page=30)
    assert [(lsn.page_from, lsn.page_to) for lsn in lessons] == [
        (1, 9), (10, 19), (20, 30), (20, 30)]
    figure = Figure(lecture_id=BOOK, page=25, s3_uri="s3://b/f.png", bbox=(0, 0, 1, 1))
    state = {"lecture_id": BOOK, "book_id": BOOK, "subject_id": SUBJECT,
             "chunks": [_chunk(5), _chunk(22)], "figures": [figure]}
    file_under_lessons(state, lessons)
    owners = [c.lecture_id for c in state["chunks"]]
    assert owners == [lessons[0].lesson_id, lessons[2].lesson_id, lessons[3].lesson_id]
    assert state["chunks"][1] is not state["chunks"][2]
    assert [f.lecture_id for f in state["figures"]] == [lessons[2].lesson_id,
                                                         lessons[3].lesson_id]
    parts = lesson_states(state)
    assert [len(p["chunks"]) for p in parts] == [1, 0, 1, 1]
    assert [len(p["figures"]) for p in parts] == [0, 0, 1, 1]


def test_a_unit_listed_as_its_own_lesson_is_sent_back():
    check = _validator(1, 50, 3.5)
    unit_as_lesson = LessonBreakdown(units=[
        UnitItem(title="Unit 1 Our Environment", unit_number=1,
                 lessons=[LessonItem(title="Unit 1  our environment", start_page=1)]),
        UnitItem(title="Unit 2 Water", unit_number=2,
                 lessons=[LessonItem(title="Lesson 1: Saving Water", start_page=20)])])
    with pytest.raises(ValueError, match="not a lesson"):
        check(unit_as_lesson)
    with pytest.raises(ValueError, match="no lessons"):
        check(LessonBreakdown(units=[
            UnitItem(title="Unit 1", lessons=[LessonItem(title="Lesson 1", start_page=1)]),
            UnitItem(title="Unit 2", lessons=[])]))


def test_lessons_run_on_across_units_in_book_order():
    breakdown = LessonBreakdown(units=[
        UnitItem(title="Unit 1", unit_number=1, lessons=[
            LessonItem(title="Lesson 1", start_page=3),
            LessonItem(title="Lesson 2", start_page=8)]),
        UnitItem(title="Unit 2", unit_number=2, lessons=[
            LessonItem(title="Lesson 1", start_page=15)])])
    lessons = lessons_from(breakdown, book_id=BOOK, first_page=1, last_page=30)
    assert [(lsn.unit_number, lsn.title, lsn.page_from, lsn.page_to) for lsn in lessons] == [
        (1, "Lesson 1", 1, 7), (1, "Lesson 2", 8, 14), (2, "Lesson 1", 15, 30)]
    assert lessons[2].label == "Unit 2 — Lesson 1"
    assert len({lsn.lesson_id for lsn in lessons}) == 3


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
    model = _Model('{"units": [{"title": "Unit 1", "unit_number": 1, "lessons": ['
                   '{"title": "Lesson 1", "start_page": 1}, '
                   '{"title": "Lesson 2", "start_page": 4}]}]}')
    lessons = detect_lessons(state, get_config(), llm=model)
    assert [(lsn.title, lsn.page_from, lsn.page_to) for lsn in lessons] == [
        ("Lesson 1", 1, 3), ("Lesson 2", 4, 8)]
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


def test_a_background_book_with_no_lessons_is_filed_as_one_not_failed():
    """Question banks and revision booklets often have no lessons to find, and
    the split refused them; as background they only need to be searchable."""
    from ingestion.books import BreakdownFailed, lessons_for
    from shared.config import get_config
    from shared.namespaces import BACKGROUND, SYLLABUS

    pages = [_page(n, "Question bank", f"question {n}") for n in range(1, 6)]
    model = _Model('{"units": []}')
    state = {"lecture_id": BOOK, "parsed_pages": pages, "lesson_title": "Bank",
             "namespace": BACKGROUND}
    lessons = lessons_for(state, get_config(), llm=model)
    assert [(lsn.title, lsn.page_from, lsn.page_to) for lsn in lessons] == [("Bank", 1, 5)]
    assert lessons[0].lesson_id == lesson_id_for(BOOK, 0)
    with pytest.raises(BreakdownFailed):
        lessons_for({**state, "namespace": SYLLABUS}, get_config(), llm=_Model('{"units": []}'))


def test_a_term_revision_after_the_last_lesson_is_filed_as_its_own_section():
    """A book's closing revision and exams were filed under its last lesson:
    133 pages of one Arabic book under "دعاء شاعرة"."""
    breakdown = _breakdown(1, 10, 20)
    breakdown.sections = [SectionItem(title="Final revision", start_page=30),
                          SectionItem(title="Model exams", start_page=45)]
    lessons = lessons_from(breakdown, book_id=BOOK, first_page=1, last_page=80)
    assert [(lsn.title, lsn.page_from, lsn.page_to) for lsn in lessons] == [
        ("Lesson 1", 1, 9), ("Lesson 2", 10, 19), ("Lesson 3", 20, 29),
        ("Final revision", 30, 44), ("Model exams", 45, 80)]
    assert lessons[3].label == "Final revision"
    assert len({lsn.lesson_id for lsn in lessons}) == 5


def test_a_section_before_the_first_lesson_keeps_its_pages():
    """A supplement opening with 105 pages of weekly assessments filed them all
    under its first lesson."""
    breakdown = _breakdown(40, 50, 60)
    breakdown.sections = [SectionItem(title="Weekly assessments", start_page=3)]
    lessons = lessons_from(breakdown, book_id=BOOK, first_page=1, last_page=70)
    assert [(lsn.title, lsn.page_from, lsn.page_to) for lsn in lessons] == [
        ("Weekly assessments", 1, 39), ("Lesson 1", 40, 49), ("Lesson 2", 50, 59),
        ("Lesson 3", 60, 70)]


def test_a_lesson_that_takes_in_the_pages_after_it_is_sent_back():
    """Correct first and last lessons ran to at most 2.7x a book's median
    lesson; those holding a term's revision ran 4x-34x, and nothing caught it."""
    check = _validator(1, 200, 3.5)
    with pytest.raises(ValueError, match="sections"):
        check(_breakdown(1, 11, 21, 31, 41))
    split = _breakdown(1, 11, 21, 31, 41)
    split.sections = [SectionItem(title="Final revision", start_page=51)]
    check(split)


def test_a_book_of_only_exams_can_be_all_sections():
    check = _validator(1, 50, 3.5)
    check(LessonBreakdown(sections=[SectionItem(title="Exam 1", start_page=1),
                                    SectionItem(title="Exam 2", start_page=20)]))
    for bad in ([SectionItem(title="A", start_page=5), SectionItem(title="B", start_page=5)],
                [SectionItem(title="", start_page=5)],
                [SectionItem(title="A", start_page=99)]):
        with pytest.raises(ValueError):
            check(LessonBreakdown(sections=bad))
    clash = _breakdown(1, 10)
    clash.sections = [SectionItem(title="Revision", start_page=10)]
    with pytest.raises(ValueError, match="one or the other"):
        check(clash)


def test_lessons_on_the_same_pages_of_a_background_book_are_filed_once():
    """Every lesson sharing pages held its own copy of them, and search drops
    repeats by chunk id, so one passage filled up to six results."""
    from ingestion.books import lessons_for
    from shared.config import get_config
    from shared.namespaces import BACKGROUND, SYLLABUS

    pages = [_page(n, f"Lessons {n}", f"text {n}") for n in range(1, 31)]
    reply = ('{"units": [{"title": "Unit 1", "unit_number": 1, "lessons": ['
             '{"title": "Lesson 1", "start_page": 1}, {"title": "Lesson 2", "start_page": 1}, '
             '{"title": "Lesson 3", "start_page": 11}, {"title": "Lesson 4", "start_page": 21}, '
             '{"title": "Lesson 5", "start_page": 21}]}]}')
    state = {"lecture_id": BOOK, "parsed_pages": pages, "namespace": BACKGROUND}
    lessons = lessons_for(state, get_config(), llm=_Model(reply))
    assert [(lsn.title, lsn.page_from, lsn.page_to) for lsn in lessons] == [
        ("Lesson 1 / Lesson 2", 1, 10), ("Lesson 3", 11, 20), ("Lesson 4 / Lesson 5", 21, 30)]
    state["chunks"] = [_chunk(n) for n in range(1, 31)]
    file_under_lessons(state, lessons)
    assert len(state["chunks"]) == 30

    ministry = lessons_for({**state, "namespace": SYLLABUS}, get_config(), llm=_Model(reply))
    assert len(ministry) == 5


class _RateLimited:
    def invoke(self, prompt):
        raise RuntimeError("Error code: 429 - Resource exhausted")


def test_a_rate_limit_while_splitting_fails_the_book_rather_than_filing_it_whole():
    """A 429 during the split was taken as "this book has no lessons", and two
    supplements were filed as one lesson each."""
    from ingestion.books import lessons_for
    from shared.config import get_config
    from shared.namespaces import BACKGROUND

    pages = [_page(n, "Unit 1", f"text {n}") for n in range(1, 6)]
    state = {"lecture_id": BOOK, "parsed_pages": pages, "namespace": BACKGROUND}
    with pytest.raises(RuntimeError, match="429"):
        lessons_for(state, get_config(), llm=_RateLimited())
