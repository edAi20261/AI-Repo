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
from shared.namespaces import SYLLABUS

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
    check = _validator(1, 50, 3.5, 20, 3)
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
    check = _validator(1, 50, 3.5, 20, 3)
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

    db = fake_postgres(fetchone=[(False,), (7,)])
    lessons = [{"lesson_id": lesson_id_for(BOOK, n), "ordinal": n, "title": f"Lesson {n}",
                "unit_number": 1} for n in range(2)]
    LessonCatalog(get_config()).register(subject_id=SUBJECT, book_title="Gem",
                                         lessons=lessons)
    inserts = [args for sql, args in db.statements if "INSERT INTO content.lessons" in sql]
    assert [(a[0], a[1], a[3], a[6]) for a in inserts] == [
        (lessons[0]["lesson_id"], SUBJECT, "Lesson 0", 7),
        (lessons[1]["lesson_id"], SUBJECT, "Lesson 1", 8)]
    assert db.committed


def test_a_book_ingested_again_under_another_subject_moves_its_lessons_there(fake_postgres):
    """A book re-ingested into its right subject kept every lesson under the
    subject it was first filed in, at that subject's numbering: the new subject
    stayed empty and the old one kept showing the book."""
    from shared.config import get_config

    db = fake_postgres(fetchone=[(False,), (0,)])
    LessonCatalog(get_config()).register(subject_id=SUBJECT, book_title="Gem", lessons=[
        {"lesson_id": lesson_id_for(BOOK, 0), "ordinal": 0, "title": "Lesson 0",
         "unit_number": 1}])
    (sql,) = [sql for sql, _ in db.statements if "INSERT INTO content.lessons" in sql]
    updated = sql.split("DO UPDATE SET", 1)[1]
    assert "subject_id = EXCLUDED.subject_id" in updated
    assert "order_index = EXCLUDED.order_index" in updated


def test_a_books_lessons_are_numbered_after_the_shown_lessons_only(fake_postgres):
    """Twelve background books came first in a subject, so the one shown was
    numbered from 583 and a student read "Lesson 588"."""
    from shared.config import get_config
    from shared.namespaces import BACKGROUND

    db = fake_postgres(fetchone=[(False,), (0,)])
    LessonCatalog(get_config()).register(subject_id=SUBJECT, book_title="Gem", lessons=[
        {"lesson_id": lesson_id_for(BOOK, 0), "ordinal": 0, "title": "Lesson 0",
         "unit_number": 1}])
    (sql, args), = [(sql, args) for sql, args in db.statements if "MAX(order_index)" in sql]
    assert "status <> 'archived'" in sql
    assert args["background_source"] == get_config().get(
        "ingestion.books.lesson_source")[BACKGROUND]


@pytest.mark.parametrize(("status", "shown"), [
    ("ready", True), ("needs_review", False), ("failed", False)])
def test_a_ministry_lesson_is_published_once_it_is_ready(fake_postgres, status, shown):
    """Every lesson a book was split into stayed a draft, and students only see
    published lessons: 3,004 ministry lessons were ready and none was shown, and
    with no author no teacher could publish them."""
    from shared.config import get_config

    db = fake_postgres()
    LessonCatalog(get_config()).finished(lesson_id_for(BOOK, 0), status=status,
                                         errors=[], version=1)
    (args,) = [args for sql, args in db.statements if "UPDATE content.lessons" in sql]
    assert args["publish"] is shown
    assert args["shown_source"] == get_config().get("ingestion.books.lesson_source")[SYLLABUS]
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
    check = _validator(1, 200, 3.5, 20, 3)
    with pytest.raises(ValueError, match="sections"):
        check(_breakdown(1, 11, 21, 31, 41))
    split = _breakdown(1, 11, 21, 31, 41)
    split.sections = [SectionItem(title="Final revision", start_page=51)]
    check(split)


def test_a_book_of_only_exams_can_be_all_sections():
    check = _validator(1, 50, 3.5, 20, 3)
    check(LessonBreakdown(sections=[SectionItem(title="Exam 1", start_page=1),
                                    SectionItem(title="Exam 2", start_page=20)]))
    for bad in ([SectionItem(title="A", start_page=5), SectionItem(title="B", start_page=5)],
                [SectionItem(title="", start_page=5)],
                [SectionItem(title="A", start_page=99)]):
        with pytest.raises(ValueError):
            check(LessonBreakdown(sections=bad))


def test_sections_listed_out_of_page_order_are_kept_in_page_order():
    """Holt Mathematics Course 1 was sent back three times because the model
    listed its back-of-book sections before its chapter reviews; every section
    started on a page of its own, and the book was never ingested."""
    check = _validator(1, 80, 3.5, 20, 3)
    breakdown = _breakdown(1, 10, 20)
    breakdown.sections = [SectionItem(title="Skills bank", start_page=60),
                          SectionItem(title="Chapter 1 review", start_page=30)]
    check(breakdown)
    lessons = lessons_from(breakdown, book_id=BOOK, first_page=1, last_page=80)
    assert [(lsn.title, lsn.page_from, lsn.page_to) for lsn in lessons][-2:] == [
        ("Chapter 1 review", 30, 59), ("Skills bank", 60, 80)]


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


def test_a_short_lesson_in_a_short_book_is_not_sent_back_as_too_long():
    """A 17-page notebook of one-page lessons had a four-page lesson sent back
    until the whole notebook was filed as one lesson."""
    check = _validator(1, 17, 3.5, 20, 3)
    check(_breakdown(1, 5, 6, 7, 8, 9, 10, 11))


def test_a_unit_revision_starting_on_its_lessons_page_shares_those_pages():
    """A supplement opens every unit with a revision on the same page as the
    unit's lessons; that answer was refused until the book was filed whole."""
    breakdown = _breakdown(6, 6, 23)
    breakdown.sections = [SectionItem(title="Unit revision", start_page=6)]
    _validator(1, 40, 3.5, 20, 3)(breakdown)
    merged = lessons_from(breakdown, book_id=BOOK, first_page=1, last_page=40,
                          shared_pages="merge")
    assert [(lsn.title, lsn.page_from, lsn.page_to) for lsn in merged] == [
        ("Lesson 1 / Lesson 2 / Unit revision", 1, 22), ("Lesson 3", 23, 40)]
    copied = lessons_from(breakdown, book_id=BOOK, first_page=1, last_page=40)
    assert [(lsn.title, lsn.page_from) for lsn in copied] == [
        ("Lesson 1", 1), ("Lesson 2", 1), ("Unit revision", 1), ("Lesson 3", 23)]


class _Replies:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def invoke(self, prompt):
        self.prompts.append(str(prompt))
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]

        class _Message:
            content = reply

        return _Message()


def test_book_apparatus_listed_as_a_lesson_is_sent_back_to_be_a_section():
    """A book's index, glossary, credits and selected answers were published as
    lessons: ~21 of them held only navigation, and the tutor declined every
    question asked on them."""
    from shared.config import get_config

    pages = [_page(n, f"Heading {n}", f"text on page {n}") for n in range(1, 13)]
    state = {"lecture_id": BOOK, "parsed_pages": pages}
    as_lesson = ('{"units": [{"title": "Unit 1", "unit_number": 1, "lessons": ['
                 '{"title": "Lesson 1", "start_page": 1, "apparatus": false}, '
                 '{"title": "Lesson 2", "start_page": 5, "apparatus": false}, '
                 '{"title": "Index", "start_page": 10, "apparatus": true}]}]}')
    as_section = ('{"units": [{"title": "Unit 1", "unit_number": 1, "lessons": ['
                  '{"title": "Lesson 1", "start_page": 1, "apparatus": false}, '
                  '{"title": "Lesson 2", "start_page": 5, "apparatus": false}]}], '
                  '"sections": [{"title": "Index", "start_page": 10, "apparatus": true}]}')
    model = _Replies(as_lesson, as_section)
    lessons = detect_lessons(state, get_config(), llm=model)
    assert len(model.prompts) == 2
    assert "Index" in model.prompts[1]
    assert [(lsn.title, lsn.apparatus) for lsn in lessons] == [
        ("Lesson 1", False), ("Lesson 2", False), ("Index", True)]


def _revision_and_index() -> list:
    breakdown = _breakdown(1, 10)
    breakdown.sections = [SectionItem(title="Final revision", start_page=20),
                          SectionItem(title="Index", start_page=28, apparatus=True)]
    return lessons_from(breakdown, book_id=BOOK, first_page=1, last_page=30)


def test_book_apparatus_is_filed_where_no_student_is_shown_it(fake_postgres):
    """A section became a ministry lesson like any other and was published the
    moment it was ready: an index, a glossary, a list of credits. A term's
    revision, also a section, is used by students and stays a shown lesson."""
    from shared.config import get_config
    from shared.namespaces import BACKGROUND

    state = {"lecture_id": BOOK, "book_id": BOOK, "subject_id": SUBJECT, "chunks": [],
             "figures": []}
    file_under_lessons(state, _revision_and_index())
    db = fake_postgres(fetchone=[(False,), (0,)])
    LessonCatalog(get_config()).register(subject_id=SUBJECT, book_title="Gem",
                                         lessons=state["lessons"])
    sources = get_config().get("ingestion.books.lesson_source")
    upserts = [(sql, args) for sql, args in db.statements if "INSERT INTO content.lessons" in sql]
    assert [(args[3], args[2]) for _, args in upserts] == [
        ("Lesson 1", sources[SYLLABUS]), ("Lesson 2", sources[SYLLABUS]),
        ("Final revision", sources[SYLLABUS]), ("Index", sources[BACKGROUND])]
    sql = upserts[0][0]
    assert "source = EXCLUDED.source" in sql
    assert "'published' AND EXCLUDED.source <>" in sql


def test_only_book_apparatus_is_hidden_from_students():
    from ingestion.books import shown
    from shared.namespaces import BACKGROUND

    assert shown({"apparatus": False}, SYLLABUS)
    assert not shown({"apparatus": True}, SYLLABUS)
    assert not shown({"apparatus": False}, BACKGROUND)


def test_a_book_split_into_lessons_tells_each_lesson_whether_it_is_shown():
    """A term's revision and model exams are published today and students use
    them; only the book's apparatus is kept from them."""
    state = {"lecture_id": BOOK, "book_id": BOOK, "subject_id": SUBJECT,
             "namespace": SYLLABUS, "chunks": [], "figures": []}
    file_under_lessons(state, _revision_and_index())
    state["lessons"][0]["pdf"] = {"key": "k"}
    state["lessons"][1]["pdf_error"] = "AccessDenied"
    parts = lesson_states(state)
    assert [p["shown"] for p in parts] == [True, True, True, False]
    assert [p["lesson_pdf"] for p in parts] == [{"key": "k"}, None, None, None]
    assert [p["lesson_pdf_error"] for p in parts] == [None, "AccessDenied", None, None]


def test_apparatus_sharing_a_page_with_a_lesson_does_not_hide_the_lesson():
    breakdown = _breakdown(1, 10)
    breakdown.sections = [SectionItem(title="Credits", start_page=10, apparatus=True)]
    merged = lessons_from(breakdown, book_id=BOOK, first_page=1, last_page=20,
                          shared_pages="merge")
    assert [(lsn.title, lsn.apparatus) for lsn in merged] == [
        ("Lesson 1", False), ("Lesson 2 / Credits", False)]


def test_a_lesson_named_like_its_unit_among_other_lessons_is_a_lesson():
    """Reveal Math Course 1's Module 7 is "Relationships Between Two Variables"
    and so is its Lesson 7-1; the book failed on every attempt for listing it."""
    check = _validator(1, 24, 3.5, 20, 3)
    check(LessonBreakdown(units=[UnitItem(
        title="Relationships Between Two Variables", unit_number=7, lessons=[
            LessonItem(title="Relationships Between Two Variables", start_page=1),
            LessonItem(title="Write Equations to Represent Relationships", start_page=9),
            LessonItem(title="Graphs of Relationships", start_page=17)])]))


def test_a_long_lesson_the_model_keeps_after_being_asked_again_is_kept(caplog):
    """Into Literature's Diary of Anne Frank, Act One, is one 55-page play. Each
    attempt was sent back as too long, so the book was never ingested."""
    check = _validator(1, 200, 3.5, 20, 3)
    play = _breakdown(1, 11, 21, 31, 41)
    with pytest.raises(ValueError, match="sections"):
        check(play)
    with caplog.at_level("WARNING", logger="ingestion.books"):
        check(play)
    assert "kept" in caplog.text


def test_long_lessons_are_questioned_once_per_book(caplog):
    """Asked about the Anne Frank play, the model split it and then listed two
    paired texts as one 44-page lesson; that was sent back too, and the book
    failed on its last attempt."""
    check = _validator(1, 200, 3.5, 20, 3)
    with pytest.raises(ValueError, match="sections"):
        check(_breakdown(1, 11, 21, 31, 41))
    with caplog.at_level("WARNING", logger="ingestion.books"):
        check(_breakdown(1, 11, 21, 31, 36))
    assert "kept" in caplog.text
