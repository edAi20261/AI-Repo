from __future__ import annotations

import logging

import pytest

from ingestion.books import (
    LessonCatalog,
    SplitLesson,
    TermColumnMissing,
    _validator,
    detect_lessons,
    lesson_id_for,
    lessons_from,
    read_lesson_terms,
)
from ingestion.schemas import LessonBreakdown, LessonItem, SectionItem, UnitItem
from scripts.backfill_lesson_terms import Book, apply_plan, page_text, plan_rows, read_plan
from scripts.ingest_queue import Book as QueuedBook
from scripts.ingest_queue import ingest, parse_books

BOOK = "0743ec4c-b00d-5e17-b573-9b60b5593a26"
SUBJECT = "8c2f0a10-0001-4a00-9000-000000000001"


class _Model:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def invoke(self, prompt):
        self.prompts.append(str(prompt))
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]

        class _Message:
            content = reply

        return _Message()


def _full_year() -> LessonBreakdown:
    return LessonBreakdown(units=[
        UnitItem(title="Unit 1", unit_number=1, term=1, lessons=[
            LessonItem(title="Lesson 1", start_page=1)]),
        UnitItem(title="Unit 1", unit_number=1, term=2, lessons=[
            LessonItem(title="Lesson 1", start_page=10)])],
        sections=[SectionItem(title="Revision", start_page=20, term=2)])


def _config():
    from shared.config import get_config

    return get_config()


def test_each_lesson_of_a_book_for_the_whole_year_takes_the_term_of_its_part():
    """The G9 books for the whole year numbered their units 1-5 twice, and nothing
    said the second run was the second term."""
    lessons = lessons_from(_full_year(), book_id=BOOK, first_page=1, last_page=30)
    assert [(lsn.title, lsn.term) for lsn in lessons] == [
        ("Lesson 1", 1), ("Lesson 1", 2), ("Revision", 2)]


def test_the_term_the_operator_gives_wins_over_what_the_model_read():
    lessons = lessons_from(_full_year(), book_id=BOOK, first_page=1, last_page=30, term=2)
    assert {lsn.term for lsn in lessons} == {2}


@pytest.mark.parametrize(("given", "read"), [(2, 2), ("2", 2), ("", None), (None, None),
                                             ("second", None)])
def test_a_term_is_read_as_a_whole_number_or_none(given, read):
    assert UnitItem(title="Unit 1", term=given).term == read


@pytest.mark.parametrize("terms", [(1, 7), (2, 1)])
def test_terms_out_of_the_school_year_or_out_of_order_are_sent_back(terms):
    breakdown = LessonBreakdown(units=[
        UnitItem(title=f"Unit {n}", unit_number=n, term=term,
                 lessons=[LessonItem(title=f"Lesson {n}", start_page=n * 5)])
        for n, term in enumerate(terms, 1)])
    with pytest.raises(ValueError, match="term"):
        _validator(1, 20, 3.5, 20, _config().get("ingestion.books.max_term"))(breakdown)


def test_the_model_is_asked_for_the_term_and_its_answer_reaches_the_lessons():
    from tests.unit.test_books import _page

    pages = [_page(n, f"Lesson {n}", f"text on page {n}") for n in range(1, 5)]
    model = _Model('{"units": [{"title": "Unit 1", "unit_number": 1, "term": 2, "lessons": ['
                   '{"title": "Lesson 1", "start_page": 1}]}]}')
    lessons = detect_lessons({"lecture_id": BOOK, "parsed_pages": pages}, _config(),
                             llm=model)
    assert [lsn.term for lsn in lessons] == [2]
    assert "term" in model.prompts[0]


def _lessons(terms):
    return [{"lesson_id": lesson_id_for(BOOK, n), "ordinal": n, "title": f"Lesson {n}",
             "unit_number": 1, "term": term} for n, term in enumerate(terms)]


def test_with_a_term_column_a_books_terms_are_stored_and_the_subject_put_in_term_order(
        fake_postgres):
    """Christian Religion G7 showed the second term's lessons 0-18 and the first
    term's 19-37, because each book was appended after the last one ingested."""
    db = fake_postgres(fetchone=[(True,), (0,)])
    LessonCatalog(_config()).register(subject_id=SUBJECT, book_title="Book",
                                      lessons=_lessons([2, 2]))
    inserts = [(sql, args) for sql, args in db.statements if "INSERT INTO content.lessons" in sql]
    assert all("term" in sql and args[-1] == 2 for sql, args in inserts)
    (order,) = [sql for sql, _ in db.statements if "row_number()" in sql]
    assert "ORDER BY term NULLS LAST, order_index" in order
    assert db.committed


def test_without_a_term_column_the_book_is_still_filed_and_the_lost_terms_are_logged(
        fake_postgres, caplog):
    db = fake_postgres(fetchone=[(False,), (0,)])
    with caplog.at_level(logging.WARNING, logger="ingestion.books"):
        LessonCatalog(_config()).register(subject_id=SUBJECT, book_title="Book",
                                          lessons=_lessons([1, 1]))
    inserts = [sql for sql, _ in db.statements if "INSERT INTO content.lessons" in sql]
    assert len(inserts) == 2 and not any("term" in sql for sql in inserts)
    assert not any("row_number()" in sql for sql, _ in db.statements)
    assert "no term column" in caplog.text


def test_terms_are_never_set_on_a_database_without_the_column(fake_postgres):
    db = fake_postgres(fetchone=[(False,)])
    with pytest.raises(TermColumnMissing):
        LessonCatalog(_config()).set_terms(SUBJECT, {lesson_id_for(BOOK, 0): 1})
    assert not db.committed


def test_set_terms_stores_each_term_and_orders_the_subject_by_them(fake_postgres):
    db = fake_postgres(fetchone=[(True,)], rowcount=1)
    changed = LessonCatalog(_config()).set_terms(
        SUBJECT, {lesson_id_for(BOOK, 0): 2, lesson_id_for(BOOK, 1): None})
    sets = [args for sql, args in db.statements if "SET term" in sql]
    assert [(a["term"], a["subject"]) for a in sets] == [(2, SUBJECT), (None, SUBJECT)]
    assert any("row_number()" in sql for sql, _ in db.statements)
    assert changed == 2 and db.committed


def test_a_book_line_may_name_the_term_the_whole_book_is_for():
    (book,) = parse_books(f"s3://b/a.pdf|{SUBJECT}|ministry|2")
    seen = {}

    def run_step(step, env):
        seen[step] = env["BOOK_TERM"]
        return 0

    assert ingest(book, run_step=run_step)
    assert set(seen.values()) == {"2"}
    assert parse_books(f"s3://b/a.pdf|{SUBJECT}|ministry") == [
        QueuedBook("s3://b/a.pdf", SUBJECT, "ministry")]
    with pytest.raises(ValueError, match="term"):
        parse_books(f"s3://b/a.pdf|{SUBJECT}|ministry|second")


def test_a_term_the_school_year_does_not_have_is_refused_before_any_page_is_read(
        tmp_path, fake_postgres):
    from tests.unit.test_fast_ingestion import _lesson_state

    book = tmp_path / "book.pdf"
    book.write_bytes(b"%PDF-1.4")
    fake_postgres(fetchone=[(SUBJECT, None, "Religion", "ar", "Grade 7")] * 2)
    with pytest.raises(ValueError, match="--term"):
        _lesson_state(str(book), "--subject-id", SUBJECT, "--term", "9")


def test_the_term_of_each_split_lesson_is_read_and_every_lesson_must_be_answered():
    lessons = [SplitLesson(0, "Lesson 1", "Term 2 ..."), SplitLesson(1, "Lesson 2", "...")]
    model = _Model('{"lessons": [{"ordinal": 0, "term": 2}]}',
                   '{"lessons": [{"ordinal": 0, "term": 2}, {"ordinal": 1, "term": 2}]}')
    assert read_lesson_terms("[1] cover", lessons, _config(), llm=model) == {0: 2, 1: 2}
    assert len(model.prompts) == 2
    assert "list every lesson once" in model.prompts[1]


def test_a_transcribed_page_is_read_in_either_form_the_page_cache_holds():
    from tests.unit.test_books import _page

    assert page_text({"text": "Second Term", "figures": []}) == "Second Term"
    assert "Second Term" in page_text(_page(1, "Second Term", "Unit 1"))


def test_a_plan_is_read_book_by_book_and_a_book_that_fails_is_reported_not_dropped():
    good = Book(BOOK, SUBJECT, "", [(lesson_id_for(BOOK, 0), SplitLesson(0, "L", ""))])
    bad = Book("b" * 8, SUBJECT, "", [])

    def read(book):
        if book is bad:
            raise RuntimeError("model unavailable")
        return {0: 1}

    rows, failed = read_plan([good, bad], read, workers=2)
    assert rows == plan_rows(good, {0: 1})
    assert rows[0]["term"] == 1
    assert failed == [bad.book_id]


def test_a_reviewed_plan_is_stored_subject_by_subject():
    stored = {}

    class _Catalog:
        def set_terms(self, subject_id, terms):
            stored[subject_id] = terms
            return len(terms)

    rows = [{"subject_id": SUBJECT, "lesson_id": "a", "term": "1"},
            {"subject_id": SUBJECT, "lesson_id": "b", "term": ""},
            {"subject_id": "other", "lesson_id": "c", "term": "2"}]
    assert apply_plan(rows, _Catalog()) == 3
    assert stored == {SUBJECT: {"a": 1, "b": None}, "other": {"c": 2}}
