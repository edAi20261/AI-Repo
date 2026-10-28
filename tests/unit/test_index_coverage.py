from __future__ import annotations

from ingestion.books import lesson_id_for
from scripts.index_coverage import book_keys, candidate_ids

BOOK = "0b6f3f1e-1111-5aaa-8bbb-000000000001"


class _Pages:
    def __init__(self, pages):
        self._pages = pages

    def get_paginator(self, _name):
        return self

    def paginate(self, Bucket, Prefix):
        return self._pages.get(Prefix, [])


def test_lessons_cut_from_a_book_are_not_counted_as_books():
    """The bucket holds 307 lesson PDFs beside 195 books; counting them made the
    books that were never ingested look like five hundred."""
    client = _Pages({"ai-artifacts/books/": [{"Contents": [
        {"Key": "ai-artifacts/books/math-g7-t1.pdf"},
        {"Key": "ai-artifacts/books/lessons/subject/lesson.pdf"},
        {"Key": "ai-artifacts/books/notes.txt"},
    ]}]})
    assert book_keys(client, "b", ["ai-artifacts/books/"]) == ["ai-artifacts/books/math-g7-t1.pdf"]


def test_a_book_is_looked_up_under_the_ids_ingestion_files_its_lessons_by():
    """The index stores lesson ids, never the file's path; matching on the path
    found none of the books that were ingested the day before."""
    ids = candidate_ids(BOOK)
    assert ids[0] == BOOK
    assert lesson_id_for(BOOK, 0) in ids and lesson_id_for(BOOK, 250) in ids
