from __future__ import annotations

import os

import pytest

from scripts.ingest_queue import Book, main, parse_books

SUBJECT = "8c2f0a10-0001-4a00-9000-00000000000a"


def test_a_book_list_names_each_books_source_subject_and_kind():
    books = parse_books(f"""
        # supplements
        s3://b/books/a.pdf|{SUBJECT}|external

        s3://b/books/b.pdf | {SUBJECT} | ministry
    """)
    assert books == [Book("s3://b/books/a.pdf", SUBJECT, "external"),
                     Book("s3://b/books/b.pdf", SUBJECT, "ministry")]


@pytest.mark.parametrize("text", ["", "s3://b/a.pdf|external", f"s3://b/a.pdf|{SUBJECT}|bonus",
                                  f"|{SUBJECT}|external"])
def test_a_book_list_that_cannot_be_read_is_refused_before_anything_runs(text):
    with pytest.raises(ValueError):
        parse_books(text)


def test_each_book_runs_the_four_steps_on_this_machine_in_its_own_folder():
    """With no jobs to fan out across, each step is the only share of itself."""
    seen: list[tuple[str, str, str, str, str]] = []

    def run_step(step, env):
        seen.append((step, env["SOURCE"], env["BOOK_KIND"], env["CLOUD_RUN_TASK_COUNT"],
                     env["RUN_DIR"]))
        assert os.path.isdir(env["RUN_DIR"])
        return 0

    books = [Book("s3://b/a.pdf", SUBJECT, "external"), Book("s3://b/b.pdf", SUBJECT, "ministry")]
    assert main(books, run_step=run_step) == 0
    assert [(s[0], s[1], s[2], s[3]) for s in seen] == [
        (step, book.source, book.kind, "1") for book in books
        for step in ("read", "prepare", "embed", "finish")]
    assert len({s[4] for s in seen}) == 2
    assert not any(os.path.exists(s[4]) for s in seen)


def test_a_book_that_fails_is_reported_and_the_next_one_still_runs():
    calls: list[tuple[str, str]] = []

    def run_step(step, env):
        calls.append((step, env["SOURCE"]))
        return 1 if (step, env["SOURCE"]) == ("prepare", "s3://b/a.pdf") else 0

    books = [Book("s3://b/a.pdf", SUBJECT, "external"), Book("s3://b/b.pdf", SUBJECT, "external")]
    assert main(books, run_step=run_step) == 1
    assert calls[:2] == [("read", "s3://b/a.pdf"), ("prepare", "s3://b/a.pdf")]
    assert calls[2:] == [(step, "s3://b/b.pdf") for step in ("read", "prepare", "embed", "finish")]
