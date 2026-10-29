from __future__ import annotations

from contextlib import contextmanager

import pytest

from shared import postgres
from tests.conftest import FakePostgres

BOOK = "0743ec4c-b00d-5e17-b573-9b60b5593a26"
SUBJECT = "8c2f0a10-0001-4a00-9000-00000000000a"


class _FlakyPool:
    def __init__(self, db: FakePostgres, refusals: int) -> None:
        self.db = db
        self.refusals = refusals
        self.sizes: list = []

    @contextmanager
    def connection(self):
        if self.refusals:
            self.refusals -= 1
            raise OSError("remaining connection slots are reserved")
        yield self.db


@pytest.fixture
def flaky(monkeypatch):
    def install(refusals: int) -> _FlakyPool:
        pool = _FlakyPool(FakePostgres(), refusals)

        def pool_for(url, connect_timeout, statement_timeout_ms, config, size=None):
            pool.sizes.append(size)
            return pool

        monkeypatch.setattr(postgres, "_pool_for", pool_for)
        monkeypatch.setattr("ingestion.layout.page_cache.time.sleep", lambda _s: None)
        return pool

    return install


def _cache():
    from ingestion.layout.page_cache import PageCache
    from shared.config import get_config

    return PageCache(get_config(), model="m", prompt_version="v1")


def _page():
    from ingestion.layout.blocks import PageMeta, ParsedPage

    return ParsedPage(meta=PageMeta(doc_id=BOOK, page=3, width=600, height=800))


def test_a_refused_connection_does_not_turn_the_page_cache_off(flaky):
    """With a second book reading, the database refused one connection and the
    cache switched itself off for the rest of the machine's run, so every page
    it went on to pay for was thrown away."""
    pool = flaky(refusals=1)
    cache = _cache()
    cache.put(BOOK, 3, _page())
    assert cache.report()["cache_enabled"] is True
    assert any("page_transcriptions" in sql for sql in pool.db.sql)


def test_a_database_that_stays_down_still_turns_the_cache_off(flaky):
    """Retrying forever would hold a reader on a dead database; after the
    configured attempts the cache gives up, as it did before."""
    flaky(refusals=99)
    cache = _cache()
    cache.put(BOOK, 3, _page())
    assert cache.report()["cache_enabled"] is False


def test_the_page_cache_asks_for_its_own_small_pool(flaky):
    """Six readers a book, each with a pool of eight, filled the 79 connections
    the database allows with one book; two books could not read at once."""
    from shared.config import get_config

    pool = flaky(refusals=0)
    _cache().put(BOOK, 3, _page())
    assert pool.sizes == [get_config().get("ingestion.page_cache.pool_size")]
    assert pool.sizes[0] < get_config().get("database.pool_size")


def test_two_books_of_one_subject_register_their_lessons_one_after_the_other(fake_postgres):
    """Two books of one subject ingested together each took the next free
    lesson position at the same moment, and their lessons interleaved."""
    from ingestion.books import LessonCatalog, lesson_id_for
    from shared.config import get_config

    db = fake_postgres(fetchone=[(False,), (0,)])
    lessons = [{"lesson_id": lesson_id_for(BOOK, 0), "ordinal": 0, "title": "Lesson 1",
                "unit_number": 1}]
    LessonCatalog(get_config()).register(subject_id=SUBJECT, book_title="Book",
                                         lessons=lessons)
    statements = [sql for sql, _ in db.statements]
    lock = next(i for i, sql in enumerate(statements) if "pg_advisory_xact_lock" in sql)
    order = next(i for i, sql in enumerate(statements) if "MAX(order_index)" in sql)
    assert lock < order
    assert db.statements[lock][1] == (SUBJECT,)


def _read_with(monkeypatch, outcome: dict) -> int:
    import argparse

    import ingestion.distributed as distributed

    class _Doc:
        def __len__(self):
            return 57

        def close(self):
            pass

    monkeypatch.setattr(distributed, "lesson_state", lambda args, config: {})
    monkeypatch.setattr(distributed.PipelineContext, "build", staticmethod(lambda config: None))
    monkeypatch.setattr(distributed.pdf, "open_document", lambda source: _Doc())
    monkeypatch.setattr(distributed, "read_ahead", lambda *a, **k: dict(outcome))
    args = argparse.Namespace(source="book.pdf", shard=1, of=6, workers=8)
    return distributed._cmd_read(args, config=None)


def test_a_reader_that_reads_nothing_on_a_first_run_stops_the_book(monkeypatch):
    """A share where every call fails on a first read is a credential, quota or
    network fault, and the next machine would meet it too."""
    assert _read_with(monkeypatch, {"to_read": 57, "already_paid_for": 0,
                                    "layout_read": 0, "transcribed": 0}) == 1


def test_a_page_that_fails_again_on_a_re_run_does_not_stop_the_book(monkeypatch):
    """A re-run of a book found 56 of a reader's 57 pages already read; the one
    left had failed the first time, failed again, and the whole book stopped."""
    assert _read_with(monkeypatch, {"to_read": 1, "already_paid_for": 113,
                                    "layout_read": 0, "transcribed": 0}) == 0


def test_one_unreadable_page_on_a_first_run_does_not_stop_the_book(monkeypatch):
    """A reader of enVision Algebra 1 had one page of its 47 that needed a model.
    The model answered that the page was unreadable, nothing else had failed,
    and the whole 283-page book stopped as if the credential were dead."""
    assert _read_with(monkeypatch, {"to_read": 1, "already_paid_for": 0,
                                    "layout_read": 0, "transcribed": 0}) == 0


def test_a_books_lessons_are_finished_several_at_a_time_in_book_order(monkeypatch):
    """One lesson at a time, a 36-lesson supplement spent an hour and a half
    in finish, almost all of it waiting on model calls."""
    import threading
    import time

    import ingestion.distributed as distributed

    running, peak, built = [0], [0], []
    lock = threading.Lock()

    def stages(part, ctx, names):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.05)
        with lock:
            running[0] -= 1
        if part["lecture_id"] == "l2":
            raise RuntimeError("model unavailable")
        return {**part, "stats": {"index": {"chunks_written": 1}}}

    class _Ctx:
        def __getattr__(self, name):
            built.append(name)
            return object()

    class _Catalog:
        settled = []

        def finished(self, lesson_id, *, status, errors, version):
            _Catalog.settled.append((lesson_id, status))

    monkeypatch.setattr(distributed, "run_stages", stages)
    parts = [{"lecture_id": f"l{n}", "lesson_title": f"L{n}"} for n in range(6)]
    out = distributed._finish_lessons(parts, _Ctx(), _Catalog(), 4)
    assert [part["lecture_id"] for part, *_ in out] == [f"l{n}" for n in range(6)]
    assert [status for _p, _d, status, _e in out] == [
        "ready", "ready", "failed", "ready", "ready", "ready"]
    assert peak[0] > 1
    assert sorted(set(built)) == sorted(
        ["embedder", "topic_extractor", "dependency_deriver", "chunk_store", "vector_store"])
