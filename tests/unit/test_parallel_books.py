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

    db = fake_postgres(fetchone=[(0,)])
    lessons = [{"lesson_id": lesson_id_for(BOOK, 0), "ordinal": 0, "title": "Lesson 1",
                "unit_number": 1}]
    LessonCatalog(get_config()).register(subject_id=SUBJECT, book_title="Book",
                                         lessons=lessons)
    statements = [sql for sql, _ in db.statements]
    lock = next(i for i, sql in enumerate(statements) if "pg_advisory_xact_lock" in sql)
    order = next(i for i, sql in enumerate(statements) if "MAX(order_index)" in sql)
    assert lock < order
    assert db.statements[lock][1] == (SUBJECT,)
