"""Doubles shared by every test that needs one.

There were nine: two Postgres, seven LLM, three embedder, two S3 — each written
by whoever needed one that day, each with a different idea of what the thing it
replaced looks like. That is how the Postgres doubles came to accept
``connect(url)`` and nothing else, so adding a timeout to the real connection
broke fourteen tests that were not about connections at all.

A double's job is to fail when production would. Nine of them can only agree by
coincidence.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager

import pytest

from shared import postgres


class FakePostgres:
    """Stands in for a psycopg connection and its cursors.

    Both, deliberately: the tests want to assert on the statements a call made,
    and threading that through a separate cursor object bought nothing.

    ``fetchone``/``fetchall`` take a queue — one entry per call, in order, for a
    tool that runs several statements. ``rows`` is the simpler form: the same
    result whenever it is asked. ``rowcount`` is what every statement reports
    having touched.
    """

    def __init__(self, *, fetchone=None, fetchall=None, rows=None, explode=False,
                 rowcount=0):
        # psycopg reports how many rows a statement touched, and a DELETE that
        # reports nothing is how a "nothing was deleted" receipt gets written
        # over a delete that worked.
        self.rowcount = rowcount
        self.sql: list[str] = []
        self.statements: list[tuple[str, object]] = []
        self.last_args: object = None
        self.committed = False
        self.rows = list(rows or [])
        self.explode = explode
        self._one = deque(fetchone or [])
        self._all = deque(fetchall or [])

    # --- connection ---
    def cursor(self):
        return self

    def commit(self):
        self.committed = True

    def close(self):
        pass

    # --- cursor ---
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        if self.explode:
            raise RuntimeError("database unreachable")
        self.sql.append(sql)
        self.last_args = params
        self.statements.append((" ".join(sql.split()), params))

    def fetchone(self):
        return self._one.popleft() if self._one else None

    def fetchall(self):
        return self._all.popleft() if self._all else self.rows


class RealDatabaseTouched(AssertionError):
    """A unit test reached for a real Postgres connection."""


@pytest.fixture(autouse=True)
def _no_real_database(monkeypatch):
    """Locally ``DATABASE_URL`` is production, so a test that forgot its double would
    read, or write, the real thing and still pass. Every test starts with the pool
    refusing; ``fake_postgres`` replaces it for the tests that want one."""
    import psycopg
    import psycopg_pool

    def _refuse(*a, **k):
        raise RealDatabaseTouched(
            "a unit test opened a real database connection; use the fake_postgres fixture")

    class _RefusingPool:
        check_connection = staticmethod(_refuse)

        def __init__(self, *a, **k):
            _refuse()

    monkeypatch.setattr(psycopg_pool, "ConnectionPool", _RefusingPool)
    monkeypatch.setattr(psycopg, "connect", _refuse)
    monkeypatch.setattr(postgres, "_pools", {})


@pytest.fixture(autouse=True)
def _no_real_models(monkeypatch):
    """CI has no .env, so every model call takes the deterministic stub path; a laptop
    with a provider key in .env made real calls instead, which turned a 40-second run
    into eleven minutes and a network hiccup into a failed test. Every test now sees
    what CI sees; a test that wants a key sets one itself."""
    from shared import config as config_module
    from shared import llm

    keys = {env for _, env in llm._OPENAI_COMPAT.values()} | set(llm._NATIVE_KEY.values())
    for name in (*keys, "APP_ENV"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
    monkeypatch.setattr(config_module, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(config_module, "_INSTANCE", None)


@pytest.fixture()
def fake_postgres(monkeypatch):
    """Install a Postgres double; return it so the test can inspect it.

    Takes either a prepared :class:`FakePostgres` or the arguments to build one.
    """

    def install(db: FakePostgres | None = None, **kwargs) -> FakePostgres:
        db = db if db is not None else FakePostgres(**kwargs)

        class _Pool:
            """A pool of exactly one double, handed out as often as asked."""

            @contextmanager
            def connection(self):
                yield db

        # Patched at the pool rather than at psycopg.connect: connections are
        # pooled now, so nothing on a turn calls psycopg.connect any more and a
        # double installed there would never be reached.
        monkeypatch.setattr(postgres, "_pool_for", lambda *a, **k: _Pool())
        monkeypatch.setattr(postgres, "_open_once", lambda *a, **k: db)
        return db

    return install
