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
        return db

    return install
