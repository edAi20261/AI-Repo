"""The one way this service opens a Postgres connection.

Every read tool used to call ``psycopg.connect`` itself — five times inside the
module that already defined a ``connect`` helper, and once more in the session
store — so a connection's timeouts were whatever the call site remembered. Four
of the six remembered nothing, which means a database that hangs rather than
refuses could hold a student's turn open until the HTTP client gave up. psycopg
has no default timeout; the only bound is the caller's patience.

Timeouts belong here, where they cannot be bypassed, rather than at the call
sites that already proved they would be.

The offline pipeline is deliberately not routed through this: batch ingestion
runs statements that are *meant* to take minutes, and a statement timeout tuned
for a student's turn would kill them.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from .config import Config


def pg_url(raw: str) -> str:
    """Strip the SQLAlchemy driver suffix so psycopg accepts the URL."""
    return raw.replace("+asyncpg", "").replace("+psycopg", "")


@contextmanager
def connect(
    config: Config | None = None,
    *,
    connect_timeout: Optional[int] = None,
    statement_timeout_ms: Optional[int] = None,
) -> Iterator[object]:
    """A psycopg connection to ``ai.*``, bounded on both ends.

    Read tools use ``with connect(config) as conn``; write tools call
    ``conn.commit()`` before the block exits. The overrides exist for callers
    that sit somewhere hotter than a tool call and want a tighter bound.
    """
    import psycopg

    config = config or Config.load()
    if connect_timeout is None:
        connect_timeout = config.get("database.connect_timeout_seconds")
    if statement_timeout_ms is None:
        statement_timeout_ms = config.get("database.statement_timeout_ms")

    conn = psycopg.connect(
        pg_url(config.secret("DATABASE_URL")),
        connect_timeout=connect_timeout,
        options=f"-c statement_timeout={statement_timeout_ms}",
    )
    try:
        yield conn
    finally:
        conn.close()
