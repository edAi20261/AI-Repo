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

Connections are pooled, for the same reason the timeouts are here rather than
at the call sites: it is the one place that can be changed once. Measured
against the production instance, opening a connection took 486ms and the read
it was opened for took 65ms — so a turn spent 86% of its database time saying
hello, and two turns arriving together said it twice. That is the shape of an
intermittent failure: fine alone, slow in pairs, a timeout under load.
"""

from __future__ import annotations

import atexit
import logging
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

from .config import Config

logger = logging.getLogger(__name__)

# One pool per distinct bound. Callers that want a tighter statement timeout
# than a tool call get their own pool rather than a connection carrying
# somebody else's setting: `options` is fixed when a connection is opened, and
# a pool that mixed them would hand a 5s turn a 500ms connection at random.
_pools: dict[tuple, object] = {}
_lock = threading.Lock()


def pg_url(raw: str) -> str:
    """Strip the SQLAlchemy driver suffix so psycopg accepts the URL."""
    return raw.replace("+asyncpg", "").replace("+psycopg", "")


def _pool_for(url: str, connect_timeout: int, statement_timeout_ms: int,
              config: Config) -> object:
    """The pool serving this exact bound, opened the first time it is wanted.

    Built the first time a connection is wanted, never at import: resolving
    one on a health-check path is how a deploy loop starts. It fills in the
    background, so the turn that builds it does not wait for all of it.

    ``check`` costs one round trip and buys back the thing pooling otherwise
    takes away — a connection the server has dropped since it was last used is
    replaced here rather than raising inside a student's turn. The sweep behind
    that choice is in config.yaml beside the sizes.
    """
    from psycopg_pool import ConnectionPool

    key = (url, connect_timeout, statement_timeout_ms)
    with _lock:
        pool = _pools.get(key)
        if pool is None:
            size = config.get("database.pool_size")
            pool = ConnectionPool(
                url,
                min_size=size,
                max_size=size,
                max_lifetime=config.get("database.pool_max_lifetime_seconds"),
                timeout=config.get("database.pool_wait_seconds"),
                check=(ConnectionPool.check_connection
                       if config.get("database.pool_check_connections") else None),
                kwargs={"connect_timeout": connect_timeout,
                        "options": f"-c statement_timeout={statement_timeout_ms}"},
                open=True,
            )
            _pools[key] = pool
            logger.info("opened a connection pool", extra={
                "size": size, "statement_timeout_ms": statement_timeout_ms})
    return pool


@contextmanager
def connect(
    config: Config | None = None,
    *,
    connect_timeout: Optional[int] = None,
    statement_timeout_ms: Optional[int] = None,
) -> Iterator[object]:
    """A pooled psycopg connection to ``ai.*``, bounded on both ends.

    Read tools use ``with connect(config) as conn``; write tools call
    ``conn.commit()`` before the block exits. The overrides exist for callers
    that sit somewhere hotter than a tool call and want a tighter bound.

    The connection goes back to the pool at the end of the block, with any
    transaction the caller left open rolled back — so a read that forgot to
    commit cannot leak its transaction into the next turn to borrow it.
    """
    config = config or Config.load()
    if connect_timeout is None:
        connect_timeout = config.get("database.connect_timeout_seconds")
    if statement_timeout_ms is None:
        statement_timeout_ms = config.get("database.statement_timeout_ms")

    pool = _pool_for(pg_url(config.secret("DATABASE_URL")),
                     connect_timeout, statement_timeout_ms, config)
    with pool.connection() as conn:
        yield conn


def close_pools() -> None:
    """Close every pool. Registered at exit so a script does not hang on one."""
    with _lock:
        for pool in _pools.values():
            try:
                pool.close()
            except Exception as exc:
                logger.warning("a connection pool would not close",
                               extra={"error": str(exc)[:200]})
        _pools.clear()


atexit.register(close_pools)
