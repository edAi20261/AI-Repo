"""The one way this service reaches Qdrant.

There were eight constructions across four packages and four different ideas of
a timeout — 15 seconds in one script, 120 in the ingestion store, and nothing at
all in the three the agent uses on a student's turn. A Qdrant that hung rather
than refused therefore hung the turn: the SDK's own default is generous, and
retrieval sits between the student's question and every word of the answer.

The two budgets are genuinely different, so they are two keys rather than one
compromise. Upserting a lesson is batch work that is allowed to take minutes;
answering a question is not.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import Optional

from .config import Config


def query_client(config: Config, *, timeout: Optional[float] = None):
    """A Qdrant client for the read path — bounded by the query budget."""
    if timeout is None:
        timeout = config.get("vector_store.query_timeout_seconds")
    return _shared_client(config.secret("QDRANT_URL", "http://localhost:6333"),
                          config.secret("QDRANT_API_KEY"), timeout,
                          config.get("vector_store.max_concurrent_queries"))


@lru_cache(maxsize=8)
def _shared_client(url: str, api_key: Optional[str], timeout: float, concurrency: int):
    """One client per process, so its connections are kept and reused.

    A client built per search paid a TLS handshake each time: 0.38s for a
    search that takes 0.13s on a warm connection, several times a turn.
    """
    from qdrant_client import QdrantClient

    return _Bounded(QdrantClient(url=url, api_key=api_key, timeout=timeout),
                    threading.BoundedSemaphore(concurrency))


class _Bounded:
    """A client whose calls wait for one of a fixed number of slots."""

    def __init__(self, client, slots: threading.BoundedSemaphore) -> None:
        self._client, self._slots = client, slots

    def __getattr__(self, name: str):
        attr = getattr(self._client, name)
        if not callable(attr):
            return attr

        def bounded(*args, **kwargs):
            with self._slots:
                return attr(*args, **kwargs)

        return bounded


def upsert_client(config: Config, location: str, *, timeout: Optional[float] = None):
    """A Qdrant client for the write path — bounded by the batch budget.

    ``location`` is ``:memory:`` for the in-process store the tests and the dry
    run use, where there is no connection to time out.
    """
    from qdrant_client import QdrantClient

    if location == ":memory:":
        return QdrantClient(location=":memory:")
    if timeout is None:
        timeout = config.get("vector_store.timeout_seconds")
    return QdrantClient(
        url=location, api_key=config.secret("QDRANT_API_KEY"), timeout=int(timeout)
    )
