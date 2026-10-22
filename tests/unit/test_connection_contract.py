"""Every connection this service opens is bounded on both ends.

Postgres and Qdrant both, because the failure is the same one twice.

psycopg has no default timeout. A host that hangs rather than refuses holds the
connection open until the caller gives up, which for a student's turn means
whatever the HTTP client allows — the turn does not fail, it stalls.

Five read tools opened their own connection instead of calling the helper that
already sat above them in the same file, and none of the five passed a timeout.
That is the shape of bug this file exists to make impossible: not "someone wrote
it wrong" but "the right way was optional".
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import shared.postgres as postgres
import shared.vector_store as vector_store
from shared.config import Config

ROOT = Path(__file__).resolve().parents[2]

# The offline pipeline is deliberately exempt: batch statements are meant to take
# minutes, and it talks to Postgres through SQLAlchemy rather than psycopg.
SERVED_AT_RUNTIME = ("agent", "system", "shared")


def _connect_calls(tree: ast.AST) -> list[ast.Call]:
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "connect"
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "psycopg"]


def _runtime_modules() -> list[Path]:
    return sorted(p for package in SERVED_AT_RUNTIME
                  for p in (ROOT / package).rglob("*.py"))


@pytest.mark.parametrize("path", _runtime_modules(), ids=lambda p: p.name)
def test_nothing_on_a_turn_opens_its_own_connection(path: Path):
    """One connection helper, so one place decides what a hang costs."""
    if path == Path(postgres.__file__):
        return
    calls = _connect_calls(ast.parse(path.read_text()))
    assert not calls, (
        f"{path.relative_to(ROOT)}:{calls[0].lineno} calls psycopg.connect "
        "directly — use shared.postgres.connect, which carries the timeouts")


def _pooled_with(monkeypatch, config, *, reset=True, **overrides):
    """Build a pool with the real settings and a fake connection underneath."""
    import psycopg_pool

    seen = {}

    class _Conn:
        def close(self):
            pass

    class _FakePool:
        def __init__(self, url, **kwargs):
            seen.update(kwargs)
            seen["url"] = url

        def connection(self):
            from contextlib import contextmanager

            @contextmanager
            def _held():
                yield _Conn()

            return _held()

        def close(self):
            pass

    _FakePool.check_connection = staticmethod(lambda conn: None)
    if reset:
        postgres._pools.clear()
    monkeypatch.setattr(psycopg_pool, "ConnectionPool", _FakePool)
    with postgres.connect(config, **overrides):
        pass
    return seen


def test_the_helper_passes_both_timeouts(monkeypatch):
    """The timeouts are the reason the helper exists.

    They now reach the server as the pool's connection kwargs rather than as
    one call's arguments, which is the same guarantee from one level further
    back: every connection the pool hands out was opened with them."""
    seen = _pooled_with(monkeypatch, Config.load())

    assert seen["kwargs"]["connect_timeout"] > 0
    assert "statement_timeout=" in seen["kwargs"]["options"]


def test_a_caller_can_ask_for_a_tighter_bound(monkeypatch):
    """The session store sits at the front of every turn and wants less than a
    tool call does."""
    seen = _pooled_with(monkeypatch, Config.load(),
                        connect_timeout=2, statement_timeout_ms=500)

    assert seen["kwargs"]["connect_timeout"] == 2
    assert seen["kwargs"]["options"] == "-c statement_timeout=500"


def test_two_bounds_do_not_share_one_pool(monkeypatch):
    """`options` is fixed when a connection is opened. One pool serving both
    bounds would hand a tool call's 5s turn a connection carrying the session
    store's 500ms, at random — an intermittent failure by construction."""
    _pooled_with(monkeypatch, Config.load(), statement_timeout_ms=500)
    _pooled_with(monkeypatch, Config.load(), statement_timeout_ms=5000, reset=False)

    assert len(postgres._pools) == 2


def test_the_pool_is_full_so_a_burst_never_pays_for_a_handshake(monkeypatch):
    """A pool that grows on demand opens connections exactly when it is
    busiest. Measured at 8 concurrent reads, growing on demand put p95 at
    474ms against 239ms for a full one — the tail this was meant to remove."""
    seen = _pooled_with(monkeypatch, Config.load())

    assert seen["min_size"] == seen["max_size"] > 0


def test_a_connection_is_proved_alive_before_a_turn_gets_it(monkeypatch):
    """Without this, a connection the server dropped while it sat in the pool
    raises inside a student's turn — an intermittent failure by construction,
    and the harder kind to reproduce."""
    seen = _pooled_with(monkeypatch, Config.load())

    assert seen["check"] is not None


def test_a_connection_that_died_while_idle_fails_fast_instead_of_hanging(monkeypatch):
    """The connection check sends SELECT 1 on a pooled socket. When the socket
    was dropped silently while idle (no RST), nothing answers and nothing is
    bounded: statement_timeout lives on the server, which never sees the query.
    A tester's run sat in psycopg's wait() until the 120s turn limit and every
    database-backed request answered 504 until the process was restarted.

    Keepalives find a socket that died while idle; tcp_user_timeout bounds a
    send that is never acknowledged. Both must end well inside a turn."""
    config = Config.load()
    kwargs = _pooled_with(monkeypatch, config)["kwargs"]

    assert kwargs["keepalives"] == 1
    idle_detection = (kwargs["keepalives_idle"]
                      + kwargs["keepalives_interval"] * kwargs["keepalives_count"])
    unacknowledged = kwargs["tcp_user_timeout"] / 1000
    assert max(idle_detection, unacknowledged) < config.get("api.request_timeout") / 2


# --- Qdrant -----------------------------------------------------------------

def _qdrant_constructions(tree: ast.AST) -> list[ast.Call]:
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "QdrantClient"]


@pytest.mark.parametrize("path", _runtime_modules(), ids=lambda p: p.name)
def test_nothing_on_a_turn_builds_its_own_qdrant_client(path: Path):
    """Three of the eight constructions were on the read path and none of the
    three passed a timeout, so a hung cluster hung the turn — retrieval sits
    between the question and every word of the answer."""
    if path == Path(vector_store.__file__):
        return
    calls = _qdrant_constructions(ast.parse(path.read_text()))
    assert not calls, (
        f"{path.relative_to(ROOT)}:{calls[0].lineno} builds a QdrantClient "
        "directly — use shared.vector_store, which carries the budgets")


def test_the_read_path_is_bounded_far_tighter_than_the_write_path():
    """One number could not serve both: a lesson's upsert is allowed minutes,
    a student's search is not."""
    config = Config.load()
    assert config.get("vector_store.query_timeout_seconds") < \
        config.get("vector_store.timeout_seconds")


def test_the_query_client_carries_the_query_budget(monkeypatch):
    seen = {}
    import qdrant_client
    monkeypatch.setattr(qdrant_client, "QdrantClient",
                        lambda **kwargs: seen.update(kwargs) or object())
    vector_store._shared_client.cache_clear()
    vector_store.query_client(Config.load())
    assert seen["timeout"] == Config.load().get("vector_store.query_timeout_seconds")


def test_searches_share_one_query_client(monkeypatch):
    """A client per search opened a new TLS connection to Qdrant every time:
    0.38s for a search a warm connection answers in 0.13s."""
    import qdrant_client
    built = []
    monkeypatch.setattr(qdrant_client, "QdrantClient",
                        lambda **kwargs: built.append(kwargs) or object())
    vector_store._shared_client.cache_clear()
    first = vector_store.query_client(Config.load())
    second = vector_store.query_client(Config.load())
    vector_store._shared_client.cache_clear()
    assert first is second and len(built) == 1


def test_searches_beyond_the_cap_wait_rather_than_reach_qdrant(monkeypatch):
    """Thirty turns and a topic batch searching at once pushed Qdrant past the
    5s read timeout, and the turns waiting on those searches failed."""
    import threading
    import time

    import qdrant_client

    inside, peak, lock = [0], [0], threading.Lock()

    class _Slow:
        def query_points(self, *a, **k):
            with lock:
                inside[0] += 1
                peak[0] = max(peak[0], inside[0])
            time.sleep(0.05)
            with lock:
                inside[0] -= 1

    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda **kwargs: _Slow())
    vector_store._shared_client.cache_clear()
    client = vector_store.query_client(Config.load())
    threads = [threading.Thread(target=client.query_points) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    vector_store._shared_client.cache_clear()
    assert peak[0] == Config.load().get("vector_store.max_concurrent_queries")
