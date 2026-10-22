"""How many turns the service runs at once, and what waits for what.

Ten tutor messages sent together in pre-production: eight began in under a
second and the ninth and tenth waited 18-19 s, because every call into the
agent shared one limiter of eight. A summary or a stored report waited behind
chat turns the same way, and /ai/memory ran its queries on the event loop,
stalling every other request while it did.
"""

from __future__ import annotations

import threading

import anyio
import pytest

from shared.config import get_config
from system.billing import Billing
from system.main import turn_executor
from system.runtime import AgentRuntime
from system.settings import Settings


class _HeldGraph:
    """A graph whose turns hold their slot until they are released."""

    def __init__(self) -> None:
        self.release = threading.Event()

    def invoke(self, _payload):
        self.release.wait(5)
        return {"response": "done"}

    async def astream(self, _payload, stream_mode=None):
        while not self.release.is_set():
            await anyio.sleep(0.01)
        yield {"compose": {"response": "done"}}


def _runtime(graph, *, max_concurrency=1, max_reads=1) -> AgentRuntime:
    return AgentRuntime(get_config(reload=True), graph, None, billing=Billing(None),
                        max_concurrency=max_concurrency, max_reads=max_reads, timeout=10)


def test_a_turn_waiting_for_a_slot_hears_back_at_once(monkeypatch):
    """The stream answered its headers and then said nothing until a slot
    freed, so a waiting turn looked exactly like a hung one."""
    monkeypatch.setenv("APP_ENV", "test")
    graph = _HeldGraph()
    runtime = _runtime(graph)

    async def run():
        first = await runtime.stream_chat({"user_id": "u-1", "message": "hi"})
        second = await runtime.stream_chat({"user_id": "u-2", "message": "hi"})
        async with anyio.create_task_group() as group:
            group.start_soon(lambda: _drain(first))
            await anyio.sleep(0.05)
            with anyio.fail_after(1):
                stage, _ = await second.__anext__()
            graph.release.set()
            group.start_soon(lambda: _drain(second))
        return stage

    assert anyio.run(run) == "queued"


async def _drain(updates):
    async for _ in updates:
        pass


def test_a_page_read_does_not_wait_behind_chat_turns(monkeypatch):
    """A learner summary took 0.2 s alone and 11.6 s at p95 while quizzes were
    being marked: it queued for the same slots as the turns."""
    monkeypatch.setenv("APP_ENV", "test")
    graph = _HeldGraph()
    runtime = _runtime(graph)
    monkeypatch.setattr("system.runtime.held", lambda user_id, config=None: {"user": user_id})

    async def run():
        async with anyio.create_task_group() as group:
            group.start_soon(runtime.chat, {"user_id": "u-1", "message": "hi"})
            await anyio.sleep(0.05)
            assert runtime.in_flight == 1
            with anyio.fail_after(1):
                record = await runtime.memory("u-2")
            graph.release.set()
        return record

    assert anyio.run(run) == {"user": "u-2"}


def test_turns_beyond_the_first_eight_run_at_once(monkeypatch):
    """The ninth turn queued for a slot while the model calls of the first eight
    were only waiting on the network."""
    monkeypatch.setenv("APP_ENV", "test")
    settings = Settings.from_config(get_config(reload=True))
    assert settings.max_concurrency > 8
    started = []
    graph = _HeldGraph()
    runtime = _runtime(graph, max_concurrency=settings.max_concurrency)

    def turn(payload):
        started.append(payload["user_id"])
        graph.release.wait(5)
        return {"response": "done"}

    graph.invoke = turn

    async def run():
        async with anyio.create_task_group() as group:
            for i in range(10):
                group.start_soon(runtime.chat, {"user_id": f"u-{i}", "message": "hi"})
            await anyio.sleep(0.2)
            count = len(started)
            graph.release.set()
        return count

    assert anyio.run(run) == 10


@pytest.mark.parametrize("cpus", [1, 2])
def test_streamed_turns_are_not_capped_by_the_default_executor(monkeypatch, cpus):
    """A streamed turn's nodes run on the loop's default executor, which holds
    min(32, cpus + 4) threads: six on a one-vCPU task, fewer than the turns the
    limiter lets in."""
    monkeypatch.setenv("APP_ENV", "test")
    settings = Settings.from_config(get_config(reload=True))
    executor = turn_executor(settings)
    try:
        assert executor._max_workers >= settings.max_concurrency > min(32, cpus + 4)
    finally:
        executor.shutdown()


class _Conn:
    def __init__(self) -> None:
        self.rolled_back = 0
        self.info = type("Info", (), {"transaction_status": 0})()

    def rollback(self):
        self.rolled_back += 1
        self.info.transaction_status = 0


class _CountingPool:
    def __init__(self) -> None:
        self.checkouts = 0
        self.conn = _Conn()

    def connection(self):
        from contextlib import contextmanager

        @contextmanager
        def checkout():
            self.checkouts += 1
            yield self.conn

        return checkout()


def test_a_summary_reads_on_one_connection(monkeypatch):
    """The learner summary checked a connection out for each of its seven
    queries, paying a liveness round trip each time and queuing for the pool
    seven times under load."""
    from agent import analytics
    from shared import postgres
    from shared.postgres import connect

    pool = _CountingPool()
    monkeypatch.setattr(postgres, "_pool_for", lambda *a, **k: pool)
    seen = []

    def _read(*_args, **_kwargs):
        with connect(get_config()) as conn:
            seen.append(conn)
        return []

    for name in ("get_scope_topics", "get_mastery", "get_weekly_activity",
                 "get_mastery_trend", "get_top_mistakes"):
        monkeypatch.setattr(analytics, name, _read)

    analytics.learner_summary("u-1", config=get_config())

    assert pool.checkouts == 1 and len(seen) >= 4 and len(set(map(id, seen))) == 1


def test_a_failed_read_does_not_poison_the_shared_connection(monkeypatch):
    """One failed statement leaves a Postgres transaction aborted, and every
    later query on that connection would fail with it."""
    from psycopg.pq import TransactionStatus

    from shared import postgres
    from shared.postgres import connect, one_connection

    pool = _CountingPool()
    monkeypatch.setattr(postgres, "_pool_for", lambda *a, **k: pool)

    with one_connection(get_config()):
        with connect(get_config()) as conn:
            conn.info.transaction_status = TransactionStatus.INERROR
        with connect(get_config()) as conn:
            assert conn.info.transaction_status != TransactionStatus.INERROR

    assert pool.conn.rolled_back == 1 and pool.checkouts == 1
