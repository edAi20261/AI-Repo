"""AgentRuntime.stream_chat contract — the SSE turn behind ``POST /ai/chat/stream``.

What broke: ``stream_chat`` opened its capacity slot and deadline with
``async with self._limiter, anyio.fail_after(...)``. ``anyio.fail_after`` is a
*sync* context manager (no ``__aenter__``), so entering it under ``async with``
raised ``TypeError: '_GeneratorContextManager' object does not support the
asynchronous context manager protocol`` before a single stage streamed — every
``/ai/chat/stream`` request 500'd. There was no test over the streaming path, so
it shipped. These run the real ``stream_chat`` against a fake graph.
"""

from __future__ import annotations

from typing import Any

import anyio

from shared.config import get_config
from system.billing import Billing
from system.runtime import AgentRuntime, Metered


class _FakeGraph:
    """A graph whose ``astream`` yields LangGraph-style ``{stage: delta}`` updates."""

    def __init__(self, updates: list[dict[str, Any]]) -> None:
        self._updates = updates

    async def astream(self, _payload, stream_mode=None):
        for update in self._updates:
            yield update


def _runtime(updates: list[dict[str, Any]]) -> AgentRuntime:
    return AgentRuntime(
        get_config(),
        _FakeGraph(updates),
        visual_graph=None,
        billing=Billing(None),
        max_concurrency=2,
        timeout=30.0,
    )


def _drain(runtime: AgentRuntime, payload: dict[str, Any]) -> list[tuple[str, Any]]:
    collected: list[tuple[str, Any]] = []

    async def run() -> None:
        async for item in await runtime.stream_chat(payload):
            collected.append(item)

    anyio.run(run)
    return collected


def test_a_student_stream_reaches_every_stage_and_a_final_state():
    """Streaming a turn yields one event per graph stage and a terminal
    ``__final__`` snapshot — the behaviour the SSE endpoint depends on. Before
    the fix this raised a TypeError from ``async with anyio.fail_after(...)``
    instead of yielding anything."""
    runtime = _runtime(
        [
            {"classify": {"language": "en"}},
            {"compose": {"response": "Fuel, heat, oxygen [1]."}},
        ]
    )

    events = _drain(runtime, {"user_id": "u1", "user_type": "student", "message": "hi"})

    stages = [stage for stage, _ in events]
    assert stages == ["classify", "compose", "__final__"]

    final_stage, final = events[-1]
    assert final_stage == "__final__"
    # __final__ carries the accumulated state folded from every stage's delta,
    # with what the turn cost beside it.
    assert isinstance(final, Metered)
    assert final.result["language"] == "en"
    assert final.result["response"] == "Fuel, heat, oxygen [1]."
