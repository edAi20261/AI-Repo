"""Runtime agent facade.

The compiled LangGraph is built once per process and reused; compiling per
request would reload the embedder and every subgraph. Graph execution is
synchronous and CPU/IO heavy, so it is offloaded to a worker thread under a
capacity limiter — the event loop stays free to accept requests, and the number
of concurrent turns stays bounded.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, Optional

import anyio
from anyio import CapacityLimiter

from agent.config import get_config
from agent.graph import build_graph
from agent.graph.grade import run_grade
from agent.graph.report import run_report
from agent.graph.subgraphs import build_visual_sg
from shared.config import Config

from .errors import ServiceUnavailableError, UpstreamTimeoutError

logger = logging.getLogger(__name__)


class AgentRuntime:
    """Owns the compiled graphs and mediates every call into the agent."""

    def __init__(
        self,
        config: Config,
        graph: Any,
        visual_graph: Any,
        *,
        max_concurrency: int,
        timeout: float,
    ) -> None:
        self._config = config
        self._graph = graph
        self._visual_graph = visual_graph
        self._limiter = CapacityLimiter(max_concurrency)
        self._timeout = timeout

    @classmethod
    def build(
        cls,
        config: Config | None = None,
        *,
        max_concurrency: int = 8,
        timeout: float = 120.0,
    ) -> AgentRuntime:
        config = config or get_config()
        graph = build_graph(config)
        visual_graph = build_visual_sg(config=config)
        logger.info(
            "runtime graphs compiled",
            extra={"max_concurrency": max_concurrency, "timeout": timeout},
        )
        return cls(
            config, graph, visual_graph, max_concurrency=max_concurrency, timeout=timeout
        )

    @property
    def config(self) -> Config:
        return self._config

    @property
    def in_flight(self) -> int:
        return self._limiter.borrowed_tokens

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._offload(self._graph.invoke, payload)

    async def grade(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._offload(run_grade, payload, config=self._config)

    async def report(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._offload(run_report, payload, config=self._config)

    async def render(
        self,
        specs: list[dict[str, Any]],
        *,
        user_id: Optional[str] = None,
        subject_id: Optional[str] = None,
        language: str = "en",
    ) -> list[dict[str, Any]]:
        """Render visual specs that an entrypoint returned unrendered.

        The chat graph renders inline; ``run_report`` returns its trend-chart spec
        for the caller to render, which is what this does.
        """
        if not specs:
            return []
        result = await self._offload(
            self._visual_graph.invoke,
            {
                "specs": specs,
                "user_id": user_id,
                "subject_id": subject_id,
                "language": language,
                "content": {},
                "max_per_turn": len(specs),
            },
        )
        return result.get("visuals", []) or []

    async def stream_chat(self, payload: dict[str, Any]) -> AsyncIterator[tuple[str, Any]]:
        """Yield ``(stage, update)`` as the graph advances, then ``("__final__", state)``.

        Stage-level progress, not token-level: the agent's nodes call their models
        synchronously, so this reports which stage is running rather than partial text.
        """
        state: dict[str, Any] = {}
        # fail_after is a synchronous context manager used inside async code;
        # `async with` on it raises before the first node ever runs.
        try:
            async with self._limiter:
                with anyio.fail_after(self._timeout):
                    async for update in self._graph.astream(payload, stream_mode="updates"):
                        for stage, delta in update.items():
                            if isinstance(delta, dict):
                                state.update(delta)
                            yield stage, delta
        except TimeoutError as exc:
            logger.warning("streamed turn exceeded %ss", self._timeout)
            raise UpstreamTimeoutError("The request took too long to complete.") from exc
        yield "__final__", state

    async def _offload(self, fn, *args, **kwargs) -> dict[str, Any]:
        def call() -> dict[str, Any]:
            return fn(*args, **kwargs)

        try:
            with anyio.fail_after(self._timeout):
                return await anyio.to_thread.run_sync(
                    call, abandon_on_cancel=True, limiter=self._limiter
                )
        except TimeoutError as exc:
            # A thread cannot be killed: the work runs to completion and holds its slot.
            # The caller is released rather than left hanging.
            logger.warning("turn exceeded %ss", self._timeout)
            raise UpstreamTimeoutError("The request took too long to complete.") from exc
        except MemoryError as exc:
            raise ServiceUnavailableError("The service is out of capacity.") from exc
