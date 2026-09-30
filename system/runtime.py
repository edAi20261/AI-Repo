"""Runtime agent facade.

The compiled LangGraph is built once per process and reused; compiling per
request would reload the embedder and every subgraph. Graph execution is
synchronous and CPU/IO heavy, so it is offloaded to a worker thread under a
capacity limiter — the event loop stays free to accept requests, and the number
of concurrent turns stays bounded.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, TypeVar

import anyio
from anyio import CapacityLimiter

from agent.config import get_config
from agent.analytics import learner_summary
from agent.class_analytics import class_analytics
from agent.credits import Action
from agent.graph import build_graph
from agent.graph.grade import run_grade
from agent.graph.report import run_class_report, run_report
from agent.graph.subgraphs import build_visual_sg
from agent.homework import HomeworkCapture, capture
from agent.tagging import suggest_topics
from agent.tools import get_latest_report
from shared.config import Config
from shared.metering import Meter, start_meter

from .billing import Bill, Billing
from .errors import ServiceUnavailableError, UpstreamTimeoutError

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class Metered:
    result: Any
    meter: Meter


def _never(_result: Any) -> bool:
    return False


def _degraded(result: dict[str, Any]) -> bool:
    return bool(result.get("degraded"))


def _unreadable(result: HomeworkCapture) -> bool:
    return result.unreadable


class AgentRuntime:
    """Owns the compiled graphs and mediates every call into the agent."""

    def __init__(
        self,
        config: Config,
        graph: Any,
        visual_graph: Any,
        *,
        billing: Billing,
        max_concurrency: int,
        timeout: float,
    ) -> None:
        self._config = config
        self._graph = graph
        self._visual_graph = visual_graph
        self._billing = billing
        self._limiter = CapacityLimiter(max_concurrency)
        self._timeout = timeout

    @classmethod
    def build(
        cls,
        config: Config | None = None,
        *,
        billing: Billing,
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
            config, graph, visual_graph, billing=billing,
            max_concurrency=max_concurrency, timeout=timeout,
        )

    @property
    def config(self) -> Config:
        return self._config

    @property
    def in_flight(self) -> int:
        return self._limiter.borrowed_tokens

    async def chat(self, payload: dict[str, Any]) -> Metered:
        return await self._billed(payload["user_id"], Action.CHAT_MESSAGE,
                                  partial(self._graph.invoke, payload))

    async def grade(self, payload: dict[str, Any], *,
                    billed_user_id: str) -> Metered:
        return await self._billed(billed_user_id, Action.GRADE_ANSWERS,
                                  partial(run_grade, payload, config=self._config),
                                  units=len(payload["submissions"]), failed=_degraded)

    async def report(self, payload: dict[str, Any], *,
                     billed_user_id: str) -> Metered:
        return await self._billed(billed_user_id, Action.STUDENT_REPORT,
                                  partial(run_report, payload, config=self._config),
                                  failed=_degraded)

    async def capture_homework(self, content: bytes, *, language: Optional[str],
                               billed_user_id: str) -> Metered:
        return await self._billed(billed_user_id, Action.HOMEWORK_CAPTURE,
                                  partial(capture, content, config=self._config,
                                          language_hint=language),
                                  failed=_unreadable)

    async def class_analytics(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._offload(class_analytics, payload["user_ids"],
                                   subject_ids=payload.get("subject_ids"), config=self._config)

    async def latest_report(self, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        return await self._offload(get_latest_report, payload["scope"],
                                   user_id=payload.get("user_id"),
                                   class_id=payload.get("class_id"), config=self._config)

    async def class_report(self, payload: dict[str, Any], *,
                           billed_user_id: str) -> Metered:
        return await self._billed(billed_user_id, Action.CLASS_REPORT,
                                  partial(run_class_report, payload, config=self._config),
                                  failed=_degraded)

    async def learner_summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._offload(
            learner_summary, payload["user_id"], subject_ids=payload.get("subject_ids"),
            lecture_ids=payload.get("lecture_ids"), config=self._config)

    async def suggest_topics(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._offload(
            suggest_topics, payload["questions"], payload["subject_id"],
            lecture_ids=payload.get("lecture_ids"), config=self._config)

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
        """Charge the turn, then return an iterator of ``(stage, update)`` as the graph
        advances, ending with ``("__final__", Metered)``.

        Stage-level progress, not token-level: the agent's nodes call their models
        synchronously, so this reports which stage is running rather than partial text.
        """
        bill = await self._billing.open(payload["user_id"], Action.CHAT_MESSAGE)
        return self._stream(payload, bill)

    async def _stream(self, payload: dict[str, Any],
                      bill: Bill) -> AsyncIterator[tuple[str, Any]]:
        state: dict[str, Any] = {}
        async with self._billing.guard(bill):
            meter = start_meter()
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
            bill.settle(meter)
        yield "__final__", Metered(state, meter)

    async def _billed(self, user_id: str, action: Action, work: Callable[[], T], *,
                      units: int = 1, failed: Callable[[T], bool] = _never) -> Metered:
        async with self._billing.spend(user_id, action, units=units) as bill:
            meter = start_meter()
            result = await self._offload(work)
            if failed(result):
                bill.void()
            else:
                bill.settle(meter)
            return Metered(result, meter)

    async def _offload(self, fn, *args, **kwargs) -> Any:
        def call() -> Any:
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
