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
from agent.graph.grade import run_grade, run_reviewed
from agent.graph.report import run_class_report, run_report, stored_scopes
from agent.graph.subgraphs import build_visual_sg
from agent.homework import HomeworkCapture, MarkedHomework, capture, mark
from agent.models import PersonaCard
from agent.persona.build import BuildRequest, build_persona
from agent.persona.models import Exemplar, Persona
from agent.persona.store import PersonaStore
from agent.persona.voice import PersonaDirectory
from agent.profile import forget, held
from agent.tagging import suggest_topics
from agent.voice.service import Heard, ensure_speaking, hear, speak
from agent.tools import attach_report_visual, get_latest_report
from shared.config import Config
from shared.metering import Meter, start_meter

from .billing import Bill, Billing
from .errors import NotFoundError, ServiceUnavailableError, UpstreamTimeoutError

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


def _unmarked(result: MarkedHomework) -> bool:
    return result.unreadable or result.degraded


def _unheard(result: Heard) -> bool:
    return not result.heard


@dataclass(frozen=True)
class PersonaView:
    persona: Persona
    exemplars: list[Exemplar]


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
        max_reads: Optional[int] = None,
        personas: Optional[PersonaDirectory] = None,
    ) -> None:
        self._config = config
        self._graph = graph
        self._visual_graph = visual_graph
        self._billing = billing
        self._personas = personas
        self._limiter = CapacityLimiter(max_concurrency)
        self._reads = CapacityLimiter(max_reads or max_concurrency)
        self._timeout = timeout

    @classmethod
    def build(
        cls,
        config: Config | None = None,
        *,
        billing: Billing,
        max_concurrency: int,
        max_reads: int,
        timeout: float,
    ) -> AgentRuntime:
        config = config or get_config()
        personas = PersonaDirectory.from_config(config)
        graph = build_graph(config, personas=personas)
        visual_graph = build_visual_sg(config=config)
        logger.info(
            "runtime graphs compiled",
            extra={"max_concurrency": max_concurrency, "max_reads": max_reads,
                   "timeout": timeout},
        )
        return cls(
            config, graph, visual_graph, billing=billing,
            max_concurrency=max_concurrency, max_reads=max_reads, timeout=timeout,
            personas=personas,
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

    async def record_reviewed(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._offload(run_reviewed, payload, config=self._config)

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

    async def mark_homework(self, content: bytes, payload: dict[str, Any], *,
                            billed_user_id: str) -> Metered:
        return await self._billed(billed_user_id, Action.HOMEWORK_MARK,
                                  partial(mark, content, config=self._config, **payload),
                                  failed=_unmarked)

    async def transcribe(self, audio: bytes, content_type: str, *,
                         language: Optional[str], billed_user_id: str) -> Metered:
        return await self._billed(billed_user_id, Action.VOICE_TRANSCRIBE,
                                  partial(hear, audio, content_type, config=self._config,
                                          language=language),
                                  failed=_unheard)

    async def speak(self, text: str, language: str, *, billed_user_id: str) -> Metered:
        ensure_speaking(self._config)
        return await self._billed(billed_user_id, Action.VOICE_SPEAK,
                                  partial(speak, text, language, config=self._config))

    async def class_analytics(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._read(class_analytics, payload["user_ids"],
                                subject_ids=payload.get("subject_ids"), config=self._config)

    async def latest_report(self, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        if payload.get("user_id") and payload["scope"] not in stored_scopes(self._config):
            return None
        return await self._read(get_latest_report, payload["scope"],
                                user_id=payload.get("user_id"),
                                class_id=payload.get("class_id"),
                                subject_id=payload.get("subject_id"), config=self._config)

    async def attach_report_visual(self, report_id: str, visual: dict[str, Any]) -> bool:
        return await self._offload(attach_report_visual, report_id, visual,
                                   config=self._config)

    async def class_report(self, payload: dict[str, Any], *,
                           billed_user_id: str) -> Metered:
        return await self._billed(billed_user_id, Action.CLASS_REPORT,
                                  partial(run_class_report, payload, config=self._config),
                                  failed=_degraded)

    async def learner_summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._read(
            learner_summary, payload["user_id"], subject_ids=payload.get("subject_ids"),
            lecture_ids=payload.get("lecture_ids"), config=self._config)

    async def suggest_topics(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._read(
            suggest_topics, payload["questions"], payload["subject_id"],
            lecture_ids=payload.get("lecture_ids"), config=self._config)

    async def build_persona(self, request: BuildRequest, *,
                            billed_user_id: str) -> Metered:
        store = self._persona_store()
        built = await self._billed(billed_user_id, Action.PERSONA_BUILD,
                                   partial(build_persona, request, store=store,
                                           config=self._config),
                                   timeout=self._config.get("persona.build_timeout_seconds"))
        self._personas.forget(request.teacher_id)
        return built

    async def personas(self, teacher_id: str) -> list[Persona]:
        return await self._read(self._persona_store().versions, teacher_id)

    async def persona(self, teacher_id: str, version: int) -> PersonaView:
        store = self._persona_store()

        def read() -> PersonaView:
            persona = store.get(teacher_id, version)
            if persona is None:
                raise NotFoundError("This teacher has no persona with that version.",
                                    details={"teacher_id": teacher_id, "version": version})
            return PersonaView(persona, store.exemplars(persona.persona_id))

        return await self._read(read)

    async def memory(self, user_id: str) -> dict[str, Any]:
        return await self._read(held, user_id, config=self._config)

    async def forget_memory(self, user_id: str,
                            conversation_id: Optional[str] = None) -> dict[str, Any]:
        return await self._read(forget, user_id, conversation_id=conversation_id,
                                config=self._config)

    async def edit_persona_card(self, teacher_id: str, version: int,
                                card: PersonaCard) -> Persona:
        return await self._changing(teacher_id, self._persona_store().update_card,
                                    teacher_id, version, card)

    async def activate_persona(self, teacher_id: str, version: int) -> Persona:
        return await self._changing(teacher_id, self._persona_store().activate,
                                    teacher_id, version)

    async def deactivate_persona(self, teacher_id: str) -> Optional[Persona]:
        return await self._changing(teacher_id, self._persona_store().deactivate, teacher_id)

    async def delete_persona(self, teacher_id: str) -> int:
        return await self._changing(teacher_id, self._persona_store().delete, teacher_id)

    def _persona_store(self) -> PersonaStore:
        store = self._personas.store if self._personas else None
        if store is None:
            raise ServiceUnavailableError("Teacher personas are not enabled on this service.")
        return store

    async def _changing(self, teacher_id: str, change: Callable[..., T], *args) -> T:
        try:
            return await self._offload(change, *args)
        finally:
            self._personas.forget(teacher_id)

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
            yield "queued", None
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
                      units: int = 1, failed: Callable[[T], bool] = _never,
                      timeout: Optional[float] = None) -> Metered:
        async with self._billing.spend(user_id, action, units=units) as bill:
            meter = start_meter()
            result = await self._offload(work, timeout=timeout)
            if failed(result):
                bill.void()
            else:
                bill.settle(meter)
            return Metered(result, meter)

    async def _read(self, fn, *args, **kwargs) -> Any:
        return await self._offload(fn, *args, limiter=self._reads, **kwargs)

    async def _offload(self, fn, *args, timeout: Optional[float] = None,
                       limiter: Optional[CapacityLimiter] = None, **kwargs) -> Any:
        def call() -> Any:
            return fn(*args, **kwargs)

        limit = timeout or self._timeout
        try:
            with anyio.fail_after(limit):
                return await anyio.to_thread.run_sync(
                    call, abandon_on_cancel=True, limiter=limiter or self._limiter
                )
        except TimeoutError as exc:
            # A thread cannot be killed, but anyio gives its slot back when the
            # caller is cancelled: the abandoned work runs on outside the bound.
            logger.warning("turn exceeded %ss", limit)
            raise UpstreamTimeoutError("The request took too long to complete.") from exc
        except MemoryError as exc:
            raise ServiceUnavailableError("The service is out of capacity.") from exc
