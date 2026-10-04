"""The teacher's voice on a student's turn.

Read once per teacher and kept briefly, like subject facts: a persona changes
when the teacher approves a new one, not between the nodes of one turn. What a
prompt receives is one rendered block — the card, and the few passages of the
teacher's own words nearest to what this student asked — or nothing at all.

The block ranks below everything else in the prompt it joins. It says how to
speak; the lesson says what is true, and the language resolver says which
language to speak in.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Any, Optional

from shared.config import Config
from shared.embeddings import get_embedder

from ..models import PersonaCard, PersonaVoice, Style
from ..prompts import PERSONA_EXAMPLES_PROMPT, PERSONA_VOICE_PROMPT
from .models import Exemplar
from .store import PersonaStore

logger = logging.getLogger(__name__)

NO_VOICE = ""



class PersonaDirectory:
    """Personas and their examples, cached for ``ttl_seconds``."""

    def __init__(self, store: Optional[PersonaStore], *, ttl_seconds: float) -> None:
        self._store = store
        self._ttl = ttl_seconds
        self._voices: dict[tuple[str, Optional[int]], tuple[float, Optional[PersonaVoice]]] = {}
        self._examples: dict[str, tuple[float, list[Exemplar]]] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, config: Config) -> PersonaDirectory:
        store = PersonaStore.from_config(config) if config.get("persona.enabled") else None
        return cls(store, ttl_seconds=config.get("persona.cache_ttl_seconds"))

    @property
    def store(self) -> Optional[PersonaStore]:
        return self._store

    def voice(self, teacher_id: str, version: Optional[int] = None) -> Optional[PersonaVoice]:
        """The teacher's active persona, or ``version`` when a preview names one."""
        if self._store is None:
            logger.warning("a turn named a teacher but personas are disabled",
                           extra={"teacher_id": teacher_id})
            return None
        key = (teacher_id, version)
        hit = self._fresh(self._voices, key)
        if hit is not None:
            return hit[1]
        try:
            persona = (self._store.get(teacher_id, version) if version is not None
                       else self._store.active(teacher_id))
        except Exception:
            logger.warning("the teacher's persona could not be read; answering without it",
                           extra={"teacher_id": teacher_id, "version": version}, exc_info=True)
            return None
        voice = persona.voice() if persona else None
        if voice is None:
            logger.info("no persona for this teacher", extra={
                "teacher_id": teacher_id, "version": version})
        with self._lock:
            self._voices[key] = (time.monotonic(), voice)
        return voice

    def examples(self, voice: PersonaVoice) -> list[Exemplar]:
        hit = self._fresh(self._examples, voice.persona_id)
        if hit is not None:
            return hit[1]
        try:
            examples = self._store.exemplars(voice.persona_id) if self._store else []
        except Exception:
            logger.warning("the teacher's examples could not be read; using the card alone",
                           extra={"persona_id": voice.persona_id}, exc_info=True)
            return []
        with self._lock:
            self._examples[voice.persona_id] = (time.monotonic(), examples)
        return examples

    def forget(self, teacher_id: str) -> None:
        """Drop what this process holds for a teacher whose persona just changed."""
        with self._lock:
            for key in [k for k in self._voices if k[0] == teacher_id]:
                del self._voices[key]
            self._examples.clear()

    def _fresh(self, cache: dict, key: Any) -> Optional[tuple]:
        with self._lock:
            hit = cache.get(key)
        if hit and time.monotonic() - hit[0] < self._ttl:
            return hit
        return None


class Voices:
    """Renders the voice block a prompt receives."""

    def __init__(self, directory: PersonaDirectory, *, per_turn: int,
                 embedder: Callable[[], Any]) -> None:
        self._directory = directory
        self._per_turn = per_turn
        self._embedder = embedder

    @classmethod
    def from_config(cls, config: Config, directory: PersonaDirectory) -> Voices:
        return cls(directory, per_turn=config.get("persona.examples_per_turn"),
                   embedder=lambda: get_embedder(config))

    @property
    def directory(self) -> PersonaDirectory:
        return self._directory

    def render(self, style: Optional[Style], query: str = "", *,
               with_examples: bool = True) -> str:
        voice = style.persona if style else None
        if voice is None:
            return NO_VOICE
        examples = self._nearest(voice, query) if with_examples else []
        return render_voice(voice.teacher_name, voice.card, examples)

    def _nearest(self, voice: PersonaVoice, query: str) -> list[Exemplar]:
        examples = self._directory.examples(voice)
        if len(examples) <= self._per_turn or not query.strip():
            return examples[: self._per_turn]
        try:
            wanted = self._embedder().embed_query(query)
        except Exception:
            logger.warning("the query could not be embedded; using the first examples",
                           exc_info=True)
            return examples[: self._per_turn]
        ranked = sorted(examples, key=lambda e: _dot(wanted, e.embedding), reverse=True)
        return ranked[: self._per_turn]


def render_voice(teacher_name: str, card: PersonaCard, examples: list[Exemplar]) -> str:
    """The voice block a prompt receives: the approved card and the chosen examples,
    as data inside the instructions that frame them."""
    teacher = " ".join(teacher_name.split())
    shown = PERSONA_EXAMPLES_PROMPT.render(
        teacher=teacher,
        examples=_as_json([{"student_said": e.prompt, "teacher_said": e.text}
                           for e in examples])) if examples else ""
    return PERSONA_VOICE_PROMPT.render(
        teacher=teacher,
        card=_as_json({field: value for field, value in card.model_dump().items() if value}),
        examples=shown)


def _as_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _dot(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return float("-inf")
    return sum(x * y for x, y in zip(a, b))
