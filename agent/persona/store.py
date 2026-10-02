"""Where personas are kept, and the rules for moving one between states.

A teacher can hold many versions; at most one is ``active``, and only the active
one reaches students. A new build is always a ``draft``: nothing a model wrote
about a teacher speaks to their class until the teacher has read it and said so.

The tables are declared by the backend's Prisma schema
(docs/persona-schema-requests.md); this service never runs DDL.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from shared.config import Config
from shared.errors import ConflictError, NotFoundError
from shared.postgres import connect

from ..models import PersonaCard
from .models import Exemplar, Persona, PersonaStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NewPersona:
    teacher_id: str
    teacher_name: str
    card: PersonaCard
    consented_at: datetime
    sources: dict[str, Any] = field(default_factory=dict)


class PersonaStore(ABC):
    @abstractmethod
    def create(self, new: NewPersona, exemplars: list[Exemplar]) -> Persona:
        """Store a new draft as the teacher's next version."""

    @abstractmethod
    def get(self, teacher_id: str, version: int) -> Optional[Persona]:
        ...

    @abstractmethod
    def active(self, teacher_id: str) -> Optional[Persona]:
        ...

    @abstractmethod
    def versions(self, teacher_id: str) -> list[Persona]:
        """Every version, newest first."""

    @abstractmethod
    def exemplars(self, persona_id: str) -> list[Exemplar]:
        ...

    @abstractmethod
    def update_card(self, teacher_id: str, version: int, card: PersonaCard) -> Persona:
        ...

    @abstractmethod
    def activate(self, teacher_id: str, version: int) -> Persona:
        ...

    @abstractmethod
    def deactivate(self, teacher_id: str) -> Optional[Persona]:
        """Take the active version off students. Returns it, or None if none was."""

    @abstractmethod
    def delete(self, teacher_id: str) -> int:
        """Destroy every version and its examples. Returns how many versions went."""

    @classmethod
    def from_config(cls, config: Config) -> PersonaStore:
        kind = config.get("persona.store")
        if kind == "memory":
            return InMemoryPersonaStore()
        if kind == "postgres":
            return PostgresPersonaStore(config)
        raise ValueError(f"persona.store must be 'postgres' or 'memory', not {kind!r}")


def _missing(teacher_id: str, version: int) -> NotFoundError:
    return NotFoundError("This teacher has no persona with that version.",
                         details={"teacher_id": teacher_id, "version": version})


def _live(teacher_id: str, version: int) -> ConflictError:
    return ConflictError(
        "The active persona cannot be edited while students are using it. "
        "Deactivate it first, or edit a draft.",
        details={"teacher_id": teacher_id, "version": version})


class InMemoryPersonaStore(PersonaStore):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._personas: dict[str, Persona] = {}
        self._exemplars: dict[str, list[Exemplar]] = {}

    def create(self, new: NewPersona, exemplars: list[Exemplar]) -> Persona:
        with self._lock:
            version = 1 + max((p.version for p in self._personas.values()
                               if p.teacher_id == new.teacher_id), default=0)
            now = datetime.now(timezone.utc)
            persona = Persona(
                persona_id=str(uuid.uuid4()), teacher_id=new.teacher_id, version=version,
                status=PersonaStatus.DRAFT, teacher_name=new.teacher_name, card=new.card,
                exemplar_count=len(exemplars), sources=new.sources,
                consented_at=new.consented_at, created_at=now, updated_at=now)
            self._personas[persona.persona_id] = persona
            self._exemplars[persona.persona_id] = list(exemplars)
            return persona

    def get(self, teacher_id: str, version: int) -> Optional[Persona]:
        with self._lock:
            return self._find(teacher_id, version)

    def active(self, teacher_id: str) -> Optional[Persona]:
        with self._lock:
            return next((p for p in self._personas.values() if p.teacher_id == teacher_id
                         and p.status is PersonaStatus.ACTIVE), None)

    def versions(self, teacher_id: str) -> list[Persona]:
        with self._lock:
            return sorted((p for p in self._personas.values() if p.teacher_id == teacher_id),
                          key=lambda p: p.version, reverse=True)

    def exemplars(self, persona_id: str) -> list[Exemplar]:
        with self._lock:
            return list(self._exemplars.get(persona_id, []))

    def update_card(self, teacher_id: str, version: int, card: PersonaCard) -> Persona:
        with self._lock:
            persona = self._require(teacher_id, version)
            if persona.status is PersonaStatus.ACTIVE:
                raise _live(teacher_id, version)
            return self._put(persona, card=card)

    def activate(self, teacher_id: str, version: int) -> Persona:
        with self._lock:
            chosen = self._require(teacher_id, version)
            for persona in list(self._personas.values()):
                if (persona.teacher_id == teacher_id and persona.status is PersonaStatus.ACTIVE
                        and persona.persona_id != chosen.persona_id):
                    self._put(persona, status=PersonaStatus.INACTIVE)
            now = datetime.now(timezone.utc)
            return self._put(chosen, status=PersonaStatus.ACTIVE, activated_at=now)

    def deactivate(self, teacher_id: str) -> Optional[Persona]:
        with self._lock:
            live = next((p for p in self._personas.values() if p.teacher_id == teacher_id
                         and p.status is PersonaStatus.ACTIVE), None)
            return self._put(live, status=PersonaStatus.INACTIVE) if live else None

    def delete(self, teacher_id: str) -> int:
        with self._lock:
            doomed = [pid for pid, p in self._personas.items() if p.teacher_id == teacher_id]
            for pid in doomed:
                del self._personas[pid]
                self._exemplars.pop(pid, None)
            return len(doomed)

    def _find(self, teacher_id: str, version: int) -> Optional[Persona]:
        return next((p for p in self._personas.values()
                     if p.teacher_id == teacher_id and p.version == version), None)

    def _require(self, teacher_id: str, version: int) -> Persona:
        persona = self._find(teacher_id, version)
        if persona is None:
            raise _missing(teacher_id, version)
        return persona

    def _put(self, persona: Persona, **changes: Any) -> Persona:
        updated = persona.model_copy(update={**changes,
                                             "updated_at": datetime.now(timezone.utc)})
        self._personas[updated.persona_id] = updated
        return updated


_COLUMNS = ("persona_id::text, teacher_id::text, version, status, teacher_name, card, "
            "exemplar_count, sources, consented_at, created_at, updated_at, activated_at")

_NEXT_VERSION = ("SELECT COALESCE(max(version), 0) + 1 FROM ai.teacher_personas "
                 "WHERE teacher_id = %s::uuid")
_INSERT = (
    "INSERT INTO ai.teacher_personas "
    "  (teacher_id, version, status, teacher_name, card, exemplar_count, sources, "
    "   consented_at) "
    "VALUES (%s::uuid, %s, 'draft', %s, %s::jsonb, %s, %s::jsonb, %s) "
    f"RETURNING {_COLUMNS}")
_INSERT_EXEMPLAR = (
    "INSERT INTO ai.teacher_persona_exemplars "
    "  (persona_id, ordinal, text, prompt, move, source_id, embedding) "
    "VALUES (%s::uuid, %s, %s, %s, %s, %s, %s)")
_BY_VERSION = (f"SELECT {_COLUMNS} FROM ai.teacher_personas "
               "WHERE teacher_id = %s::uuid AND version = %s")
_ACTIVE = (f"SELECT {_COLUMNS} FROM ai.teacher_personas "
           "WHERE teacher_id = %s::uuid AND status = 'active'")
_VERSIONS = (f"SELECT {_COLUMNS} FROM ai.teacher_personas "
             "WHERE teacher_id = %s::uuid ORDER BY version DESC")
_EXEMPLARS = ("SELECT text, prompt, move, source_id, embedding "
              "FROM ai.teacher_persona_exemplars WHERE persona_id = %s::uuid ORDER BY ordinal")
_UPDATE_CARD = ("UPDATE ai.teacher_personas SET card = %s::jsonb, updated_at = now() "
                "WHERE teacher_id = %s::uuid AND version = %s AND status <> 'active' "
                f"RETURNING {_COLUMNS}")
_RETIRE = ("UPDATE ai.teacher_personas SET status = 'inactive', updated_at = now() "
           "WHERE teacher_id = %s::uuid AND status = 'active' "
           f"RETURNING {_COLUMNS}")
_ACTIVATE = ("UPDATE ai.teacher_personas "
             "SET status = 'active', activated_at = now(), updated_at = now() "
             "WHERE teacher_id = %s::uuid AND version = %s "
             f"RETURNING {_COLUMNS}")
_DELETE = "DELETE FROM ai.teacher_personas WHERE teacher_id = %s::uuid"


class PostgresPersonaStore(PersonaStore):
    def __init__(self, config: Config) -> None:
        self._config = config

    def create(self, new: NewPersona, exemplars: list[Exemplar]) -> Persona:
        from psycopg.errors import UniqueViolation

        try:
            with connect(self._config) as conn:
                with conn.cursor() as cur:
                    cur.execute(_NEXT_VERSION, (new.teacher_id,))
                    version = cur.fetchone()[0]
                    cur.execute(_INSERT, (
                        new.teacher_id, version, new.teacher_name,
                        new.card.model_dump_json(), len(exemplars),
                        json.dumps(new.sources, ensure_ascii=False), new.consented_at))
                    persona = _persona(cur.fetchone())
                    cur.executemany(_INSERT_EXEMPLAR, [
                        (persona.persona_id, ordinal, e.text, e.prompt, e.move, e.source_id,
                         e.embedding)
                        for ordinal, e in enumerate(exemplars)])
                conn.commit()
        except UniqueViolation as error:
            raise ConflictError("Another persona for this teacher was stored at the same "
                                "moment. Build it again.",
                                details={"teacher_id": new.teacher_id}) from error
        return persona

    def get(self, teacher_id: str, version: int) -> Optional[Persona]:
        return self._one(_BY_VERSION, (teacher_id, version))

    def active(self, teacher_id: str) -> Optional[Persona]:
        return self._one(_ACTIVE, (teacher_id,))

    def versions(self, teacher_id: str) -> list[Persona]:
        with connect(self._config) as conn, conn.cursor() as cur:
            cur.execute(_VERSIONS, (teacher_id,))
            return [_persona(row) for row in cur.fetchall()]

    def exemplars(self, persona_id: str) -> list[Exemplar]:
        with connect(self._config) as conn, conn.cursor() as cur:
            cur.execute(_EXEMPLARS, (persona_id,))
            return [Exemplar(text=text, prompt=prompt, move=move, source_id=source_id,
                             embedding=list(embedding or []))
                    for text, prompt, move, source_id, embedding in cur.fetchall()]

    def update_card(self, teacher_id: str, version: int, card: PersonaCard) -> Persona:
        with connect(self._config) as conn:
            with conn.cursor() as cur:
                cur.execute(_UPDATE_CARD, (card.model_dump_json(), teacher_id, version))
                row = cur.fetchone()
            conn.commit()
        if row is not None:
            return _persona(row)
        existing = self.get(teacher_id, version)
        raise _live(teacher_id, version) if existing else _missing(teacher_id, version)

    def activate(self, teacher_id: str, version: int) -> Persona:
        with connect(self._config) as conn:
            with conn.cursor() as cur:
                cur.execute(_BY_VERSION, (teacher_id, version))
                if cur.fetchone() is None:
                    raise _missing(teacher_id, version)
                cur.execute(_RETIRE, (teacher_id,))
                cur.execute(_ACTIVATE, (teacher_id, version))
                row = cur.fetchone()
            conn.commit()
        return _persona(row)

    def deactivate(self, teacher_id: str) -> Optional[Persona]:
        with connect(self._config) as conn:
            with conn.cursor() as cur:
                cur.execute(_RETIRE, (teacher_id,))
                row = cur.fetchone()
            conn.commit()
        return _persona(row) if row else None

    def delete(self, teacher_id: str) -> int:
        with connect(self._config) as conn:
            with conn.cursor() as cur:
                cur.execute(_DELETE, (teacher_id,))
                deleted = cur.rowcount
            conn.commit()
        return max(0, deleted)

    def _one(self, sql: str, params: tuple) -> Optional[Persona]:
        with connect(self._config) as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        return _persona(row) if row else None


def _persona(row: tuple) -> Persona:
    (persona_id, teacher_id, version, status, teacher_name, card, exemplar_count, sources,
     consented_at, created_at, updated_at, activated_at) = row
    return Persona(
        persona_id=persona_id, teacher_id=teacher_id, version=version,
        status=PersonaStatus(status), teacher_name=teacher_name,
        card=PersonaCard.model_validate(card), exemplar_count=exemplar_count,
        sources=sources or {}, consented_at=consented_at, created_at=created_at,
        updated_at=updated_at, activated_at=activated_at)
