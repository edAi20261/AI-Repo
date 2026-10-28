"""Where a conversation is kept between turns.

Rows live in ``ai.user_memory`` beside the other per-session facts, one kind per
record: each turn, the running summary of turns folded out of the window, and
the session's standing state. No new table: the columns already describe a fact
about a user in a conversation, with an expiry.

A folded turn changes its ``kind`` rather than being deleted, so the row that
holds what was actually said outlives its trip through the summary. The kind is
the watermark: what is a ``conversation_turn`` is quoted, what is a
``conversation_folded_turn`` is already summarised, and nothing has to remember
which is which. Folded turns are dropped only when there are more of them than
a rewrite could read.

Nothing here may cost a student their answer. Reads and writes have their own
short timeouts and a circuit breaker, and a store that cannot be reached reads
as a conversation that has just begun.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Optional

from pydantic import ValidationError

from shared.postgres import connect

from .records import Conversation, SessionState, Turn

logger = logging.getLogger(__name__)

TURN = "conversation_turn"
FOLDED = "conversation_folded_turn"
SUMMARY = "conversation_summary"
STATE = "conversation_state"
KINDS = (TURN, FOLDED, SUMMARY, STATE)
NOTE = "conversation_note"

REMEMBERING_TYPES = frozenset({"student", "teacher"})

DEFAULT_TTL_HOURS = 720
# A backstop on one conversation's rows, not a policy: the fold prunes what it
# cannot reread, so reaching this means something else is wrong. Newest first,
# so what this drops is the oldest folded turn — the one the fold would have
# dropped next anyway. The standing state and the summary are rewritten every
# time they change, which puts them among the newest rows and out of reach of it.
MAX_ROWS = 400
DEFAULT_CONNECT_TIMEOUT = 2
DEFAULT_STATEMENT_TIMEOUT_MS = 500
DEFAULT_FAILURE_THRESHOLD = 3
DEFAULT_BREAKER_SECONDS = 30

_LOAD = (
    "SELECT memory_id::text, kind, fact FROM ("
    "  SELECT memory_id, kind, fact, created_at FROM ai.user_memory "
    "  WHERE user_id = %s::uuid AND source_conversation_id = %s::uuid "
    "    AND kind = ANY(%s) AND (expires_at IS NULL OR expires_at > now()) "
    "  ORDER BY created_at DESC, memory_id DESC "
    "  LIMIT %s"
    ") newest ORDER BY created_at, memory_id"
)

_INSERT = (
    "INSERT INTO ai.user_memory "
    "  (user_id, user_type, fact, kind, source_conversation_id, expires_at) "
    "VALUES (%s::uuid, %s, %s, %s, %s::uuid, now() + make_interval(hours => %s))"
)

# Restoring writes several turns in one transaction, where now() is the same
# instant for every row — and a load orders by created_at then memory_id, which
# is a random uuid. Without an explicit stamp per row the recovered
# conversation would come back shuffled. Each is placed a second before the
# one after it, so they sit in order and just behind the turn being answered.
_INSERT_AT = (
    "INSERT INTO ai.user_memory "
    "  (user_id, user_type, fact, kind, source_conversation_id, created_at, expires_at) "
    "VALUES (%s::uuid, %s, %s, %s, %s::uuid, now() - make_interval(secs => %s), "
    "        now() + make_interval(hours => %s))"
)

_DELETE_KIND = (
    "DELETE FROM ai.user_memory "
    "WHERE user_id = %s::uuid AND source_conversation_id = %s::uuid AND kind = %s"
)

_DELETE_IDS = (
    "DELETE FROM ai.user_memory "
    "WHERE user_id = %s::uuid AND source_conversation_id = %s::uuid "
    "  AND memory_id = ANY(%s::uuid[])"
)

_RELABEL_IDS = (
    "UPDATE ai.user_memory SET kind = %s "
    "WHERE user_id = %s::uuid AND source_conversation_id = %s::uuid "
    "  AND memory_id = ANY(%s::uuid[])"
)


class ConversationStore(ABC):
    """The operations every backing store offers."""

    @abstractmethod
    def load(self, user_id: Optional[str], session_id: Optional[str]) -> Conversation:
        ...

    @abstractmethod
    def append(self, user_id, user_type, session_id, turn: Turn) -> bool:
        ...

    @abstractmethod
    def save_state(self, user_id, user_type, session_id, state: SessionState) -> bool:
        ...

    @abstractmethod
    def restore(self, user_id, user_type, session_id, turns: list[Turn],
                state: SessionState) -> bool:
        """Put a conversation back that was recovered from elsewhere.

        Written once, when nothing is remembered and a transcript survives, so
        that every turn after it behaves like any other rather than recovering
        again.
        """

    @abstractmethod
    def fold(self, user_id, user_type, session_id, *, summary: str, note: str,
             folded: list[Turn], discard: list[Turn]) -> bool:
        """Record the rewritten summary and mark ``folded`` as covered by it.

        ``note`` is the same fold put for the student, who is shown what is
        remembered; it is never read back into a prompt.

        ``discard`` is the part of the folded past a later rewrite could no
        longer afford to read, and is the only thing here that destroys a turn.
        """

    @staticmethod
    def remembers(user_id, user_type, session_id) -> bool:
        return bool(user_id and session_id and user_type in REMEMBERING_TYPES)

    @classmethod
    def from_config(cls, config) -> ConversationStore:
        settings = config.get("agent.conversation") or {}
        if settings.get("store", "postgres") == "memory":
            return InMemoryConversationStore()
        return PostgresConversationStore(config, settings)


class InMemoryConversationStore(ConversationStore):
    """A process-local store for tests, evaluation runs and a service with no database."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
        self._next = 0

    def load(self, user_id, session_id) -> Conversation:
        if not (user_id and session_id):
            return Conversation()
        return _assemble(self._rows.get((user_id, session_id), []))

    def append(self, user_id, user_type, session_id, turn) -> bool:
        if not self.remembers(user_id, user_type, session_id):
            return False
        self._add(user_id, session_id, TURN, turn.model_dump_json(exclude={"id"}))
        return True

    def save_state(self, user_id, user_type, session_id, state) -> bool:
        if not self.remembers(user_id, user_type, session_id):
            return False
        self._drop_kind(user_id, session_id, STATE)
        self._add(user_id, session_id, STATE, state.model_dump_json())
        return True

    def restore(self, user_id, user_type, session_id, turns, state) -> bool:
        if not (self.remembers(user_id, user_type, session_id) and turns):
            return False
        for turn in turns:
            self._add(user_id, session_id, TURN, turn.model_dump_json(exclude={"id"}))
        self.save_state(user_id, user_type, session_id, state)
        return True

    def fold(self, user_id, user_type, session_id, *, summary, note, folded, discard) -> bool:
        if not self.remembers(user_id, user_type, session_id):
            return False
        covered = {turn.id for turn in folded}
        gone = {turn.id for turn in discard}
        rows = self._rows[(user_id, session_id)]
        rows[:] = [(row[0], FOLDED if row[0] in covered else row[1], row[2])
                   for row in rows if row[0] not in gone and row[1] not in (SUMMARY, NOTE)]
        self._add(user_id, session_id, SUMMARY, summary)
        self._add(user_id, session_id, NOTE, note)
        return True

    def _add(self, user_id, session_id, kind, fact) -> None:
        self._next += 1
        self._rows[(user_id, session_id)].append((str(self._next), kind, fact))

    def _drop_kind(self, user_id, session_id, kind) -> None:
        rows = self._rows[(user_id, session_id)]
        rows[:] = [row for row in rows if row[1] != kind]


class PostgresConversationStore(ConversationStore):
    """The conversation, kept in ``ai.user_memory``."""

    def __init__(self, config, settings: dict[str, Any]) -> None:
        self._config = config
        self._ttl_hours = settings.get("ttl_hours", DEFAULT_TTL_HOURS)
        self._connect_timeout = settings.get(
            "connect_timeout_seconds", DEFAULT_CONNECT_TIMEOUT)
        self._statement_timeout_ms = settings.get(
            "statement_timeout_ms", DEFAULT_STATEMENT_TIMEOUT_MS)
        self._failure_threshold = settings.get(
            "failure_threshold", DEFAULT_FAILURE_THRESHOLD)
        self._breaker_seconds = settings.get("breaker_seconds", DEFAULT_BREAKER_SECONDS)
        self._failures = 0
        self._closed_until = 0.0

    def load(self, user_id, session_id) -> Conversation:
        if not (user_id and session_id):
            return Conversation()
        rows = self._run([(_LOAD, (user_id, session_id, list(KINDS), MAX_ROWS))],
                         fetch=True, label="load")
        return _assemble(rows or [])

    def append(self, user_id, user_type, session_id, turn) -> bool:
        if not self.remembers(user_id, user_type, session_id):
            return False
        return bool(self._run([(_INSERT, (
            user_id, user_type, turn.model_dump_json(exclude={"id"}), TURN,
            session_id, self._ttl_hours))], label="append"))

    def save_state(self, user_id, user_type, session_id, state) -> bool:
        if not self.remembers(user_id, user_type, session_id):
            return False
        return bool(self._run([
            (_DELETE_KIND, (user_id, session_id, STATE)),
            (_INSERT, (user_id, user_type, state.model_dump_json(), STATE,
                       session_id, self._ttl_hours)),
        ], label="state"))

    def restore(self, user_id, user_type, session_id, turns, state) -> bool:
        if not (self.remembers(user_id, user_type, session_id) and turns):
            return False
        statements = [
            (_INSERT_AT, (user_id, user_type, turn.model_dump_json(exclude={"id"}),
                          TURN, session_id, len(turns) - index, self._ttl_hours))
            for index, turn in enumerate(turns)
        ]
        statements += [
            (_DELETE_KIND, (user_id, session_id, STATE)),
            (_INSERT, (user_id, user_type, state.model_dump_json(), STATE,
                       session_id, self._ttl_hours)),
        ]
        return bool(self._run(statements, label="restore"))

    def fold(self, user_id, user_type, session_id, *, summary, note, folded, discard) -> bool:
        if not self.remembers(user_id, user_type, session_id):
            return False
        covered = [turn.id for turn in folded if turn.id]
        gone = [turn.id for turn in discard if turn.id]
        statements = [(_RELABEL_IDS, (FOLDED, user_id, session_id, covered))]
        if gone:
            statements.append((_DELETE_IDS, (user_id, session_id, gone)))
        statements += [
            (_DELETE_KIND, (user_id, session_id, SUMMARY)),
            (_INSERT, (user_id, user_type, summary, SUMMARY, session_id,
                       self._ttl_hours)),
            (_DELETE_KIND, (user_id, session_id, NOTE)),
            (_INSERT, (user_id, user_type, note, NOTE, session_id, self._ttl_hours)),
        ]
        return bool(self._run(statements, label="fold"))

    def _run(self, statements: list, *, fetch: bool = False, label: str = ""):
        if time.monotonic() < self._closed_until:
            return None
        try:
            with connect(
                self._config,
                connect_timeout=self._connect_timeout,
                statement_timeout_ms=self._statement_timeout_ms,
            ) as conn:
                rows = None
                with conn.cursor() as cur:
                    for sql, params in statements:
                        cur.execute(sql, params)
                    if fetch:
                        rows = cur.fetchall()
                conn.commit()
        except Exception as error:  # noqa: BLE001
            self._record_failure(label, error)
            return None
        self._failures = 0
        return rows if fetch else True

    def _record_failure(self, label: str, error: BaseException) -> None:
        self._failures += 1
        logger.warning("conversation store %s failed", label or "call", exc_info=error)
        if self._failures >= self._failure_threshold:
            self._closed_until = time.monotonic() + self._breaker_seconds
            self._failures = 0
            logger.warning("conversation store paused for %ss", self._breaker_seconds)


def _assemble(rows) -> Conversation:
    conversation = Conversation()
    for memory_id, kind, fact in rows:
        try:
            if kind in (TURN, FOLDED):
                turn = Turn.model_validate_json(fact).model_copy(update={"id": memory_id})
                (conversation.turns if kind == TURN else conversation.folded).append(turn)
            elif kind == SUMMARY:
                conversation.summary = fact
            elif kind == STATE:
                conversation.state = SessionState.model_validate_json(fact)
        except ValidationError:
            logger.warning("a stored %s no longer fits its shape; skipping it", kind)
    return conversation
