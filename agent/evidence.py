"""What an answer was built from, kept for the turn that follows it.

A student who says "explain that more simply" is asking about the passages the
last answer used. Searching the syllabus for their words finds something else —
measured on a real textbook, five unrelated pages — because those words name
nothing. The only right answer is the one already on the table, said
differently.

So a turn records the chunks behind its answer and the next turn reaches them
by pointer rather than by search. Evidence that does not outlive the turn is
the reason re-presentation cannot work at all; every broken grounded tutor
looks the same from the outside for the same reason.

Stored in ``ai.user_memory`` beside the session's language instruction rather
than in a table of its own: it is the same kind of fact with the same lifetime,
the same circuit breaker and the same rule — nothing here may cost a student
their answer. A read that fails is indistinguishable from a session with no
previous answer, which is the behaviour the service already has.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError

from shared.postgres import connect

logger = logging.getLogger(__name__)

KIND = "turn_evidence"

# ai.memory_user_type covers these two. A parent turn remembers nothing.
REMEMBERING_TYPES = frozenset({"student", "teacher"})

# A follow-up arrives in the same sitting or not at all.
DEFAULT_TTL_HOURS = 12
DEFAULT_CONNECT_TIMEOUT = 2
DEFAULT_STATEMENT_TIMEOUT_MS = 500
DEFAULT_FAILURE_THRESHOLD = 3
DEFAULT_BREAKER_SECONDS = 30

_SELECT = (
    "SELECT fact FROM ai.user_memory "
    "WHERE user_id = %s::uuid AND kind = %s AND source_conversation_id = %s::uuid "
    "  AND (expires_at IS NULL OR expires_at > now()) "
    "ORDER BY created_at DESC LIMIT 1"
)

_DELETE = (
    "DELETE FROM ai.user_memory "
    "WHERE user_id = %s::uuid AND kind = %s AND source_conversation_id = %s::uuid"
)

_INSERT = (
    "INSERT INTO ai.user_memory "
    "  (user_id, user_type, fact, kind, source_conversation_id, expires_at) "
    "VALUES (%s::uuid, %s, %s, %s, %s::uuid, now() + make_interval(hours => %s))"
)


class EvidenceSet(BaseModel):
    """The passages one answer was built from, and where they came from.

    The subject and topics travel with the chunks because a follow-up names
    neither. Re-inferring them from "can you explain more?" sent an English
    science answer to the Arabic corpus: the prerequisite topics came back in
    Arabic under an English explanation, and the next turn searched the wrong
    subject entirely.
    """

    chunk_ids: list[str] = Field(default_factory=list)
    query: str = ""
    lecture_id: Optional[str] = None
    subject_id: Optional[str] = None
    topic_ids: list[str] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.chunk_ids


class EvidenceStore:
    """The last answer's evidence, for one conversation.

    Only the last: a follow-up can only follow the turn before it, and keeping
    every turn would grow without bound and let "explain more" reach back to a
    question three turns ago.
    """

    def __init__(
        self,
        *,
        config: Any,
        user_id: Optional[str],
        user_type: Optional[str],
        session_id: Optional[str],
    ) -> None:
        self._config = config
        self._user_id = user_id
        self._user_type = user_type
        self._session_id = session_id
        settings = (config.get("agent.evidence") or {}) if config else {}
        self._ttl_hours = settings.get("ttl_hours", DEFAULT_TTL_HOURS)
        self._connect_timeout = settings.get(
            "connect_timeout_seconds", DEFAULT_CONNECT_TIMEOUT)
        self._statement_timeout_ms = settings.get(
            "statement_timeout_ms", DEFAULT_STATEMENT_TIMEOUT_MS)
        self._failure_threshold = settings.get(
            "failure_threshold", DEFAULT_FAILURE_THRESHOLD)
        self._breaker_seconds = settings.get(
            "breaker_seconds", DEFAULT_BREAKER_SECONDS)
        self._failures = 0
        self._closed_until = 0.0

    @property
    def usable(self) -> bool:
        """Whether this turn is one the service remembers anything for."""
        return bool(self._user_id and self._session_id
                    and self._user_type in REMEMBERING_TYPES)

    def recall(self) -> Optional[EvidenceSet]:
        """What the last answer in this conversation was built from."""
        if not self.usable:
            return None
        rows = self._run(
            [(_SELECT, (self._user_id, KIND, self._session_id))],
            fetch=True, label="recall")
        if not rows:
            return None
        try:
            return EvidenceSet.model_validate_json(rows[0][0])
        except ValidationError:
            # The shape changed under it. Stale, not fatal: the turn searches
            # the way it always did.
            logger.warning("stored evidence no longer fits its shape; forgetting it")
            return None

    def remember(self, evidence: EvidenceSet) -> bool:
        """Keep this answer's evidence, replacing whatever came before it.

        An answer grounded in nothing is not kept: remembering an empty set
        would let the next turn re-present a refusal as if it were an
        explanation.
        """
        if not self.usable or evidence.is_empty:
            return False
        ok = bool(self._run([
            (_DELETE, (self._user_id, KIND, self._session_id)),
            (_INSERT, (self._user_id, self._user_type, evidence.model_dump_json(),
                       KIND, self._session_id, self._ttl_hours)),
        ], label="remember"))
        if ok:
            logger.info("evidence remembered", extra={"chunks": len(evidence.chunk_ids)})
        return ok

    # --- plumbing -----------------------------------------------------------

    def _run(self, statements: list, *, fetch: bool = False, label: str = ""):
        """Run one or more statements in a single transaction, or give up."""
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
        except Exception as error:  # noqa: BLE001 — never cost a turn its answer
            self._record_failure(label, error)
            return None
        self._failures = 0
        return rows if fetch else True

    def _record_failure(self, label: str, error: BaseException) -> None:
        self._failures += 1
        logger.warning("turn evidence %s failed", label or "call", exc_info=error)
        if self._failures >= self._failure_threshold:
            self._closed_until = time.monotonic() + self._breaker_seconds
            self._failures = 0
            logger.warning("turn evidence store paused for %ss", self._breaker_seconds)
