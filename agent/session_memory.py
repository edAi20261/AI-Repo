"""What the tutor remembers about a conversation between turns.

The service answers each turn from scratch — no checkpointer, no transcript. That
is fine for a question, and wrong for an instruction: "reply in Arabic from now
on" was honoured once and forgotten before the next message arrived.

One field survives, the language the student asked to be answered in, stored
against the session in ``ai.user_memory``. It is read at the top of every turn
and written at the end of the ones that changed it.

Nothing here may cost a student their answer. Every call returns a neutral value
on failure, and a read that fails is indistinguishable from a session with no
instruction — which is the behaviour the service shipped with.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from .config import Config
from shared.postgres import connect

logger = logging.getLogger(__name__)

KIND = "response_language"

# ai.memory_user_type covers these two. A parent turn simply remembers nothing.
REMEMBERING_TYPES = frozenset({"student", "teacher"})

DEFAULT_TTL_HOURS = 720
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


class SessionLanguage:
    """The language a session was told to answer in.

    A database that hangs rather than refuses is the dangerous case: this read
    sits at the front of every turn, so without its own timeouts one bad host
    would stall every request until the service-level ceiling. After a few
    consecutive failures the breaker opens and turns stop paying for it at all.
    """

    def __init__(
        self,
        config: Config,
        *,
        ttl_hours: int = DEFAULT_TTL_HOURS,
        connect_timeout: int = DEFAULT_CONNECT_TIMEOUT,
        statement_timeout_ms: int = DEFAULT_STATEMENT_TIMEOUT_MS,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        breaker_seconds: int = DEFAULT_BREAKER_SECONDS,
    ) -> None:
        self._config = config
        self._ttl_hours = ttl_hours
        self._connect_timeout = connect_timeout
        self._statement_timeout_ms = statement_timeout_ms
        self._failure_threshold = failure_threshold
        self._breaker_seconds = breaker_seconds
        self._failures = 0
        self._closed_until = 0.0

    @classmethod
    def from_config(cls, config: Config) -> Optional[SessionLanguage]:
        """The store, or ``None`` when session memory is switched off."""
        if not config.get("language.sticky.enabled", True):
            return None
        return cls(
            config,
            ttl_hours=config.get("language.sticky.ttl_hours", DEFAULT_TTL_HOURS),
            connect_timeout=config.get(
                "language.sticky.connect_timeout_seconds", DEFAULT_CONNECT_TIMEOUT),
            statement_timeout_ms=config.get(
                "language.sticky.statement_timeout_ms", DEFAULT_STATEMENT_TIMEOUT_MS),
            failure_threshold=config.get(
                "language.sticky.failure_threshold", DEFAULT_FAILURE_THRESHOLD),
            breaker_seconds=config.get(
                "language.sticky.breaker_seconds", DEFAULT_BREAKER_SECONDS),
        )

    # --- the three operations ----------------------------------------------

    def get(self, user_id: Optional[str], session_id: Optional[str]) -> Optional[str]:
        """The language this session was told to answer in, if it still stands."""
        if not user_id or not session_id:
            return None
        rows = self._run(
            [(_SELECT, (user_id, KIND, session_id))], fetch=True, label="read")
        return rows[0][0] if rows else None

    def set(
        self,
        user_id: Optional[str],
        session_id: Optional[str],
        *,
        code: str,
        user_type: Optional[str],
        evidence: Optional[str] = None,
    ) -> bool:
        """Remember an instruction. Replaces whatever the session held before."""
        if not (user_id and session_id and code):
            return False
        if user_type not in REMEMBERING_TYPES:
            logger.info("session language not stored for %r", user_type)
            return False
        # Delete then insert, in one transaction, so a session holds one row
        # rather than a growing pile: the read stays a single lookup and the
        # table stays the size of live conversations. What changed when is in
        # the logs. Two statements, because a parameterised execute carries
        # exactly one command.
        ok = bool(self._run(
            [
                (_DELETE, (user_id, KIND, session_id)),
                (_INSERT, (user_id, user_type, code, KIND, session_id, self._ttl_hours)),
            ],
            label="write",
        ))
        if ok:
            logger.info("session language set", extra={
                "session_language": code, "session_language_evidence": evidence})
        return ok

    def clear(self, user_id: Optional[str], session_id: Optional[str]) -> bool:
        """Forget the instruction: later turns follow the message again."""
        if not (user_id and session_id):
            return False
        ok = bool(self._run([(_DELETE, (user_id, KIND, session_id))], label="clear"))
        if ok:
            logger.info("session language cleared")
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
        logger.warning("session language %s failed", label or "call", exc_info=error)
        if self._failures >= self._failure_threshold:
            self._closed_until = time.monotonic() + self._breaker_seconds
            self._failures = 0
            logger.warning("session language store paused for %ss",
                           self._breaker_seconds)
