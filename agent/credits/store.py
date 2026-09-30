from __future__ import annotations

import logging
import threading
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from typing import Optional

from shared.config import Config
from shared.errors import ServiceError, ServiceUnavailableError
from shared.postgres import connect

from .ledger import Entry, Kind, Mutation, Wallet

logger = logging.getLogger(__name__)

Operation = Callable[[Optional[Wallet]], Mutation]


class CreditStore(ABC):
    @abstractmethod
    def transact(self, user_id: str, operation: Operation) -> Mutation:
        ...

    @abstractmethod
    def history(self, user_id: str, *, page: int, page_size: int) -> tuple[list[Entry], int]:
        ...

    @abstractmethod
    def record_cost(self, entry_id: str, cost_usd: float) -> None:
        ...

    @classmethod
    def from_config(cls, config: Config) -> CreditStore:
        kind = config.get("credits.store")
        if kind == "memory":
            return InMemoryCreditStore()
        if kind == "postgres":
            return PostgresCreditStore(config)
        raise ValueError(f"credits.store must be 'postgres' or 'memory', not {kind!r}")


class InMemoryCreditStore(CreditStore):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._wallets: dict[str, Wallet] = {}
        self._entries: list[Entry] = []

    def transact(self, user_id: str, operation: Operation) -> Mutation:
        with self._lock:
            mutation = operation(self._wallets.get(user_id))
            now = datetime.now(timezone.utc)
            stored = tuple(replace(entry, id=uuid.uuid4().hex, created_at=now)
                           for entry in mutation.entries)
            self._wallets[user_id] = mutation.wallet
            self._entries.extend(stored)
            return Mutation(mutation.wallet, stored)

    def history(self, user_id: str, *, page: int, page_size: int) -> tuple[list[Entry], int]:
        with self._lock:
            mine = [entry for entry in reversed(self._entries) if entry.user_id == user_id]
        start = (page - 1) * page_size
        return mine[start:start + page_size], len(mine)

    def record_cost(self, entry_id: str, cost_usd: float) -> None:
        with self._lock:
            self._entries = [replace(entry, cost_usd=cost_usd) if entry.id == entry_id else entry
                             for entry in self._entries]


_LOCK = "SELECT pg_advisory_xact_lock(hashtext(%s))"

_WALLET = (
    "SELECT plan, allowance, extra, period_started_at, period_ends_at "
    "FROM ai.credit_wallets WHERE user_id = %s"
)

_SAVE_WALLET = (
    "INSERT INTO ai.credit_wallets "
    "(user_id, plan, allowance, extra, period_started_at, period_ends_at) "
    "VALUES (%s, %s, %s, %s, %s, %s) "
    "ON CONFLICT (user_id) DO UPDATE SET plan = EXCLUDED.plan, "
    "allowance = EXCLUDED.allowance, extra = EXCLUDED.extra, "
    "period_started_at = EXCLUDED.period_started_at, "
    "period_ends_at = EXCLUDED.period_ends_at, updated_at = now()"
)

_ADD_ENTRY = (
    "INSERT INTO ai.credit_ledger "
    "(user_id, kind, action, allowance_delta, extra_delta, balance_after, "
    "reverses_id, reference_id, actor_id, note) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
    "RETURNING id::text, created_at"
)

_HISTORY = (
    "SELECT id::text, kind, action, allowance_delta, extra_delta, balance_after, "
    "reverses_id::text, reference_id, actor_id, note, cost_usd, created_at "
    "FROM ai.credit_ledger WHERE user_id = %s "
    "ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s"
)

_COUNT = "SELECT count(*) FROM ai.credit_ledger WHERE user_id = %s"

_RECORD_COST = "UPDATE ai.credit_ledger SET cost_usd = %s WHERE id = %s"


class PostgresCreditStore(CreditStore):
    def __init__(self, config: Config) -> None:
        self._config = config
        self._connect_timeout = config.get("credits.connect_timeout_seconds")
        self._statement_timeout_ms = config.get("credits.statement_timeout_ms")

    def transact(self, user_id: str, operation: Operation) -> Mutation:
        with _failing_closed("transaction", user_id):
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(_LOCK, (user_id,))
                cur.execute(_WALLET, (user_id,))
                row = cur.fetchone()
                current = Wallet(user_id, *row) if row else None
                mutation = operation(current)
                wallet = mutation.wallet
                if wallet != current:
                    cur.execute(_SAVE_WALLET, (user_id, wallet.plan, wallet.allowance,
                                               wallet.extra, wallet.period_started_at,
                                               wallet.period_ends_at))
                stored = []
                for entry in mutation.entries:
                    cur.execute(_ADD_ENTRY, (
                        user_id, entry.kind.value, entry.action, entry.allowance_delta,
                        entry.extra_delta, entry.balance_after, entry.reverses_id,
                        entry.reference_id, entry.actor_id, entry.note))
                    entry_id, created_at = cur.fetchone()
                    stored.append(replace(entry, id=entry_id, created_at=created_at))
                conn.commit()
        return Mutation(wallet, tuple(stored))

    def history(self, user_id: str, *, page: int, page_size: int) -> tuple[list[Entry], int]:
        with _failing_closed("history", user_id):
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(_HISTORY, (user_id, page_size, (page - 1) * page_size))
                rows = cur.fetchall()
                cur.execute(_COUNT, (user_id,))
                (total,) = cur.fetchone()
        return [_entry(user_id, row) for row in rows], total

    def record_cost(self, entry_id: str, cost_usd: float) -> None:
        with _failing_closed("cost", entry_id):
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(_RECORD_COST, (cost_usd, entry_id))
                conn.commit()

    def _connect(self):
        return connect(self._config, connect_timeout=self._connect_timeout,
                       statement_timeout_ms=self._statement_timeout_ms)


@contextmanager
def _failing_closed(label: str, subject: str) -> Iterator[None]:
    try:
        yield
    except ServiceError:
        raise
    except Exception as error:
        logger.error("credit store %s failed", label, extra={"subject": subject}, exc_info=error)
        raise ServiceUnavailableError(
            "Credits cannot be checked right now. Please try again shortly.") from error


def _entry(user_id: str, row: tuple) -> Entry:
    (entry_id, kind, action, allowance_delta, extra_delta, balance_after, reverses_id,
     reference_id, actor_id, note, cost_usd, created_at) = row
    return Entry(user_id=user_id, kind=Kind(kind), action=action,
                 allowance_delta=allowance_delta, extra_delta=extra_delta,
                 balance_after=balance_after, reverses_id=reverses_id,
                 reference_id=reference_id, actor_id=actor_id, note=note,
                 cost_usd=float(cost_usd) if cost_usd is not None else None,
                 id=entry_id, created_at=created_at)
