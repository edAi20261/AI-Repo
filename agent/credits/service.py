from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from shared.config import Config
from shared.errors import InsufficientCreditsError

from . import ledger
from .catalog import Action, CreditCatalog
from .ledger import Entry, Mutation, Wallet
from .store import CreditStore

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class CreditSummary:
    user_id: str
    plan: str
    balance: int
    allowance_remaining: int
    allowance_total: int
    extra: int
    renews_at: datetime


class Credits:
    def __init__(self, store: CreditStore, catalog: CreditCatalog, *,
                 clock: Clock = _utc_now) -> None:
        self._store = store
        self._catalog = catalog
        self._clock = clock

    @classmethod
    def from_config(cls, config: Config) -> Credits:
        return cls(CreditStore.from_config(config), CreditCatalog.from_config(config))

    @property
    def catalog(self) -> CreditCatalog:
        return self._catalog

    def summary(self, user_id: str) -> CreditSummary:
        return self._summarise(self._transact(user_id, lambda ready: Mutation(ready)).wallet)

    def history(self, user_id: str, *, page: int, page_size: int) -> tuple[list[Entry], int]:
        return self._store.history(user_id, page=page, page_size=page_size)

    def charge(self, user_id: str, action: Action, *, units: int,
               reference_id: Optional[str]) -> Entry:
        price = self._catalog.price(action, units)

        def spend(wallet: Wallet) -> Mutation:
            charged = ledger.debit(wallet, price, action=action.value, reference_id=reference_id)
            if charged is None:
                raise InsufficientCreditsError(
                    "There are not enough credits left for this.",
                    details={"required": price, "balance": wallet.balance,
                             "renews_at": wallet.period_ends_at.isoformat()})
            return charged

        return self._transact(user_id, spend).entries[-1]

    def refund(self, charge: Entry) -> Entry:
        return self._transact(charge.user_id, lambda wallet: ledger.reverse(wallet, charge)).entries[-1]

    def record_cost(self, charge: Entry, cost_usd: float) -> None:
        self._store.record_cost(charge.id, cost_usd)

    def grant(self, user_id: str, amount: int, *, granted_by: str,
              note: Optional[str]) -> CreditSummary:
        mutation = self._transact(
            user_id, lambda wallet: ledger.grant(wallet, amount, actor_id=granted_by, note=note))
        return self._summarise(mutation.wallet)

    def assign_plan(self, user_id: str, plan_code: str, *, changed_by: str) -> CreditSummary:
        plan = self._catalog.plan(plan_code)
        mutation = self._transact(
            user_id, lambda wallet: ledger.restart(wallet, plan, self._clock(), actor_id=changed_by))
        return self._summarise(mutation.wallet)

    def _transact(self, user_id: str, step: Callable[[Wallet], Mutation]) -> Mutation:
        def operation(stored: Optional[Wallet]) -> Mutation:
            ready = self._ready(user_id, stored)
            return ready.then(step(ready.wallet))

        return self._store.transact(user_id, operation)

    def _ready(self, user_id: str, stored: Optional[Wallet]) -> Mutation:
        now = self._clock()
        if stored is None:
            return ledger.open_wallet(user_id, self._catalog.plan(self._catalog.default_plan), now)
        return ledger.renew_if_due(stored, self._catalog.plan(stored.plan), now)

    def _summarise(self, wallet: Wallet) -> CreditSummary:
        return CreditSummary(
            user_id=wallet.user_id, plan=wallet.plan, balance=wallet.balance,
            allowance_remaining=wallet.allowance,
            allowance_total=self._catalog.plan(wallet.plan).monthly_credits,
            extra=wallet.extra, renews_at=wallet.period_ends_at)
