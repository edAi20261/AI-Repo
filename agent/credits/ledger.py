from __future__ import annotations

import calendar
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from typing import Optional

from .catalog import Plan


class Kind(StrEnum):
    ALLOWANCE_RENEWAL = "allowance_renewal"
    GRANT = "grant"
    USAGE = "usage"
    REFUND = "refund"


@dataclass(frozen=True)
class Wallet:
    user_id: str
    plan: str
    allowance: int
    extra: int
    period_started_at: datetime
    period_ends_at: datetime

    @property
    def balance(self) -> int:
        return self.allowance + self.extra


@dataclass(frozen=True)
class Entry:
    user_id: str
    kind: Kind
    allowance_delta: int
    extra_delta: int
    balance_after: int
    action: Optional[str] = None
    reverses_id: Optional[str] = None
    reference_id: Optional[str] = None
    actor_id: Optional[str] = None
    note: Optional[str] = None
    cost_usd: Optional[float] = None
    id: Optional[str] = None
    created_at: Optional[datetime] = None

    @property
    def amount(self) -> int:
        return self.allowance_delta + self.extra_delta


@dataclass(frozen=True)
class Mutation:
    wallet: Wallet
    entries: tuple[Entry, ...] = ()

    def then(self, following: Mutation) -> Mutation:
        return Mutation(following.wallet, self.entries + following.entries)


def period_starting_at(start: datetime) -> tuple[datetime, datetime]:
    year, month = (start.year + 1, 1) if start.month == 12 else (start.year, start.month + 1)
    day = min(start.day, calendar.monthrange(year, month)[1])
    return start, start.replace(year=year, month=month, day=day)


def open_wallet(user_id: str, plan: Plan, now: datetime) -> Mutation:
    started, ends = period_starting_at(now)
    empty = Wallet(user_id=user_id, plan=plan.code, allowance=0, extra=0,
                   period_started_at=started, period_ends_at=ends)
    return _post(empty, Kind.ALLOWANCE_RENEWAL, plan.monthly_credits, 0)


def renew_if_due(wallet: Wallet, plan: Plan, now: datetime) -> Mutation:
    if wallet.period_ends_at > now:
        return Mutation(wallet)
    return restart(wallet, plan, now)


def restart(wallet: Wallet, plan: Plan, now: datetime, *,
            actor_id: Optional[str] = None) -> Mutation:
    started, ends = period_starting_at(now)
    moved = replace(wallet, plan=plan.code, period_started_at=started, period_ends_at=ends)
    return _post(moved, Kind.ALLOWANCE_RENEWAL, plan.monthly_credits - wallet.allowance, 0,
                 actor_id=actor_id)


def debit(wallet: Wallet, price: int, *, action: str,
          reference_id: Optional[str]) -> Optional[Mutation]:
    if wallet.balance < price:
        return None
    from_allowance = min(wallet.allowance, price)
    return _post(wallet, Kind.USAGE, -from_allowance, from_allowance - price,
                 action=action, reference_id=reference_id)


def reverse(wallet: Wallet, charge: Entry) -> Mutation:
    return _post(wallet, Kind.REFUND, -charge.allowance_delta, -charge.extra_delta,
                 action=charge.action, reverses_id=charge.id, reference_id=charge.reference_id)


def grant(wallet: Wallet, amount: int, *, actor_id: str, note: Optional[str]) -> Mutation:
    return _post(wallet, Kind.GRANT, 0, amount, actor_id=actor_id, note=note)


def _post(wallet: Wallet, kind: Kind, allowance_delta: int, extra_delta: int,
          **details: Optional[str]) -> Mutation:
    changed = replace(wallet, allowance=wallet.allowance + allowance_delta,
                      extra=wallet.extra + extra_delta)
    entry = Entry(user_id=wallet.user_id, kind=kind, allowance_delta=allowance_delta,
                  extra_delta=extra_delta, balance_after=changed.balance, **details)
    return Mutation(changed, (entry,))
