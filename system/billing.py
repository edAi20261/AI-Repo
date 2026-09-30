from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import partial
from typing import Optional, TypeVar

import anyio

from agent.credits import Action, CreditCatalog, Credits, CreditSummary, Entry
from shared.config import Config
from shared.metering import Meter

from .context import get_request_id
from .errors import ServiceUnavailableError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Bill:
    def __init__(self, credits: Optional[Credits], charge: Optional[Entry]) -> None:
        self._credits = credits
        self._charge = charge
        self._meter: Optional[Meter] = None
        self._voided = False

    def settle(self, meter: Meter) -> None:
        self._meter = meter

    def void(self) -> None:
        self._voided = True

    async def close(self) -> None:
        charge, self._charge = self._charge, None
        if charge is None or self._credits is None:
            return
        try:
            if self._voided:
                await anyio.to_thread.run_sync(self._credits.refund, charge)
            elif self._meter is not None:
                await anyio.to_thread.run_sync(
                    self._credits.record_cost, charge, self._meter.cost_usd)
        except Exception:
            logger.exception("closing the bill failed", extra={
                "charge_id": charge.id, "user_id": charge.user_id, "refund": self._voided})


class Billing:
    def __init__(self, credits: Optional[Credits]) -> None:
        self._credits = credits
        if credits is None:
            logger.warning("credits are disabled: AI usage is not charged")

    @classmethod
    def from_config(cls, config: Config) -> Billing:
        return cls(Credits.from_config(config) if config.get("credits.enabled") else None)

    async def open(self, user_id: str, action: Action, *, units: int = 1) -> Bill:
        if self._credits is None:
            return Bill(None, None)
        charge = await anyio.to_thread.run_sync(partial(
            self._credits.charge, user_id, action, units=units, reference_id=get_request_id()))
        return Bill(self._credits, charge)

    @asynccontextmanager
    async def guard(self, bill: Bill) -> AsyncIterator[Bill]:
        try:
            yield bill
        except BaseException:
            bill.void()
            raise
        finally:
            with anyio.CancelScope(shield=True):
                await bill.close()

    @asynccontextmanager
    async def spend(self, user_id: str, action: Action, *, units: int = 1) -> AsyncIterator[Bill]:
        async with self.guard(await self.open(user_id, action, units=units)) as bill:
            yield bill

    @property
    def catalog(self) -> CreditCatalog:
        return self._enabled().catalog

    async def summary(self, user_id: str) -> CreditSummary:
        return await self._run(lambda credits: credits.summary(user_id))

    async def history(self, user_id: str, *, page: int, page_size: int) -> tuple[list[Entry], int]:
        return await self._run(
            lambda credits: credits.history(user_id, page=page, page_size=page_size))

    async def grant(self, user_id: str, amount: int, *, granted_by: str,
                    note: Optional[str]) -> CreditSummary:
        return await self._run(
            lambda credits: credits.grant(user_id, amount, granted_by=granted_by, note=note))

    async def assign_plan(self, user_id: str, plan: str, *, changed_by: str) -> CreditSummary:
        return await self._run(
            lambda credits: credits.assign_plan(user_id, plan, changed_by=changed_by))

    async def _run(self, call: Callable[[Credits], T]) -> T:
        credits = self._enabled()
        return await anyio.to_thread.run_sync(call, credits)

    def _enabled(self) -> Credits:
        if self._credits is None:
            raise ServiceUnavailableError("Credits are not enabled on this service.")
        return self._credits
