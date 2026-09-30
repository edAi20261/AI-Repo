"""The credits contract: what a user may spend, and what they have spent it on."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import Field

from agent.credits import CreditCatalog, CreditSummary, Entry

from .common import Schema


class Allowance(Schema):
    remaining: int
    total: int = Field(description="What the plan grants every period.")


class CreditBalanceResponse(Schema):
    request_id: str
    user_id: str
    plan: str
    balance: int = Field(description="Allowance left plus extra credits.")
    allowance: Allowance
    extra: int = Field(description="Granted credits; they never expire and are spent last.")
    renews_at: datetime

    @classmethod
    def from_summary(cls, summary: CreditSummary, request_id: str) -> CreditBalanceResponse:
        return cls(
            request_id=request_id, user_id=summary.user_id, plan=summary.plan,
            balance=summary.balance, extra=summary.extra, renews_at=summary.renews_at,
            allowance=Allowance(remaining=summary.allowance_remaining,
                                total=summary.allowance_total))


class LedgerEntry(Schema):
    id: str
    kind: str
    action: Optional[str] = None
    amount: int = Field(description="Credits added (positive) or spent (negative).")
    balance_after: int
    reference_id: Optional[str] = Field(
        default=None, description="The request that spent or refunded the credits.")
    note: Optional[str] = None
    created_at: datetime

    @classmethod
    def from_entry(cls, entry: Entry) -> LedgerEntry:
        return cls(id=entry.id, kind=entry.kind.value, action=entry.action,
                   amount=entry.amount, balance_after=entry.balance_after,
                   reference_id=entry.reference_id, note=entry.note,
                   created_at=entry.created_at)


class LedgerResponse(Schema):
    request_id: str
    user_id: str
    entries: list[LedgerEntry]
    page: int
    page_size: int
    total: int


class GrantRequest(Schema):
    amount: int = Field(gt=0, le=100_000)
    granted_by: str = Field(min_length=1, description="The admin who granted them.")
    note: Optional[str] = Field(default=None, max_length=500)


class PlanChangeRequest(Schema):
    plan: str = Field(min_length=1)
    changed_by: str = Field(min_length=1)


class PlanItem(Schema):
    code: str
    monthly_credits: int


class PriceItem(Schema):
    action: str
    credits: int
    per: int = Field(description="Units one charge of `credits` covers, e.g. answers graded.")


class CatalogResponse(Schema):
    request_id: str
    default_plan: str
    plans: list[PlanItem]
    prices: list[PriceItem]

    @classmethod
    def from_catalog(cls, catalog: CreditCatalog, request_id: str) -> CatalogResponse:
        return cls(
            request_id=request_id, default_plan=catalog.default_plan,
            plans=[PlanItem(code=plan.code, monthly_credits=plan.monthly_credits)
                   for plan in catalog.plans.values()],
            prices=[PriceItem(action=action.value, credits=price.credits, per=price.per)
                    for action, price in catalog.prices.items()])
