"""Credits — balances, history, grants and plans, one user at a time.

Who may read or change whose credits is the platform's decision; this service
only checks the service token.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from ..dependencies import Authenticated, BillingDep, RequestId
from ..schemas.common import ERROR_RESPONSES
from ..schemas.credits import (
    CatalogResponse,
    CreditBalanceResponse,
    GrantRequest,
    LedgerEntry,
    LedgerResponse,
    PlanChangeRequest,
)

router = APIRouter(
    prefix="/ai/credits", tags=["credits"], dependencies=[Authenticated],
    responses=ERROR_RESPONSES,
)


@router.get("/plans", response_model=CatalogResponse, summary="Plans and what each action costs")
async def catalog(billing: BillingDep, request_id: RequestId) -> CatalogResponse:
    return CatalogResponse.from_catalog(billing.catalog, request_id)


@router.get("/{user_id}", response_model=CreditBalanceResponse, summary="A user's credit balance")
async def balance(user_id: str, billing: BillingDep,
                  request_id: RequestId) -> CreditBalanceResponse:
    return CreditBalanceResponse.from_summary(await billing.summary(user_id), request_id)


@router.get("/{user_id}/ledger", response_model=LedgerResponse,
            summary="A user's credit history, newest first")
async def ledger(
    user_id: str,
    billing: BillingDep,
    request_id: RequestId,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> LedgerResponse:
    entries, total = await billing.history(user_id, page=page, page_size=page_size)
    return LedgerResponse(request_id=request_id, user_id=user_id, page=page,
                          page_size=page_size, total=total,
                          entries=[LedgerEntry.from_entry(entry) for entry in entries])


@router.post("/{user_id}/grants", response_model=CreditBalanceResponse,
             summary="Give a user extra credits that never expire")
async def grant(user_id: str, payload: GrantRequest, billing: BillingDep,
                request_id: RequestId) -> CreditBalanceResponse:
    summary = await billing.grant(user_id, payload.amount, granted_by=payload.granted_by,
                                  note=payload.note)
    return CreditBalanceResponse.from_summary(summary, request_id)


@router.put("/{user_id}/plan", response_model=CreditBalanceResponse,
            summary="Move a user to another plan; their allowance restarts at it")
async def change_plan(user_id: str, payload: PlanChangeRequest, billing: BillingDep,
                      request_id: RequestId) -> CreditBalanceResponse:
    summary = await billing.assign_plan(user_id, payload.plan, changed_by=payload.changed_by)
    return CreditBalanceResponse.from_summary(summary, request_id)
