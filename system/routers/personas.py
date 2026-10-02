"""Teacher personas: build one from the teacher's material, review it, approve it.

Which teacher a caller may act for is decided by the platform, as it is for every
endpoint on this service. What is decided here is the one rule the platform
cannot see: nothing reaches a student until the teacher has approved a version.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from agent.models import PersonaCard

from ..dependencies import Authenticated, RequestId, RuntimeDep
from ..schemas.common import ERROR_RESPONSES, PAID_RESPONSES
from ..schemas.personas import (
    ExemplarView,
    PersonaBuildRequest,
    PersonaBuildResponse,
    PersonaDeactivateResponse,
    PersonaDeleteResponse,
    PersonaDetailResponse,
    PersonaListResponse,
    PersonaResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ai/personas", tags=["personas"], dependencies=[Authenticated],
    responses=ERROR_RESPONSES,
)


@router.post(
    "/build",
    response_model=PersonaBuildResponse,
    summary="Draft a new persona version from the teacher's material",
    responses=PAID_RESPONSES,
)
async def build(payload: PersonaBuildRequest, runtime: RuntimeDep,
                request_id: RequestId) -> PersonaBuildResponse:
    built = await runtime.build_persona(payload.to_request(),
                                        billed_user_id=payload.billed_user_id)
    response = PersonaBuildResponse.of(built.result, request_id, built.meter)
    logger.info("persona built", extra={
        "teacher_id": payload.teacher_id, "version": response.persona.version,
        "sources": len(payload.sources), "skipped": len(response.skipped),
        "exemplars": len(response.exemplars)})
    return response


@router.get(
    "/{teacher_id}",
    response_model=PersonaListResponse,
    summary="Every persona version a teacher has, and which one students hear",
)
async def versions(teacher_id: str, runtime: RuntimeDep,
                   request_id: RequestId) -> PersonaListResponse:
    found = await runtime.personas(teacher_id)
    active = next((p.version for p in found if p.status == "active"), None)
    return PersonaListResponse(request_id=request_id, active_version=active, versions=found)


@router.get(
    "/{teacher_id}/versions/{version}",
    response_model=PersonaDetailResponse,
    summary="One persona version, with the examples chosen from the teacher's words",
)
async def detail(teacher_id: str, version: int, runtime: RuntimeDep,
                 request_id: RequestId) -> PersonaDetailResponse:
    view = await runtime.persona(teacher_id, version)
    return PersonaDetailResponse(request_id=request_id, persona=view.persona,
                                 exemplars=[ExemplarView.of(e) for e in view.exemplars])


@router.put(
    "/{teacher_id}/versions/{version}/card",
    response_model=PersonaResponse,
    summary="Replace a version's card with the teacher's corrections",
)
async def edit_card(teacher_id: str, version: int, card: PersonaCard, runtime: RuntimeDep,
                    request_id: RequestId) -> PersonaResponse:
    persona = await runtime.edit_persona_card(teacher_id, version, card)
    return PersonaResponse(request_id=request_id, persona=persona)


@router.post(
    "/{teacher_id}/versions/{version}/activate",
    response_model=PersonaResponse,
    summary="Approve a version: it becomes the one the teacher's students hear",
)
async def activate(teacher_id: str, version: int, runtime: RuntimeDep,
                   request_id: RequestId) -> PersonaResponse:
    persona = await runtime.activate_persona(teacher_id, version)
    logger.info("persona activated", extra={"teacher_id": teacher_id, "version": version})
    return PersonaResponse(request_id=request_id, persona=persona)


@router.post(
    "/{teacher_id}/deactivate",
    response_model=PersonaDeactivateResponse,
    summary="Stop using the teacher's persona; students hear the default tutor",
)
async def deactivate(teacher_id: str, runtime: RuntimeDep,
                     request_id: RequestId) -> PersonaDeactivateResponse:
    retired = await runtime.deactivate_persona(teacher_id)
    return PersonaDeactivateResponse(request_id=request_id,
                                     deactivated_version=retired.version if retired else None)


@router.delete(
    "/{teacher_id}",
    response_model=PersonaDeleteResponse,
    summary="Destroy every persona version and example held for a teacher",
)
async def delete(teacher_id: str, runtime: RuntimeDep,
                 request_id: RequestId) -> PersonaDeleteResponse:
    deleted = await runtime.delete_persona(teacher_id)
    logger.info("persona deleted on request", extra={
        "teacher_id": teacher_id, "versions": deleted})
    return PersonaDeleteResponse(request_id=request_id, versions_deleted=deleted)
