"""Liveness and readiness probes. Unauthenticated by design — orchestrators
call them without credentials, and they expose no data."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from ..dependencies import SettingsDep
from ..schemas.health import HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(environment=settings.app_env)


@router.get("/ready", response_model=ReadinessResponse, summary="Readiness probe")
async def ready(request: Request, response: Response, settings: SettingsDep) -> ReadinessResponse:
    runtime = getattr(request.app.state, "runtime", None)
    ready_now = runtime is not None
    if not ready_now:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="ready" if ready_now else "starting",
        environment=settings.app_env,
        graph_ready=ready_now,
        in_flight=runtime.in_flight if runtime else 0,
    )
