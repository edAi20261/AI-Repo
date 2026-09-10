"""FastAPI dependencies.

Process-wide singletons live on ``app.state`` and are reached through these
accessors, so a router never imports the runtime directly and tests can swap
either one out.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from .context import get_request_id
from .errors import ServiceUnavailableError
from .ingestion_jobs import IngestionJobManager
from .runtime import AgentRuntime
from .security import verify_service_token
from .settings import Settings


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_runtime(request: Request) -> AgentRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise ServiceUnavailableError("The runtime graph is not available.")
    return runtime


def get_jobs(request: Request) -> IngestionJobManager:
    jobs = getattr(request.app.state, "jobs", None)
    if jobs is None:
        raise ServiceUnavailableError("The ingestion worker is not available.")
    return jobs


def request_id() -> str:
    return get_request_id()


Authenticated = Depends(verify_service_token)
RuntimeDep = Annotated[AgentRuntime, Depends(get_runtime)]
JobsDep = Annotated[IngestionJobManager, Depends(get_jobs)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
RequestId = Annotated[str, Depends(request_id)]
