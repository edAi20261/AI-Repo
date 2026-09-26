"""Application factory.

The compiled graph is expensive to build, so it is created once during startup
and shared by every request. Startup fails loudly rather than serving a
half-configured service.
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from shared.config import Config, get_config
from shared.embeddings import warm_embedder
from shared.logging import setup_logging

from .errors import register_exception_handlers
from .ingestion_jobs import IngestionJobManager
from .middleware import RequestContextMiddleware
from .notifier import build_notifier
from .routers import chat, grading, health, homework, ingestion, memory, reports
from .runtime import AgentRuntime
from .security import (
    assert_stores_configured,
    assert_token_configured,
    warn_if_index_empty,
)
from .settings import Settings

logger = logging.getLogger(__name__)

API_TITLE = "EduAI AI Service"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
Service-to-service API over the EduAI runtime agent and ingestion pipeline.

Not public: the platform backend authenticates the end user, then calls this
service with the shared `x-service-token` secret.
"""


def create_app(config: Config | None = None) -> FastAPI:
    config = config or get_config()
    settings = Settings.from_config(config)
    setup_logging(config)
    assert_token_configured(settings)
    assert_stores_configured(settings)
    warn_if_index_empty(config)

    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        lifespan=_lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    app.state.config = config
    app.state.settings = settings

    app.add_middleware(RequestContextMiddleware)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            # DELETE is here for /ai/memory/{user_id}: a student asking to be
            # forgotten is the one request this service must not refuse on a
            # preflight.
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["*"],
        )

    register_exception_handlers(app)
    for module in (health, chat, grading, homework, reports, ingestion, memory):
        app.include_router(module.router)
    return app


@asynccontextmanager
async def _lifespan(app: FastAPI):
    config: Config = app.state.config
    settings: Settings = app.state.settings

    app.state.runtime = AgentRuntime.build(
        config,
        max_concurrency=settings.max_concurrency,
        timeout=settings.request_timeout,
    )
    app.state.jobs = IngestionJobManager(
        config, settings.ingestion, notifier=build_notifier(settings)
    )
    if config.get("embeddings.warm_on_start", True):
        threading.Thread(target=warm_embedder, args=(config,),
                         name="embedder-warm-up", daemon=True).start()
    logger.info("service ready", extra={"environment": settings.app_env})

    try:
        yield
    finally:
        await app.state.jobs.drain()
        logger.info("service stopped")

