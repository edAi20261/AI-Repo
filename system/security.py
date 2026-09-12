"""Startup preflight and service-to-service authentication.

This service is not public. The platform backend calls it with the shared
secret already used in the other direction (``x-service-token``), so login and
per-user authorisation stay in one place — the platform — and never move here.
"""

from __future__ import annotations

import logging
from secrets import compare_digest
from typing import Optional

from fastapi import Depends, Request
from fastapi.security import APIKeyHeader

from shared.config import Config

from .errors import UnauthorizedError
from .settings import Settings

logger = logging.getLogger(__name__)

SERVICE_TOKEN_HEADER = "x-service-token"

# Declared as an OpenAPI security scheme, not just read off the request, so the
# header is documented and /docs offers an Authorize box for it.
service_token_scheme = APIKeyHeader(
    name=SERVICE_TOKEN_HEADER,
    auto_error=False,
    description=(
        "Shared secret (AI_WEBHOOK_SHARED_SECRET). The platform backend sends this "
        "on every call; it is the same secret this service sends back."
    ),
)


def verify_service_token(
    request: Request,
    presented: Optional[str] = Depends(service_token_scheme),
) -> None:
    """Reject a request that does not carry the shared secret.

    A service with no configured token runs unauthenticated; that is allowed in
    a development environment and refused at startup anywhere else (see
    :func:`assert_token_configured`).
    """
    settings: Settings = request.app.state.settings
    if not settings.service_token:
        return

    if not presented or not compare_digest(presented, settings.service_token):
        raise UnauthorizedError("A valid service token is required.")


def assert_token_configured(settings: Settings) -> None:
    """Fail fast rather than serve an unauthenticated production service."""
    if settings.service_token:
        return
    if settings.is_dev:
        logger.warning(
            "AI_WEBHOOK_SHARED_SECRET is unset; the service is running unauthenticated "
            "(permitted for APP_ENV=%s only).",
            settings.app_env,
        )
        return
    raise RuntimeError(
        f"AI_WEBHOOK_SHARED_SECRET is unset for APP_ENV={settings.app_env!r}. "
        "Refusing to start an unauthenticated service."
    )


def assert_stores_configured(settings: Settings) -> None:
    """Refuse to start without the stores every read path needs.

    Retrieval and mastery both go straight to Postgres and Qdrant; without them
    a chat turn fails deep inside the graph rather than at the door.
    """
    missing = [name for name in ("DATABASE_URL", "QDRANT_URL") if not Config.secret(name)]
    if not missing:
        return
    if settings.is_dev:
        logger.warning(
            "%s unset; retrieval and mastery will fail on a real turn (APP_ENV=%s).",
            " and ".join(missing),
            settings.app_env,
        )
        return
    raise RuntimeError(
        f"{' and '.join(missing)} unset for APP_ENV={settings.app_env!r}. "
        "Refusing to start a service that cannot reach its stores."
    )
