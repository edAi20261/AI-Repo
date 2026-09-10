"""Error taxonomy and the handlers that render it.

Every failure leaves the service in the same envelope — ``{error: {code,
message, request_id}}`` — so the platform backend can branch on ``code`` and
surface ``request_id`` in a support ticket. Internal detail never reaches a
client.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from agent.tools import AuthorizationError

from .context import get_request_id

logger = logging.getLogger(__name__)


class ServiceError(Exception):
    """Base class for failures with a deliberate HTTP mapping."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class UnauthorizedError(ServiceError):
    status_code = 401
    code = "unauthorized"


class ForbiddenError(ServiceError):
    status_code = 403
    code = "forbidden"


class NotFoundError(ServiceError):
    status_code = 404
    code = "not_found"


class ConflictError(ServiceError):
    status_code = 409
    code = "conflict"


class ServiceUnavailableError(ServiceError):
    status_code = 503
    code = "service_unavailable"


class TimeoutError_(ServiceError):
    """The turn exceeded ``api.request_timeout``."""

    status_code = 504
    code = "timeout"


def _envelope(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"code": code, "message": message, "request_id": get_request_id()}
    if details:
        body["details"] = details
    return {"error": body}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def _service_error(_: Request, exc: ServiceError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error("service error: %s", exc.message, exc_info=exc)
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(exc.code, exc.message, exc.details),
        )

    @app.exception_handler(AuthorizationError)
    async def _authorization_error(_: Request, exc: AuthorizationError) -> JSONResponse:
        return JSONResponse(
            status_code=403,
            content=_envelope("forbidden", str(exc) or "Not permitted for this role."),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_envelope(
                "validation_error",
                "The request body failed validation.",
                {"fields": _readable_fields(exc)},
            ),
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled error", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content=_envelope("internal_error", "An unexpected error occurred."),
        )


def _readable_fields(exc: RequestValidationError) -> list[dict[str, str]]:
    fields = []
    for err in exc.errors():
        location = ".".join(str(part) for part in err.get("loc", ()) if part != "body")
        fields.append({"field": location or "body", "message": err.get("msg", "invalid")})
    return fields
