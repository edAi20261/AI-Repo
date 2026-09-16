"""The one root every deliberate failure in this service grows from.

There were three disjoint hierarchies: ``ServiceError`` in ``system/``, which the
HTTP layer knows how to render; ``StorageError`` and ``StructuredOutputError``
in ``shared/``; and ``SourceError`` in ``ingestion/``. The last three inherited
``RuntimeError``, so nothing the agent or the pipeline raised could reach an HTTP
status except by accident — every one of them fell through to the catch-all
handler and came back as 500 "An unexpected error occurred."

S3 refusing a signed URL and the ingest code having a bug are not the same event,
and a caller that cannot tell them apart cannot retry the one that is worth
retrying.

The subclasses live here rather than in ``system/`` because raising them must not
require FastAPI: the pipeline runs as a CLI.
"""

from __future__ import annotations

from typing import Any


class ServiceError(Exception):
    """A failure with a deliberate HTTP mapping.

    ``status_code`` and ``code`` are what the API envelope renders; ``message``
    is written for whoever reads the response, so it carries no internal detail.
    """

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class UnauthorizedError(ServiceError):
    status_code = 401
    code = "unauthorized"


class NotFoundError(ServiceError):
    status_code = 404
    code = "not_found"


class ConflictError(ServiceError):
    status_code = 409
    code = "conflict"


class UnprocessableError(ServiceError):
    """The request is well-formed but the material it names cannot be used —
    a missing source file, a PDF with no readable pages."""

    status_code = 422
    code = "unprocessable"


class ServiceUnavailableError(ServiceError):
    """A dependency this service does not own is refusing or unreachable."""

    status_code = 503
    code = "service_unavailable"


class UpstreamTimeoutError(ServiceError):
    """The turn exceeded ``api.request_timeout``."""

    status_code = 504
    code = "timeout"
