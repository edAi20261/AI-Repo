"""How this service calls the platform's backend.

There were two of these — one in ``shared/backend.py`` for the agent, one in
``ingestion/backend.py`` for the pipeline — reading different endpoints of the
same API through two hand-written copies of the same GET. The copies had drifted
in the way copies do: the agent's retried a 502 three times with backoff, and the
pipeline's gave up on the first blip, abandoning a lesson mid-ingest because a
deploy restarted the backend for two seconds.

The method sets differ and should: the agent reads subjects and question banks,
the pipeline reads lessons and media. The transport underneath them does not.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 0.2


class BackendTransport:
    """A GET against the platform's service-to-service API.

    Retries transient failures — a 5xx or a connection that never landed — and
    raises a 4xx immediately, because a 4xx is this service asking for something
    wrong and a retry cannot make it right.
    """

    def __init__(
        self,
        base_url: str,
        service_token: Optional[str] = None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        # base_url points at the service-to-service prefix (…/api/ai-integration).
        self._base = base_url.rstrip("/")
        # Service-to-service auth: one shared secret, same header both directions.
        self._headers = {"x-service-token": service_token} if service_token else {}
        self._timeout = timeout
        self._max_retries = max(1, max_retries)

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        import httpx  # lazy: importing a client must not require the dependency

        # Drop None params so optional query args stay absent rather than empty.
        clean = {k: v for k, v in (params or {}).items() if v is not None}
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = httpx.get(
                    f"{self._base}{path}",
                    params=clean or None,
                    headers=self._headers,
                    timeout=self._timeout,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as error:
                if error.response is not None and error.response.status_code < 500:
                    raise
                last_error = error
            except httpx.TransportError as error:
                last_error = error
            if attempt < self._max_retries:
                backoff = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning("backend GET retry", extra={
                    "path": path, "attempt": attempt, "backoff": backoff})
                time.sleep(backoff)

        logger.error("backend GET failed", extra={"path": path}, exc_info=last_error)
        assert last_error is not None
        raise last_error
