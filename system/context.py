"""Per-request context for the HTTP layer."""

from __future__ import annotations

from shared.request_context import get_request_id, new_request_id, set_request_id

REQUEST_ID_HEADER = "x-request-id"

__all__ = ["REQUEST_ID_HEADER", "get_request_id", "new_request_id", "set_request_id"]
