"""The id of the request a log line belongs to.

Held in a :class:`~contextvars.ContextVar` so log records and error envelopes can
reach it without threading an argument through every call. It lives here because
both the HTTP layer, which sets it, and the log formatting, which reads it, need
it, and ``shared`` is the only place both may import from.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

_request_id: ContextVar[str] = ContextVar("request_id", default="")


def new_request_id() -> str:
    return uuid.uuid4().hex


def set_request_id(value: str) -> None:
    _request_id.set(value)


def get_request_id() -> str:
    return _request_id.get()
