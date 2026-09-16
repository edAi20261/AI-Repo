"""Project logging setup — one convention across ingestion, agent and system.

Convention (apply everywhere, including code written later):
  - Call ``setup_logging(config)`` ONCE at process start (a CLI entrypoint or app
    startup), before anything logs.
  - In every module: ``logger = logging.getLogger(__name__)``. Never use ``print``
    for diagnostics — ``print`` is only for deliberate output a CLI user reads.
  - Levels: DEBUG (developer detail), INFO (lifecycle — a lesson/stage/request
    started or finished, with counts), WARNING (a degraded but recoverable path,
    e.g. a stub model or a page flagged for review), ERROR (a failure that abandons
    the unit of work; pass ``exc_info=True`` to attach the traceback).
  - Attach structured context with ``extra={...}`` (e.g. ``lecture_id``, ``stage``);
    those fields are emitted as first-class keys in the JSON format.

Level and format come from ``config.yaml`` ``logging`` (``level``; ``format``:
``json`` | ``console``), so production selects ``json`` — one record per line for
log aggregation — and development selects ``console``, with no code change. Output
goes to stdout; the runtime environment decides where stdout is collected.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from typing import Any, Optional
from urllib.parse import urlsplit

# Standard LogRecord attributes — anything else on a record is treated as an
# ``extra`` field and included in the JSON output.
_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime", "taskName"}

# ``extra`` keys whose values are masked, so a stray secret/credential passed via
# extra={...} can never reach the log sink even by mistake.
_SECRET_KEY = re.compile(r"(key|token|secret|password|authorization|bearer|credential)", re.I)
_MASK = "***"

# HTTP/LLM client libraries pinned below the app level: at DEBUG they log request
# and response bodies (prompts, answers, syllabus text), so a DEBUG app level must
# never turn them into a data-exfiltration channel. Overridable via config.
_THIRD_PARTY = (
    "httpx", "httpcore", "openai", "urllib3", "qdrant_client",
    "langchain", "langchain_openai", "langgraph", "boto3", "botocore", "s3transfer",
)


def redact_uri(uri: Any) -> Any:
    """A log-safe form of a URI: drop the query string (which for a presigned S3
    URL carries the signature/credential) and any userinfo. Returns the input
    unchanged when it has neither, or isn't a parseable URL."""
    if not uri or "://" not in str(uri):
        return uri
    parts = urlsplit(str(uri))
    if not (parts.query or parts.username or parts.password):
        return uri
    return f"{parts.scheme}://{parts.hostname or ''}{parts.path}"


class RequestIdFilter(logging.Filter):
    """Stamp every record with the turn it belongs to.

    Without it only the access log carries ``request_id``, so a turn's language
    decision, retrieval and answer are separate lines with nothing tying them
    together — and a complaint about one wrong answer cannot be traced back to
    the chunks that caused it. The id is read from the same context variable the
    HTTP layer sets, so nothing has to pass it down.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "request_id", None):
            try:
                from system.context import get_request_id
            except ImportError:
                # Ingestion and the CLIs log without the HTTP layer present.
                return True
            current = get_request_id()
            if current:
                record.request_id = current
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per record, with any ``extra`` fields promoted to keys."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = _MASK if _SECRET_KEY.search(key) else value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(
    config: Any = None,
    *,
    level: Optional[str] = None,
    fmt: Optional[str] = None,
) -> logging.Logger:
    """Configure the root logger from ``config.yaml`` ``logging`` (overridable).

    Idempotent: replaces existing handlers so it is safe to call once per process.
    """
    # Precedence: explicit arg > LOG_LEVEL env (handy for one-off runs) > config > INFO.
    lvl = (level or os.environ.get("LOG_LEVEL") or _cfg(config, "logging.level") or "INFO").upper()
    style = (fmt or _cfg(config, "logging.format") or "console").lower()

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())
    if style == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s  %(message)s", "%H:%M:%S"))

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(lvl)
    root.addHandler(handler)

    # Keep third-party HTTP/LLM clients from logging request/response bodies when
    # the app level is DEBUG. Default WARNING; raise via logging.third_party_level.
    tp_level = (_cfg(config, "logging.third_party_level") or "WARNING").upper()
    for name in _THIRD_PARTY:
        logging.getLogger(name).setLevel(tp_level)
    return root


def _cfg(config: Any, dotted: str) -> Optional[str]:
    if config is None:
        return None
    getter = getattr(config, "get", None)
    return getter(dotted) if callable(getter) else None
