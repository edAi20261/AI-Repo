"""Resolve a source URI to a readable local path.

The platform hands the pipeline a presigned ``https`` URL; fixtures, the CLI and
re-runs from object storage use ``file://``, a bare path or ``s3://``. Remote
sources are streamed to a temporary file that exists only for the run, so the
rest of the pipeline always works with a local file.

Shared because a lesson PDF and the material a teacher uploads about how they
teach arrive the same way. An endpoint that takes a URI from a caller uses
``materialise_remote``: a bare path there would let a request read any file the
service can.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from shared.errors import UnprocessableError
from urllib.request import url2pathname

from shared.config import Config
from shared.logging import redact_uri
from shared.storage import s3_client

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_BYTES = 200 * 1024 * 1024
DEFAULT_CHUNK_BYTES = 1024 * 1024


class SourceError(UnprocessableError):
    """The source URI could not be resolved to a readable local file."""


@dataclass(frozen=True)
class FetchLimits:
    """Bounds applied to every remote fetch."""

    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_bytes: int = DEFAULT_MAX_BYTES
    chunk_bytes: int = DEFAULT_CHUNK_BYTES

    @classmethod
    def from_config(cls, config: Config) -> FetchLimits:
        return cls(
            timeout_seconds=float(
                config.get("ingestion.source.timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
            ),
            max_bytes=int(config.get("ingestion.source.max_bytes", DEFAULT_MAX_BYTES)),
            chunk_bytes=int(config.get("ingestion.source.chunk_bytes", DEFAULT_CHUNK_BYTES)),
        )


@contextmanager
def materialise(source_uri: str, config: Config) -> Iterator[str]:
    """Yield a local path for ``source_uri``, removing any temporary copy afterwards."""
    fetch = _FETCHERS.get(urlparse(source_uri).scheme.lower())
    if fetch is None:
        yield local_path(source_uri)
        return

    limits = FetchLimits.from_config(config)
    handle, destination = tempfile.mkstemp(prefix="ingestion-source-", suffix=_suffix(source_uri))
    os.close(handle)
    try:
        size = fetch(source_uri, destination, limits, config)
        logger.info(
            "source fetched",
            extra={"source_uri": redact_uri(source_uri), "bytes": size},
        )
        yield destination
    finally:
        Path(destination).unlink(missing_ok=True)


@contextmanager
def materialise_remote(source_uri: str, config: Config) -> Iterator[str]:
    """``materialise`` for a URI a caller supplied: remote schemes only."""
    if urlparse(source_uri).scheme.lower() not in _FETCHERS:
        raise SourceError("a source must be an https or s3 URI")
    with materialise(source_uri, config) as path:
        yield path


def local_path(source_uri: str) -> str:
    """Path of an already-local source: a ``file://`` URI or a plain path."""
    path = (
        url2pathname(urlparse(source_uri).path)
        if source_uri.startswith("file://")
        else source_uri
    )
    if not Path(path).is_file():
        raise SourceError(f"source file not found: {path}")
    return path


def _fetch_http(source_uri: str, destination: str, limits: FetchLimits, _: Config) -> int:
    import httpx

    written = 0
    try:
        with httpx.stream(
            "GET", source_uri, timeout=limits.timeout_seconds, follow_redirects=True
        ) as response:
            response.raise_for_status()
            _guard_size(_declared_size(response.headers), limits, source_uri)
            with open(destination, "wb") as sink:
                for block in response.iter_bytes(limits.chunk_bytes):
                    written += len(block)
                    _guard_size(written, limits, source_uri)
                    sink.write(block)
    except httpx.HTTPError as exc:
        raise SourceError(
            f"downloading {redact_uri(source_uri)} failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not written:
        raise SourceError(f"{redact_uri(source_uri)} returned an empty body")
    return written


def _fetch_s3(source_uri: str, destination: str, limits: FetchLimits, config: Config) -> int:
    from botocore.exceptions import BotoCoreError, ClientError

    parsed = urlparse(source_uri)
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    if not bucket or not key:
        raise SourceError(f"malformed S3 URI: {source_uri}")

    client = s3_client(config)
    try:
        head = client.head_object(Bucket=bucket, Key=key)
        _guard_size(int(head.get("ContentLength") or 0), limits, source_uri)
        client.download_file(bucket, key, destination)
    except (BotoCoreError, ClientError) as exc:
        raise SourceError(
            f"fetching s3://{bucket}/{key} failed: {type(exc).__name__}: {exc}"
        ) from exc
    return Path(destination).stat().st_size


_FETCHERS: dict[str, Callable[[str, str, FetchLimits, Config], int]] = {
    "http": _fetch_http,
    "https": _fetch_http,
    "s3": _fetch_s3,
}


def _declared_size(headers: Any) -> int:
    try:
        return int(headers.get("content-length") or 0)
    except (TypeError, ValueError):
        return 0


def _guard_size(size: int, limits: FetchLimits, source_uri: str) -> None:
    if limits.max_bytes and size > limits.max_bytes:
        raise SourceError(
            f"{redact_uri(source_uri)} is {size} bytes, above the "
            f"{limits.max_bytes} byte ingestion limit"
        )


def _suffix(source_uri: str) -> str:
    suffix = Path(urlparse(source_uri).path).suffix.lower()
    return suffix if suffix in _KNOWN_SUFFIXES else ".pdf"


_KNOWN_SUFFIXES = frozenset({
    ".pdf", ".png", ".jpg", ".jpeg",
    ".txt", ".md", ".json", ".vtt", ".srt",
})
