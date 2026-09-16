"""Object storage shared by ingestion (figure crops) and the agent (render artifacts).

Callers work with logical keys. Every uploader scopes them under the configured
``storage.key_prefix``, so writes and deletes stay inside the prefix the service's IAM
role is granted. S3 is used when ``S3_BUCKET`` is set; otherwise a local filesystem
fallback returns ``file://`` URIs.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .config import Config
from .errors import ServiceUnavailableError

DEFAULT_LOCAL_DIR = ".ingestion_artifacts"


class StorageError(ServiceUnavailableError):
    """An object-storage operation did not fully succeed."""


def scoped_key(prefix: str, key: str) -> str:
    """Place a logical key under ``prefix``, tolerating stray slashes on either side."""
    root = prefix.strip("/")
    leaf = key.lstrip("/")
    return f"{root}/{leaf}" if root else leaf


class Uploader(ABC):
    """Stores objects under a fixed key prefix; subclasses implement the raw operations."""

    def __init__(self, key_prefix: str = "") -> None:
        self._key_prefix = key_prefix

    def upload(self, data: bytes, key: str, content_type: str = "image/png") -> str:
        """Store ``data`` at the logical ``key`` and return a stable URI."""
        return self._put(data, scoped_key(self._key_prefix, key), content_type)

    def delete_prefix(self, prefix: str) -> int:
        """Delete every object under the logical ``prefix`` and return the count removed.

        Raises :class:`StorageError` when any object could not be deleted, and
        ``ValueError`` for an empty prefix, which would address the whole namespace.
        """
        if not prefix.strip("/"):
            raise ValueError("refusing to delete with an empty prefix")
        return self._delete(scoped_key(self._key_prefix, prefix))

    @abstractmethod
    def _put(self, data: bytes, key: str, content_type: str) -> str: ...

    @abstractmethod
    def _delete(self, prefix: str) -> int: ...


class S3Uploader(Uploader):
    def __init__(
        self,
        bucket: str,
        *,
        region: str | None = None,
        endpoint_url: str | None = None,
        key_prefix: str = "",
        client: Any = None,
    ) -> None:
        super().__init__(key_prefix)
        self._bucket = bucket
        self._client = client if client is not None else _s3_client(region, endpoint_url)

    def _put(self, data: bytes, key: str, content_type: str) -> str:
        self._client.put_object(
            Bucket=self._bucket, Key=key, Body=data, ContentType=content_type
        )
        return f"s3://{self._bucket}/{key}"

    def _delete(self, prefix: str) -> int:
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            pages = self._client.get_paginator("list_objects_v2").paginate(
                Bucket=self._bucket, Prefix=prefix
            )
            return sum(self._delete_batch(page.get("Contents") or [], prefix) for page in pages)
        except (BotoCoreError, ClientError) as exc:
            raise StorageError(f"deleting objects under {prefix!r} failed: {exc}") from exc

    def _delete_batch(self, contents: list[dict[str, Any]], prefix: str) -> int:
        if not contents:
            return 0
        objects = [{"Key": item["Key"]} for item in contents]
        response = self._client.delete_objects(Bucket=self._bucket, Delete={"Objects": objects})
        errors = response.get("Errors") or []
        if errors:
            raise StorageError(
                f"{len(errors)} of {len(objects)} objects under {prefix!r} were not deleted "
                f"({errors[0].get('Code', 'unknown error')})"
            )
        return len(response.get("Deleted") or [])


class LocalUploader(Uploader):
    """Filesystem fallback when S3 is unconfigured. Returns ``file://`` URIs."""

    def __init__(self, base_dir: str, *, key_prefix: str = "") -> None:
        super().__init__(key_prefix)
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    def _put(self, data: bytes, key: str, content_type: str) -> str:
        destination = self._base / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return destination.resolve().as_uri()

    def _delete(self, prefix: str) -> int:
        target = self._base / prefix
        if not target.exists():
            return 0
        removed = sum(1 for path in target.rglob("*") if path.is_file())
        try:
            shutil.rmtree(target)
        except OSError as exc:
            raise StorageError(f"deleting files under {target} failed: {exc}") from exc
        return removed


def get_uploader(config: Config) -> Uploader:
    """The configured uploader: S3 when ``S3_BUCKET`` is set, else the local fallback."""
    key_prefix = config.get("storage.key_prefix", "ai-artifacts") or ""
    bucket = config.secret("S3_BUCKET")
    if bucket:
        return S3Uploader(
            bucket,
            region=config.secret("S3_REGION"),
            endpoint_url=config.secret("S3_ENDPOINT_URL"),
            key_prefix=key_prefix,
        )
    return LocalUploader(
        config.secret("LOCAL_ARTIFACT_DIR", DEFAULT_LOCAL_DIR), key_prefix=key_prefix
    )


def s3_client(config: Config) -> Any:
    """An S3 client built from the service's configured region and endpoint."""
    return _s3_client(config.secret("S3_REGION"), config.secret("S3_ENDPOINT_URL"))


def _s3_client(region: str | None, endpoint_url: str | None) -> Any:
    import boto3

    return boto3.client("s3", region_name=region, endpoint_url=endpoint_url or None)
