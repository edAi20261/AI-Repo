"""Object-storage upload, shared by ingestion (figure crops) and the agent (render
artifacts). Two backends behind one interface: S3 when ``S3_BUCKET`` is set, else a local
filesystem fallback returning a ``file://`` URI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .config import Config


class Uploader(ABC):
    @abstractmethod
    def upload(self, data: bytes, key: str, content_type: str = "image/png") -> str:
        """Store ``data`` at ``key`` and return a stable URI."""

    @abstractmethod
    def delete_prefix(self, prefix: str) -> int:
        """Delete every object under ``prefix`` and return the count removed."""


class S3Uploader(Uploader):
    def __init__(self, bucket: str, region: str | None, endpoint_url: str | None) -> None:
        import boto3  # lazy

        self._bucket = bucket
        self._client = boto3.client(
            "s3", region_name=region, endpoint_url=endpoint_url or None
        )

    def upload(self, data: bytes, key: str, content_type: str = "image/png") -> str:
        self._client.put_object(
            Bucket=self._bucket, Key=key, Body=data, ContentType=content_type
        )
        return f"s3://{self._bucket}/{key}"

    def delete_prefix(self, prefix: str) -> int:
        paginator = self._client.get_paginator("list_objects_v2")
        removed = 0
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            objects = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if objects:
                self._client.delete_objects(
                    Bucket=self._bucket, Delete={"Objects": objects})
                removed += len(objects)
        return removed


class LocalUploader(Uploader):
    """Filesystem fallback when S3 is unconfigured. Returns a ``file://`` URI."""

    def __init__(self, base_dir: str) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    def upload(self, data: bytes, key: str, content_type: str = "image/png") -> str:
        dest = self._base / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest.resolve().as_uri()

    def delete_prefix(self, prefix: str) -> int:
        import shutil

        target = self._base / prefix
        if not target.exists():
            return 0
        removed = sum(1 for p in target.rglob("*") if p.is_file())
        shutil.rmtree(target, ignore_errors=True)
        return removed


def get_uploader(config: Config) -> Uploader:
    """Return the configured uploader; local fallback when S3 is unset."""
    bucket = config.secret("S3_BUCKET")
    if bucket:
        return S3Uploader(
            bucket,
            region=config.secret("S3_REGION"),
            endpoint_url=config.secret("S3_ENDPOINT_URL"),
        )
    base = config.secret("LOCAL_ARTIFACT_DIR", ".ingestion_artifacts")
    return LocalUploader(base)
