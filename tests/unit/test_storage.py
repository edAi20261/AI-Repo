"""Shared object-storage tests. No AWS or network: S3 is exercised through a fake client."""

from __future__ import annotations

import pytest
from botocore.exceptions import ClientError

from shared.config import Config
from shared.storage import LocalUploader, S3Uploader, StorageError, get_uploader, scoped_key

PREFIX = "ai-artifacts"


class _FakeS3:
    def __init__(self, objects=None, delete_errors=None, fail_with=None):
        self.objects = list(objects or [])
        self.delete_errors = set(delete_errors or [])
        self.fail_with = fail_with
        self.puts = []
        self.listed_prefixes = []

    def put_object(self, **kwargs):
        self.puts.append(kwargs)

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self

    def paginate(self, Bucket, Prefix):
        if self.fail_with is not None:
            raise self.fail_with
        self.listed_prefixes.append(Prefix)
        yield {"Contents": [{"Key": key} for key in self.objects if key.startswith(Prefix)]}

    def delete_objects(self, Bucket, Delete):
        keys = [item["Key"] for item in Delete["Objects"]]
        refused = [key for key in keys if key in self.delete_errors]
        return {
            "Deleted": [{"Key": key} for key in keys if key not in self.delete_errors],
            "Errors": [{"Key": key, "Code": "AccessDenied"} for key in refused],
        }


@pytest.mark.parametrize(
    ("prefix", "key", "expected"),
    [
        ("ai-artifacts", "figures/l1/p001.png", "ai-artifacts/figures/l1/p001.png"),
        ("ai-artifacts/", "figures/l1/", "ai-artifacts/figures/l1/"),
        ("/ai-artifacts/", "/artifacts/u1/chart/a.png", "ai-artifacts/artifacts/u1/chart/a.png"),
        ("", "figures/l1/p001.png", "figures/l1/p001.png"),
    ],
)
def test_scoped_key_places_logical_key_under_prefix(prefix, key, expected):
    assert scoped_key(prefix, key) == expected


def test_local_uploader_writes_under_prefix(tmp_path):
    uri = LocalUploader(str(tmp_path), key_prefix=PREFIX).upload(b"png", "figures/l1/p001.png")

    stored = tmp_path / PREFIX / "figures" / "l1" / "p001.png"
    assert stored.read_bytes() == b"png"
    assert uri == stored.resolve().as_uri()


def test_local_uploader_delete_stays_inside_prefix(tmp_path):
    scoped = LocalUploader(str(tmp_path), key_prefix=PREFIX)
    unscoped = LocalUploader(str(tmp_path))
    scoped.upload(b"a", "figures/l1/p001.png")
    scoped.upload(b"b", "figures/l10/p001.png")
    unscoped.upload(b"c", "figures/l1/p001.png")

    assert scoped.delete_prefix("figures/l1/") == 1
    assert (tmp_path / PREFIX / "figures" / "l10" / "p001.png").exists()
    assert (tmp_path / "figures" / "l1" / "p001.png").exists()


def test_delete_refuses_an_empty_prefix(tmp_path):
    uploader = LocalUploader(str(tmp_path), key_prefix=PREFIX)
    uploader.upload(b"a", "figures/l1/p001.png")

    with pytest.raises(ValueError):
        uploader.delete_prefix("/")
    assert (tmp_path / PREFIX / "figures" / "l1" / "p001.png").exists()


def test_s3_uploader_writes_under_prefix():
    client = _FakeS3()
    uri = S3Uploader("media", key_prefix=PREFIX, client=client).upload(
        b"svg", "artifacts/u1/mindmap/a.svg", "image/svg+xml"
    )

    assert client.puts == [{
        "Bucket": "media",
        "Key": "ai-artifacts/artifacts/u1/mindmap/a.svg",
        "Body": b"svg",
        "ContentType": "image/svg+xml",
    }]
    assert uri == "s3://media/ai-artifacts/artifacts/u1/mindmap/a.svg"


def test_s3_uploader_deletes_only_inside_prefix():
    client = _FakeS3(objects=[
        "ai-artifacts/figures/l1/p001.png",
        "ai-artifacts/figures/l1/p002.png",
        "ai-artifacts/figures/l10/p001.png",
        "figures/l1/p001.png",
    ])

    removed = S3Uploader("media", key_prefix=PREFIX, client=client).delete_prefix("figures/l1/")

    assert removed == 2
    assert client.listed_prefixes == ["ai-artifacts/figures/l1/"]


def test_s3_uploader_raises_when_objects_are_not_deleted():
    client = _FakeS3(
        objects=["ai-artifacts/figures/l1/p001.png"],
        delete_errors=["ai-artifacts/figures/l1/p001.png"],
    )

    with pytest.raises(StorageError, match="AccessDenied"):
        S3Uploader("media", key_prefix=PREFIX, client=client).delete_prefix("figures/l1/")


def test_s3_uploader_wraps_client_errors():
    denied = ClientError({"Error": {"Code": "AccessDenied", "Message": "denied"}}, "ListObjectsV2")
    client = _FakeS3(fail_with=denied)

    with pytest.raises(StorageError):
        S3Uploader("media", key_prefix=PREFIX, client=client).delete_prefix("figures/l1/")


def test_get_uploader_applies_configured_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "")
    monkeypatch.setenv("LOCAL_ARTIFACT_DIR", str(tmp_path))
    config = Config({"storage": {"key_prefix": PREFIX}}, "test")

    get_uploader(config).upload(b"png", "figures/l1/p001.png")

    assert (tmp_path / PREFIX / "figures" / "l1" / "p001.png").exists()


def test_shipped_config_scopes_storage_to_the_iam_prefix(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")

    assert Config.load().get("storage.key_prefix") == PREFIX
