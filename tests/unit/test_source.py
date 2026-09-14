"""Source resolution tests: local paths, presigned URLs and S3 URIs. No network."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from ingestion.utils import source as source_module
from ingestion.utils.source import FetchLimits, SourceError, materialise
from shared.config import Config

SIGNED_URL = "https://bucket.s3.eu-central-1.amazonaws.com/uploads/lesson.pdf?X-Amz-Signature=abc"


class _FakeResponse:
    def __init__(self, body: bytes, headers: dict | None = None, error: Exception | None = None):
        self._body = body
        self.headers = headers or {"content-length": str(len(body))}
        self._error = error

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def raise_for_status(self):
        if self._error is not None:
            raise self._error

    def iter_bytes(self, chunk_size):
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start : start + chunk_size]


def _serve(monkeypatch, response: _FakeResponse) -> dict:
    calls: dict = {}

    def _stream(method, url, **kwargs):
        calls.update(method=method, url=url, **kwargs)
        return response

    monkeypatch.setattr(httpx, "stream", _stream)
    return calls


def _config() -> Config:
    return Config.load()


def test_local_file_uri_resolves_without_copying(tmp_path):
    pdf = tmp_path / "lesson.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    with materialise(pdf.resolve().as_uri(), _config()) as path:
        assert Path(path) == pdf

    assert pdf.exists()


def test_plain_path_is_used_as_is(tmp_path):
    pdf = tmp_path / "lesson.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    with materialise(str(pdf), _config()) as path:
        assert path == str(pdf)


def test_missing_local_file_is_reported_clearly(tmp_path):
    missing = str(tmp_path / "absent.pdf")
    with pytest.raises(SourceError, match="not found"), materialise(missing, _config()):
        pass


def test_signed_url_is_downloaded_and_cleaned_up(monkeypatch):
    calls = _serve(monkeypatch, _FakeResponse(b"%PDF-1.4 signed body"))

    with materialise(SIGNED_URL, _config()) as path:
        downloaded = Path(path)
        assert downloaded.read_bytes() == b"%PDF-1.4 signed body"
        assert downloaded.suffix == ".pdf"

    assert not downloaded.exists()
    assert calls["url"] == SIGNED_URL
    assert calls["follow_redirects"] is True


def test_download_failure_raises_without_leaking_the_signature(monkeypatch):
    _serve(
        monkeypatch,
        _FakeResponse(b"", error=httpx.HTTPStatusError("403", request=None, response=None)),
    )

    with pytest.raises(SourceError) as excinfo, materialise(SIGNED_URL, _config()):
        pass

    assert "X-Amz-Signature" not in str(excinfo.value)


def test_empty_body_is_an_error(monkeypatch):
    _serve(monkeypatch, _FakeResponse(b"", headers={}))

    with pytest.raises(SourceError, match="empty body"), materialise(SIGNED_URL, _config()):
        pass


def test_declared_size_over_the_limit_is_refused_before_reading(monkeypatch):
    body = b"x" * 2048
    _serve(monkeypatch, _FakeResponse(body, headers={"content-length": str(len(body))}))
    monkeypatch.setattr(
        FetchLimits, "from_config", classmethod(lambda cls, config: cls(max_bytes=1024))
    )

    with pytest.raises(SourceError, match="ingestion limit"), materialise(SIGNED_URL, _config()):
        pass


def test_undeclared_oversize_body_is_refused_while_streaming(monkeypatch):
    _serve(monkeypatch, _FakeResponse(b"x" * 2048, headers={}))
    monkeypatch.setattr(
        FetchLimits,
        "from_config",
        classmethod(lambda cls, config: cls(max_bytes=1024, chunk_bytes=256)),
    )

    with pytest.raises(SourceError, match="ingestion limit"), materialise(SIGNED_URL, _config()):
        pass


class _FakeS3:
    def __init__(self, size: int = 8, body: bytes = b"%PDF-1.4"):
        self.size = size
        self.body = body
        self.downloads: list[tuple[str, str]] = []

    def head_object(self, Bucket, Key):
        return {"ContentLength": self.size}

    def download_file(self, bucket, key, destination):
        self.downloads.append((bucket, key))
        Path(destination).write_bytes(self.body)


def test_s3_uri_is_fetched_to_a_temporary_file(monkeypatch):
    client = _FakeS3()
    monkeypatch.setattr(source_module, "s3_client", lambda config: client)

    with materialise("s3://edu-ai-media/uploads/lesson.pdf", _config()) as path:
        downloaded = Path(path)
        assert downloaded.read_bytes() == b"%PDF-1.4"

    assert not downloaded.exists()
    assert client.downloads == [("edu-ai-media", "uploads/lesson.pdf")]


def test_s3_object_over_the_limit_is_never_downloaded(monkeypatch):
    client = _FakeS3(size=10_000)
    monkeypatch.setattr(source_module, "s3_client", lambda config: client)
    monkeypatch.setattr(
        FetchLimits, "from_config", classmethod(lambda cls, config: cls(max_bytes=1024))
    )

    uri = "s3://edu-ai-media/uploads/lesson.pdf"
    with pytest.raises(SourceError, match="ingestion limit"), materialise(uri, _config()):
        pass

    assert client.downloads == []


def test_malformed_s3_uri_is_rejected(monkeypatch):
    monkeypatch.setattr(source_module, "s3_client", lambda config: _FakeS3())

    with pytest.raises(SourceError, match="malformed"), materialise("s3://edu-ai-media", _config()):
        pass


def test_limits_come_from_config():
    limits = FetchLimits.from_config(_config())

    assert limits.timeout_seconds > 0
    assert limits.max_bytes > 0
    assert limits.chunk_bytes > 0
