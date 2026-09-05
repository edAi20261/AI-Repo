"""Logging-safety tests: secret masking, URI redaction, third-party clamp."""

from __future__ import annotations

import json
import logging

from shared.logging import JsonFormatter, redact_uri, setup_logging


class _Cfg:
    def __init__(self, values):
        self._v = values

    def get(self, dotted, default=None):
        return self._v.get(dotted, default)


def _format(**extra) -> dict:
    record = logging.LogRecord(
        "t", logging.INFO, __file__, 1, "msg", None, None)
    for k, v in extra.items():
        setattr(record, k, v)
    return json.loads(JsonFormatter().format(record))


def test_json_formatter_masks_secret_like_extra_keys():
    out = _format(lecture_id="lec1", api_key="sk-secret", auth_token="abc")
    assert out["lecture_id"] == "lec1"          # ordinary field kept
    assert out["api_key"] == "***"              # masked
    assert out["auth_token"] == "***"           # masked


def test_redact_uri_strips_presigned_query_and_userinfo():
    signed = "https://bucket.s3.amazonaws.com/lec.pdf?X-Amz-Signature=deadbeef"
    assert redact_uri(signed) == "https://bucket.s3.amazonaws.com/lec.pdf"
    assert redact_uri("https://u:p@host/x") == "https://host/x"
    # No query/userinfo: unchanged. Non-URLs and empties: unchanged.
    assert redact_uri("s3://bucket/lec.pdf") == "s3://bucket/lec.pdf"
    assert redact_uri("/local/path.pdf") == "/local/path.pdf"
    assert redact_uri(None) is None


def test_setup_logging_clamps_third_party_loggers():
    setup_logging(_Cfg({"logging.level": "DEBUG", "logging.format": "console"}))
    assert logging.getLogger().level == logging.DEBUG          # app level honored
    assert logging.getLogger("openai").level == logging.WARNING  # client clamped
    assert logging.getLogger("httpx").level == logging.WARNING
