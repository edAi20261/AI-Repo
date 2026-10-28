"""Asking a model to read an image — and which gateway that bills to.

The key was named in the code: OPENROUTER_API_KEY, wherever vision pointed.
Ingestion is the expensive half of this service and has to be able to bill
somewhere other than the half that answers students.
"""

from __future__ import annotations

from agent.config import get_config


# --- which gateway, and whose key -------------------------------------------

def test_the_key_follows_the_provider_vision_is_pointed_at(monkeypatch):
    """Ingestion is the expensive half and has to be able to bill somewhere
    other than the half that answers students. The key used to be the
    OpenRouter one, named in the code."""
    from shared.vision import _api_key

    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    config = get_config()
    config.override("vision.provider", "google")
    assert _api_key(config) == "google-key"
    config.override("vision.provider", "openrouter")
    assert _api_key(config) == "openrouter-key"


def test_a_deployment_can_name_its_own_variable(monkeypatch):
    from shared.vision import _api_key

    monkeypatch.setenv("VERTEX_TOKEN", "vertex-token")
    config = get_config()
    config.override("vision.provider", "google")
    config.override("vision.api_key_env", "VERTEX_TOKEN")
    assert _api_key(config) == "vertex-token"
    config.override("vision.api_key_env", None)


def test_an_unknown_provider_still_reaches_for_a_key(monkeypatch):
    """Falling back to no key at all would send an unauthenticated request and
    read as a model failure rather than a configuration one."""
    from shared.vision import _api_key

    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    config = get_config()
    config.override("vision.provider", "something-new")
    assert _api_key(config) == "openrouter-key"
    config.override("vision.provider", "openrouter")


# --- waiting out a quota, rather than losing the page to it ------------------

class _Reply:
    """Enough of an ``httpx.Response`` for the transport to read."""

    def __init__(self, status: int, content: str = "") -> None:
        self.status_code = status
        self.text = content
        self._content = content

    def raise_for_status(self):
        import httpx

        if self.status_code >= 400:
            raise httpx.HTTPStatusError("rejected", request=None, response=self)
        return self

    def json(self):
        return {"usage": {}, "choices": [{"message": {"content": self._content}}]}


def _impatient_config(monkeypatch):
    """The shipped gateway, with the waiting taken out so a test is not slow."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "a-key")
    config = get_config()
    config.override("vision.provider", "openrouter")
    config.override("vision.retry_backoff_seconds", 0.0)
    return config


def _replies(monkeypatch, *replies):
    """``httpx.post`` answering with each reply in turn. Returns the call log."""
    calls = []
    remaining = list(replies)

    def post(url, **kwargs):
        calls.append(url)
        return remaining.pop(0) if remaining else _Reply(200, "{}")

    monkeypatch.setattr("httpx.post", post)
    return calls


def test_a_rate_limited_page_is_waited_out_rather_than_lost(monkeypatch):
    """Three ingest tasks at eight pages each put twenty-four requests against
    one Vertex quota, and 429s came back every few minutes. With no retry the
    page dropped to the fallback model and, when that was limited too, was
    recorded unreadable — and a page lost to a quota counts against
    needs_review_ratio exactly as a blurred one does, so a quiet few minutes of
    rate limiting sent a whole paid-for textbook to manual review."""
    from shared.vision import _post

    calls = _replies(monkeypatch, _Reply(429), _Reply(429), _Reply(200, "the page"))

    assert _post(b"png", "read it", "a-model", _impatient_config(monkeypatch), label="t") == "the page"
    assert len(calls) == 3


def test_a_rejection_that_will_never_change_is_not_retried(monkeypatch):
    """A bad key answers the same way forever. Retrying it only spends the time
    and, on a paid gateway, the money again."""
    from shared.vision import _post

    calls = _replies(monkeypatch, _Reply(401), _Reply(200, "unreachable"))

    assert _post(b"png", "read it", "a-model", _impatient_config(monkeypatch), label="t") is None
    assert len(calls) == 1


def test_a_page_the_gateway_keeps_refusing_gives_up_loudly(monkeypatch, caplog):
    """Bounded, because a quota that is exhausted for the day would otherwise
    hold the task — and Fargate bills for a container that is only sleeping."""
    import logging

    from shared.vision import _post

    config = _impatient_config(monkeypatch)
    config.override("vision.retry_attempts", 3)
    calls = _replies(monkeypatch, _Reply(429), _Reply(429), _Reply(429), _Reply(429))

    with caplog.at_level(logging.WARNING):
        assert _post(b"png", "read it", "a-model", config, label="t") is None

    assert len(calls) == 3
    assert any("gave up waiting out the gateway" in r.message for r in caplog.records)
    config.override("vision.retry_attempts", 4)


def test_a_timeout_is_the_gateway_being_slow_not_the_page_being_unreadable(monkeypatch):
    """A page that times out once and reads on the next attempt used to be
    indexed as unreadable."""
    import httpx

    from shared.vision import _post

    answers = [httpx.ReadTimeout("too slow"), _Reply(200, "the page")]
    calls = []

    def post(url, **kwargs):
        calls.append(url)
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr("httpx.post", post)

    assert _post(b"png", "read it", "a-model", _impatient_config(monkeypatch), label="t") == "the page"
    assert len(calls) == 2


# --- a gateway is its address as well as its key ------------------------------

def _recorded_post(monkeypatch):
    """``httpx.post`` that answers once and keeps what it was sent."""
    sent = {}

    def post(url, **kwargs):
        sent["url"], sent["body"] = url, kwargs["json"]
        return _Reply(200, "the page")

    monkeypatch.setattr("httpx.post", post)
    return sent


def test_a_page_for_openai_is_sent_to_openai(monkeypatch):
    """Every gateway but Vertex was posted to OpenRouter's address, so pointing
    vision at OpenAI sent an OpenAI key to OpenRouter and every page of the
    book was rejected."""
    from shared.vision import _post

    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    sent = _recorded_post(monkeypatch)
    config = get_config()
    config.override("vision.provider", "openai")
    try:
        _post(b"png", "read it", "a-model", config, label="t")
    finally:
        config.override("vision.provider", "openrouter")

    assert sent["url"].startswith("https://api.openai.com/v1/")


def test_only_openrouter_is_asked_for_its_own_usage_field(monkeypatch):
    """``usage: {include: true}`` is OpenRouter's extension. OpenAI rejects a
    request carrying an argument it does not know, so the page is lost to a
    field that was only ever there for cost accounting."""
    from shared.vision import _post

    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    sent = _recorded_post(monkeypatch)
    config = get_config()
    config.override("vision.provider", "openai")
    try:
        _post(b"png", "read it", "a-model", config, label="t")
    finally:
        config.override("vision.provider", "openrouter")
    assert "usage" not in sent["body"]

    _post(b"png", "read it", "a-model", _impatient_config(monkeypatch), label="t")
    assert sent["body"]["usage"] == {"include": True}


# --- an image goes up in a format every gateway reads ------------------------

def _jpeg2000() -> bytes:
    import fitz

    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 8, 8), False)
    pix.clear_with(200)
    return pix.pil_tobytes(format="JPEG2000")


def test_a_figure_in_jpeg_2000_is_sent_as_an_image_openai_reads(monkeypatch):
    """Textbook figures are embedded as JPEG 2000. They were posted as they came,
    labelled PNG; Gemini read them, and OpenAI answered "You uploaded an
    unsupported image" for every one, so 2,065 of a math book's figures went
    without a caption."""
    import base64

    from shared.vision import _post

    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    sent = _recorded_post(monkeypatch)
    config = get_config()
    config.override("vision.provider", "openai")
    try:
        _post(_jpeg2000(), "caption it", "a-model", config, label="t")
    finally:
        config.override("vision.provider", "openrouter")

    url = sent["body"]["messages"][0]["content"][1]["image_url"]["url"]
    header, encoded = url.split(",", 1)
    assert header == "data:image/png;base64"
    assert base64.b64decode(encoded)[:8] == b"\x89PNG\r\n\x1a\n"


def test_a_photo_keeps_its_own_format(monkeypatch):
    """A student's homework photo is a JPEG; calling it PNG mislabels it, and
    re-encoding it as PNG would multiply its size."""
    import io

    from PIL import Image

    from shared.vision import _post

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format="JPEG")
    sent = _recorded_post(monkeypatch)
    _post(buffer.getvalue(), "read it", "a-model", _impatient_config(monkeypatch), label="t")

    url = sent["body"]["messages"][0]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
