"""Asking a model to read an image and answer in a schema.

Both halves need this and for different reasons: ingestion reads a textbook page
it is indexing, and the tutor reads a homework photo a student just took. What
they share is the transport — an image and a prompt go up, structured data comes
back — while the prompt and the schema belong to whoever is asking.

That split is why this is not the ingestion vision client moved wholesale. A page
transcription and a homework capture are different questions; posting an image is
not.
"""

from __future__ import annotations

import base64
import logging
import random
import time
from typing import Optional, TypeVar

from pydantic import BaseModel

from .config import Config
from .metering import record_usage
from .structured import recover

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT = 180
# Gemini counts its hidden thinking against this and spends thousands of tokens
# before writing a word, so a budget sized for a model without it truncates the
# answer mid-JSON. Must match config.yaml; test_config_contract enforces that.
DEFAULT_MAX_TOKENS = 16384

# A rejection that says "later", not "no". Three ingest tasks at eight pages
# each put twenty-four requests against one Vertex quota, and a 429 came back
# every few minutes; without a retry the page fell to the fallback model, and
# when that was rate-limited too the page was recorded unreadable. A page lost
# to a quota counts against ingestion.needs_review_ratio exactly as a blurred
# one does, so at 0.05 a few minutes of rate limiting could send a whole
# textbook to manual review after it had been paid for in full.
TRANSIENT_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})
# Must match config.yaml; test_config_contract enforces that.
DEFAULT_RETRY_ATTEMPTS = 4
DEFAULT_RETRY_BACKOFF_SECONDS = 4.0

# Which environment variable holds the key, per gateway. The transport is the
# same everywhere — an OpenAI-shaped /chat/completions with a bearer token —
# so the gateway is a base URL and a key name, not a code path. Google's
# OpenAI-compatible endpoint is the reason this is configurable at all:
# ingestion is the expensive half, and it has to be able to bill somewhere
# other than the half that answers students.
DEFAULT_KEY_ENV = "OPENROUTER_API_KEY"
PROVIDER_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
}

# google  -> the Gemini API, billed from AI Studio prepay credit
# vertex  -> the same models, billed to the Cloud billing account instead,
#            which is where a project's trial credit actually lives. It is the
#            only gateway here that will not take a static key.
VERTEX = "vertex"


def _api_key(config: Config) -> Optional[str]:
    """The bearer token for whichever gateway vision is pointed at.

    Vertex is the one gateway that will not take a static key: it wants an
    OAuth token minted from a service account, and those expire in an hour —
    well inside the time it takes to read one book. ``shared.gcp`` mints and
    caches it, so a three-hundred-page book pays for one token, not three
    hundred.

    ``vision.api_key_env`` wins over everything, so a deployment can name its
    own variable without this having to know about it.
    """
    provider = (config.get("vision.provider") or "openrouter").lower()
    if provider == "vertex":
        from .gcp import access_token, credential_source

        path = credential_source(config)
        if not path:
            raise VisionUnavailable(
                "vision.provider is vertex but no service account path is set; "
                "point GOOGLE_APPLICATION_CREDENTIALS at the JSON key.")
        return access_token(path)
    named = config.get("vision.api_key_env")
    if named:
        return config.secret(named)
    return config.secret(PROVIDER_KEY_ENV.get(provider, DEFAULT_KEY_ENV))


def _base_url(config: Config) -> str:
    """Where to post. Vertex's address carries the project and region in it, so
    it is built rather than configured — one fewer thing to get wrong by hand."""
    configured = config.get("vision.base_url")
    if (config.get("vision.provider") or "").lower() != VERTEX:
        return configured or DEFAULT_BASE_URL
    if configured:
        return configured
    from .gcp import vertex_base_url

    project = _project_from_credentials(config)
    if not project:
        raise VisionUnavailable(
            "vision.provider is vertex but no project is known; set vision.project "
            "or point the credentials at a service-account JSON that names one.")
    return vertex_base_url(project, config.get("vision.location") or "us-central1")


def _project_from_credentials(config: Config) -> Optional[str]:
    """The project page reading is billed to; the credential's own unless moved."""
    from .gcp import billed_project, credential_source

    return billed_project(config, credential_source(config))


class VisionUnavailable(RuntimeError):
    """Every model tried failed to produce a usable answer."""


def read_image(
    image_png: bytes,
    prompt: str,
    schema: type[M],
    *,
    config: Config,
    label: str = "vision",
) -> Optional[M]:
    """The image, read into ``schema``, or ``None`` when no model could.

    The primary model is tried first and the configured fallback only if its
    answer cannot be validated — a second call is worth paying for when the first
    produced nothing, and wasted otherwise.
    """
    models = [config.get("vision.model")]
    fallback = config.get("vision.fallback_model")
    if fallback:
        models.append(fallback)

    for model in models:
        raw = _post(image_png, prompt, model, config, label=label)
        if raw is None:
            continue
        parsed = recover(raw, schema)
        if parsed is not None:
            return parsed
        logger.warning("vision reply did not fit the schema", extra={
            "label": label, "model": model, "schema": schema.__name__})

    logger.error("vision failed on every model", extra={
        "label": label, "models": models, "schema": schema.__name__})
    return None


def _post(
    image_png: bytes, prompt: str, model: str, config: Config, *, label: str
) -> Optional[str]:
    """One page, read by one model, retried while the gateway says "later".

    A failure is logged with its reason and returns ``None``. The reason is the
    point: this used to be a bare ``except Exception: return None``, so an
    expired API key, a rate limit and a genuinely unreadable page were
    indistinguishable — operators saw "needs review" on a whole textbook and had
    nothing to act on.

    A quota rejection is the one failure that is not about this page at all, so
    it is waited out rather than counted. Everything else — a bad key, a
    malformed request, a page the model will not read — fails on the first
    answer, because retrying it only spends the same money again.
    """
    attempts = int(config.get("vision.retry_attempts", DEFAULT_RETRY_ATTEMPTS))
    backoff = float(config.get("vision.retry_backoff_seconds",
                               DEFAULT_RETRY_BACKOFF_SECONDS))
    for attempt in range(1, max(attempts, 1) + 1):
        again, body = _attempt(image_png, prompt, model, config, label=label)
        if not again:
            return body
        if attempt >= attempts:
            logger.warning("vision gave up waiting out the gateway", extra={
                "label": label, "model": model, "attempts": attempt})
            return None
        # Jittered, because eight workers rate-limited by the same quota at the
        # same moment would otherwise come back in step and rate-limit again.
        delay = backoff * (2 ** (attempt - 1)) * (0.5 + random.random())
        logger.info("vision call will be retried", extra={
            "label": label, "model": model, "attempt": attempt,
            "sleeping_seconds": round(delay, 1)})
        time.sleep(delay)
    return None


def _attempt(
    image_png: bytes, prompt: str, model: str, config: Config, *, label: str
) -> tuple[bool, Optional[str]]:
    """One call to one model.

    Returns ``(worth_retrying, body)``. The flag is what lets a caller tell a
    rejection to wait out from one that will answer the same way forever.
    """
    import httpx

    encoded = base64.b64encode(image_png).decode()
    try:
        response = httpx.post(
            f"{_base_url(config)}/chat/completions",
            headers={"Authorization": f"Bearer {_api_key(config)}"},
            json={
                "model": model,
                "max_tokens": config.get("vision.max_tokens", DEFAULT_MAX_TOKENS),
                "temperature": 0,
                # Gemini thinks before it answers and bills the thinking at the
                # output rate — 5,074 tokens of it against 496 of transcription
                # on a measured page, which made reading a page cost more than
                # the model this replaced. Transcription is copying words off an
                # image; it does not need the deliberation.
                **({"reasoning_effort": config.get("vision.reasoning_effort")}
                   if config.get("vision.reasoning_effort") else {}),
                "usage": {"include": True},
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                    ],
                }],
            },
            timeout=config.get("vision.timeout_seconds", DEFAULT_TIMEOUT),
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        status = error.response.status_code if error.response else None
        logger.warning("vision call rejected", extra={
            "label": label, "model": model, "status": status,
            "detail": (error.response.text[:200] if error.response else "")})
        return status in TRANSIENT_STATUSES, None
    except httpx.TimeoutException as error:
        # A timeout is the gateway being slow, not the page being unreadable.
        logger.warning("vision call timed out", extra={
            "label": label, "model": model, "error": type(error).__name__})
        return True, None
    except httpx.HTTPError as error:
        logger.warning("vision call failed", extra={
            "label": label, "model": model, "error": type(error).__name__})
        return False, None

    payload = response.json()
    record_usage(payload.get("usage"))
    choices = payload.get("choices") or []
    if not choices:
        logger.warning("vision returned no choices", extra={
            "label": label, "model": model})
        return False, None
    return False, choices[0].get("message", {}).get("content")
