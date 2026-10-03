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
from typing import Optional, TypeVar

from pydantic import BaseModel

from .config import Config
from .metering import record_usage
from .structured import recover

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT = 180
DEFAULT_MAX_TOKENS = 4096

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


def _api_key(config: Config) -> Optional[str]:
    """The key for whichever gateway vision is pointed at.

    ``vision.api_key_env`` wins, so a deployment can name its own variable
    without this having to know about it.
    """
    named = config.get("vision.api_key_env")
    if named:
        return config.secret(named)
    provider = (config.get("vision.provider") or "openrouter").lower()
    return config.secret(PROVIDER_KEY_ENV.get(provider, DEFAULT_KEY_ENV))


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
    """One call. A failure is logged with its reason and returns ``None``.

    The reason is the point. This used to be a bare ``except Exception: return
    None``, so an expired API key, a rate limit and a genuinely unreadable page
    were indistinguishable — operators saw "needs review" on a whole textbook and
    had nothing to act on.
    """
    import httpx

    encoded = base64.b64encode(image_png).decode()
    try:
        response = httpx.post(
            f"{config.get('vision.base_url', DEFAULT_BASE_URL)}/chat/completions",
            headers={"Authorization": f"Bearer {_api_key(config)}"},
            json={
                "model": model,
                "max_tokens": config.get("vision.max_tokens", DEFAULT_MAX_TOKENS),
                "temperature": 0,
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
        logger.warning("vision call rejected", extra={
            "label": label, "model": model,
            "status": error.response.status_code if error.response else None,
            "detail": (error.response.text[:200] if error.response else "")})
        return None
    except httpx.HTTPError as error:
        logger.warning("vision call failed", extra={
            "label": label, "model": model, "error": type(error).__name__})
        return None

    payload = response.json()
    record_usage(payload.get("usage"))
    choices = payload.get("choices") or []
    if not choices:
        logger.warning("vision returned no choices", extra={
            "label": label, "model": model})
        return None
    return choices[0].get("message", {}).get("content")
