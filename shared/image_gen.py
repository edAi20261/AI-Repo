"""Provider-agnostic illustrative image generation.

Precise educational figures (math, charts, geometry) are rendered deterministically in
``agent/render.py`` — image models are unreliable for exact figures. This module is only
for *illustrative*, non-precise pictures. Provider is a config change, not a code change
(``image_gen.provider`` + ``model``); the SDK import is lazy. When the configured provider
has no API key, this falls back to a deterministic stub with a WARNING — never silently.
"""

from __future__ import annotations

import io
import logging

from .config import Config

logger = logging.getLogger(__name__)

# provider -> the .env key that must be present for the real client.
_PROVIDER_KEY = {"openai": "OPENAI_API_KEY"}


class ImageGenerator:
    """Text-to-image behind a thin interface so callers never name a provider."""

    def generate(self, prompt: str, *, size: str | None = None) -> bytes:
        raise NotImplementedError


class StubImageGenerator(ImageGenerator):
    """Deterministic placeholder PNG (the prompt drawn on a blank canvas). Used for tests
    and whenever no real provider key is configured — so a missing key degrades loudly to
    a visible placeholder instead of failing the turn."""

    def __init__(self, size: str = "1024x1024"):
        self._size = size

    def generate(self, prompt: str, *, size: str | None = None) -> bytes:
        from PIL import Image, ImageDraw
        w, h = _wh(size or self._size)
        img = Image.new("RGB", (w, h), (238, 240, 245))
        draw = ImageDraw.Draw(img)
        text = (prompt or "")[:180]
        draw.rectangle([8, 8, w - 8, h - 8], outline=(150, 160, 175), width=3)
        draw.text((24, 24), f"[placeholder image]\n{text}", fill=(60, 70, 90))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


class OpenAIImageGenerator(ImageGenerator):
    """OpenAI Images (e.g. ``gpt-image-1``). Returns PNG bytes."""

    def __init__(self, model: str, api_key: str, size: str = "1024x1024"):
        self._model = model
        self._key = api_key
        self._size = size

    def generate(self, prompt: str, *, size: str | None = None) -> bytes:
        import base64

        from openai import OpenAI
        client = OpenAI(api_key=self._key)
        resp = client.images.generate(
            model=self._model, prompt=prompt, size=size or self._size, n=1)
        return base64.b64decode(resp.data[0].b64_json)


def _wh(size: str) -> tuple[int, int]:
    try:
        w, h = size.lower().split("x")
        return int(w), int(h)
    except Exception:
        return 1024, 1024


def get_image_generator(config: Config) -> ImageGenerator:
    """The configured image generator, or a stub (with a WARNING) when its key is absent."""
    cfg = config.get("image_gen", {}) or {}
    provider = (cfg.get("provider") or "stub").lower()
    size = cfg.get("size", "1024x1024")
    if provider == "stub":
        return StubImageGenerator(size)
    key_env = _PROVIDER_KEY.get(provider)
    api_key = config.secret(key_env) if key_env else None
    if provider == "openai" and api_key:
        return OpenAIImageGenerator(cfg.get("model", "gpt-image-1"), api_key, size)
    logger.warning("image_gen provider %r has no API key (%s); using placeholder stub",
                   provider, key_env)
    return StubImageGenerator(size)
