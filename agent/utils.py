"""Agent utilities: language detection and tolerant JSON extraction."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_ARABIC = re.compile(r"[؀-ۿ]")
_DEFAULT_SUPPORTED = ("ar", "en", "fr", "de")


def parse_structured(llm, schema, messages, config, label: str) -> dict:
    """Get a dict from the LLM for a JSON-shaped task.

    Config-driven, no probing: when ``llm.structured_output`` is true it uses the provider's
    native structured output (schema-validated), with a plain-call + ``extract_json`` fallback
    only if that call errors; when false it goes STRAIGHT to one plain call. Set the flag to
    match the CURRENT model (see ``llm.overrides``): true only for a model whose json_schema
    output is known to validate; false (the safe universal path) otherwise — no wasted calls."""
    attempt = True if config is None else config.get("llm.structured_output", True)
    if attempt:
        try:
            return llm.with_structured_output(schema).invoke(messages).model_dump()
        except Exception:
            logger.warning("%s structured output failed; JSON fallback", label)
    return extract_json(llm.invoke(messages).content) or {}


def detect_language(
    text: str,
    default: str = "en",
    *,
    supported=_DEFAULT_SUPPORTED,
    arabic_ratio: float = 0.30,
) -> str:
    """Detect the turn's language, restricted to ``supported``.

    Arabic is decided by script (fast and reliable); Latin-script languages
    (en/fr/de) are told apart with ``langdetect`` when available. Falls back to
    ``default`` (graph.md: the profile's script preference) when unsure or when
    the detected language isn't supported. Detected per turn."""
    if not text:
        return default
    arabic = len(_ARABIC.findall(text))
    letters = sum(1 for c in text if c.isalpha())
    if letters and arabic / letters >= arabic_ratio:
        return "ar" if "ar" in supported else default
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0  # deterministic
        code = detect(text)
    except Exception:
        return default
    return code if code in supported else default


# Friendly, localized message when a task can't run (LLM call failed). Never a stub or a
# stack trace to the user (en is the fallback for any unlisted language).
_DEGRADED = {
    "ar": "عذرًا، تعذّر إنجاز هذا الآن. من فضلك حاول مرة أخرى بعد قليل.",
    "en": "Sorry — I couldn't complete this right now. Please try again in a moment.",
    "fr": "Désolé — je n'ai pas pu réaliser cela pour le moment. Veuillez réessayer dans un instant.",
    "de": "Entschuldigung — das war gerade nicht möglich. Bitte versuchen Sie es gleich noch einmal.",
}


def degraded_message(language: str) -> str:
    """A user-facing 'couldn't do this right now' message for a failed task."""
    return _DEGRADED.get(language, _DEGRADED["en"])


def extract_json(text: str | None) -> dict | None:
    """Recover a JSON object from a model response that may wrap it in a ```json
    fence or surrounding prose."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    else:
        bare = re.search(r"\{.*\}", text, re.S)
        if bare:
            text = bare.group(0)
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None
