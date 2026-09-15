"""Agent utilities: tolerant JSON extraction for model output."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

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
