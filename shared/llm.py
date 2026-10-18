"""Provider-agnostic chat model access, shared by both deployable halves.

Every chat client is constructed here, so moving to a different model is a config
change (``llm.provider`` + ``llm.model`` in config.yaml), never a code change.
Per-task settings come from ``config.llm_for(task)``; SDK imports are lazy so
importing this module never requires a provider that isn't installed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial

from .config import Config

logger = logging.getLogger(__name__)

# OpenAI-compatible gateways: one key reaches many models (base URL, env var).
_OPENAI_COMPAT = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "moonshot": ("https://api.moonshot.ai/v1", "MOONSHOT_API_KEY"),
    "kimi": ("https://api.moonshot.ai/v1", "MOONSHOT_API_KEY"),
}

_NATIVE_KEY = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
}


VERTEX = "vertex"


def _vertex_gateway(config: Config) -> tuple[str, Callable[[], str]]:
    """Vertex's OpenAI-compatible base URL and a fresh bearer token.

    Not in ``_OPENAI_COMPAT`` because neither half is a constant: the address
    carries the project and region, and the credential is an OAuth token that
    expires in an hour rather than a key that does not.
    """
    from .gcp import access_token, billed_project, credential_source, vertex_base_url

    source = credential_source(config)
    if not source:
        raise ValueError(
            "a task is configured for vertex but no service account is set; "
            "GOOGLE_APPLICATION_CREDENTIALS takes a path or the JSON itself.")
    project = billed_project(config, source)
    location = config.get("vision.location") or "us-central1"
    return vertex_base_url(project, location), partial(access_token, source)


def llm_key_available(config: Config, spec: dict) -> bool:
    """True when ``spec`` names a provider/model whose credential is present."""
    provider = (spec.get("provider") or "").lower()
    if not provider or not spec.get("model"):
        return False
    if provider == VERTEX:
        from .gcp import credential_source

        return bool(credential_source(config))
    if provider in _OPENAI_COMPAT:
        return bool(config.secret(_OPENAI_COMPAT[provider][1]))
    return bool(config.secret(_NATIVE_KEY.get(provider, "")))


def get_llm(config: Config, task: str | None = None):
    """Chat model for ``task``. Native providers go through ``init_chat_model``;
    OpenAI-compatible gateways through ``ChatOpenAI`` with the gateway base URL."""
    spec = config.llm_for(task)
    provider = (spec.get("provider") or "").lower()
    model = spec.get("model")
    if not provider or not model:
        raise ValueError(f"No LLM provider/model configured for task {task!r}.")

    common = dict(
        temperature=spec.get("temperature", 0.2),
        max_tokens=spec.get("max_tokens"),
        timeout=spec.get("timeout_seconds"),
        max_retries=spec.get("max_retries", 2),
    )

    if provider == VERTEX:
        from langchain_openai import ChatOpenAI  # lazy

        from .metering import MeteringCallback

        base_url, token = _vertex_gateway(config)
        token()
        return ChatOpenAI(model=model, base_url=base_url, api_key=token,
                          callbacks=[MeteringCallback()], **common)

    if provider in _OPENAI_COMPAT:
        from langchain_openai import ChatOpenAI  # lazy

        from .metering import MeteringCallback

        base_url, key_env = _OPENAI_COMPAT[provider]
        extra_body = _openrouter_body(config, model) if provider == "openrouter" else None
        return ChatOpenAI(
            model=model,
            base_url=spec.get("base_url", base_url),
            api_key=config.secret(key_env),
            extra_body=extra_body,
            callbacks=[MeteringCallback()],
            **common,
        )

    from langchain.chat_models import init_chat_model  # lazy

    return init_chat_model(model=model, model_provider=provider, **common)


def _openrouter_body(config: Config, model: str) -> dict:
    """Per-call cost accounting, and routing that can actually honour a schema.

    OpenRouter serves the same model from several providers and only some enforce
    ``strict`` — ``require_parameters`` keeps a structured call off the endpoints
    that would treat the schema as a suggestion and hand back prose.

    ``models`` is the same idea one level up: when the chosen endpoint answers
    with an error instead of a completion, the gateway serves the next model in
    the list rather than handing us a body the SDK cannot read. Every model named
    here must itself honour ``json_schema``, or the fallback trades a failed turn
    for an unstructured one.
    """
    body: dict = {"usage": {"include": True}}
    if config.get("llm.require_structured_endpoints", True):
        body["provider"] = {"require_parameters": True}
    fallbacks = [m for m in (config.get("llm.fallback_models") or []) if m and m != model]
    if fallbacks:
        body["models"] = [model, *fallbacks]
    return body


def maybe_llm(config: Config, task: str):
    """The chat model for ``task``, or ``None`` when no key is configured (callers then take
    the deterministic stub path). The single source for 'LLM if available, else None'."""
    try:
        if llm_key_available(config, config.llm_for(task)):
            return get_llm(config, task)
    except Exception:
        logger.warning("LLM unavailable for task %r", task, exc_info=True)
    return None
