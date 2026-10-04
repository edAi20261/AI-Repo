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
