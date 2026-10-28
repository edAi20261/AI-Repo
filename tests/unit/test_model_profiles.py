"""One word moves all of ingestion between model vendors.

Ingestion's models were named on five lines in two blocks, all on Vertex. When
Vertex billing was switched off, moving to another vendor meant editing each
line and remembering which of them belonged to ingestion — and a line missed
leaves one step of every book on a gateway that can no longer be paid for.
"""

from __future__ import annotations

import pytest

from shared.config import Config

INGESTION_TASKS = ("extract_topics", "derive_deps", "split_lessons")


def _load(monkeypatch, profile):
    if profile is None:
        monkeypatch.delenv("MODEL_PROFILE", raising=False)
    else:
        monkeypatch.setenv("MODEL_PROFILE", profile)
    return Config.load()


def test_the_shipped_profile_keeps_ingestion_where_it_was(monkeypatch):
    config = _load(monkeypatch, None)
    assert config.get("vision.provider") == "vertex"
    assert {config.llm_for(t)["provider"] for t in INGESTION_TASKS} == {"vertex"}


def test_the_openai_profile_moves_every_ingestion_call_to_openai(monkeypatch):
    """A step left behind would still bill the gateway that was switched off."""
    config = _load(monkeypatch, "openai")
    assert config.get("vision.provider") == "openai"
    assert {config.llm_for(t)["provider"] for t in INGESTION_TASKS} == {"openai"}
    models = [config.get("vision.model"), config.get("vision.fallback_model"),
              *(config.llm_for(t)["model"] for t in INGESTION_TASKS)]
    assert not [m for m in models if "gemini" in m or "/" in m]


def test_the_openai_profile_leaves_the_tutor_alone(monkeypatch):
    """The profile is for ingestion; a student's turn must not move with it."""
    shipped, moved = _load(monkeypatch, None), _load(monkeypatch, "openai")
    for task in ("converse", "explain", "grade"):
        assert moved.llm_for(task) == shipped.llm_for(task)


def test_a_profile_that_does_not_exist_stops_the_run(monkeypatch):
    """A misspelt profile falling back to the shipped one would send the run to
    the gateway the operator was trying to leave."""
    with pytest.raises(ValueError, match="openia"):
        _load(monkeypatch, "openia")
