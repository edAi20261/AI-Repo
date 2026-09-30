"""Unit tests never reach a real database or a real model."""

from __future__ import annotations

import pytest

from shared.config import get_config


def test_a_test_without_a_double_cannot_reach_the_database():
    """Locally DATABASE_URL is production. A read tool called without fake_postgres
    would have queried it, and a write tool would have written to it."""
    from agent.tools import get_mastery

    with pytest.raises(AssertionError, match="real database connection"):
        get_mastery("u1", config=get_config())


def test_a_test_gets_the_stub_model_even_with_a_key_in_dotenv():
    """With a provider key in .env, a test took the real-model path and made network
    calls CI never makes, so the two disagreed about what passes."""
    from shared.llm import maybe_llm

    config = get_config(reload=True)
    assert maybe_llm(config, "explain") is None
    assert maybe_llm(config, "grade") is None
