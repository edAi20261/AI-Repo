"""Unit tests never reach a real database."""

from __future__ import annotations

import pytest

from shared.config import get_config


def test_a_test_without_a_double_cannot_reach_the_database():
    """Locally DATABASE_URL is production. A read tool called without fake_postgres
    would have queried it, and a write tool would have written to it."""
    from agent.tools import get_mastery

    with pytest.raises(AssertionError, match="real database connection"):
        get_mastery("u1", config=get_config())
