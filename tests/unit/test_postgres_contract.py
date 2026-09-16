"""Every connection this service opens is bounded on both ends.

psycopg has no default timeout. A host that hangs rather than refuses holds the
connection open until the caller gives up, which for a student's turn means
whatever the HTTP client allows — the turn does not fail, it stalls.

Five read tools opened their own connection instead of calling the helper that
already sat above them in the same file, and none of the five passed a timeout.
That is the shape of bug this file exists to make impossible: not "someone wrote
it wrong" but "the right way was optional".
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import shared.postgres as postgres
from shared.config import Config

ROOT = Path(__file__).resolve().parents[2]

# The offline pipeline is deliberately exempt: batch statements are meant to take
# minutes, and it talks to Postgres through SQLAlchemy rather than psycopg.
SERVED_AT_RUNTIME = ("agent", "system", "shared")


def _connect_calls(tree: ast.AST) -> list[ast.Call]:
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "connect"
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "psycopg"]


def _runtime_modules() -> list[Path]:
    return sorted(p for package in SERVED_AT_RUNTIME
                  for p in (ROOT / package).rglob("*.py"))


@pytest.mark.parametrize("path", _runtime_modules(), ids=lambda p: p.name)
def test_nothing_on_a_turn_opens_its_own_connection(path: Path):
    """One connection helper, so one place decides what a hang costs."""
    if path == Path(postgres.__file__):
        return
    calls = _connect_calls(ast.parse(path.read_text()))
    assert not calls, (
        f"{path.relative_to(ROOT)}:{calls[0].lineno} calls psycopg.connect "
        "directly — use shared.postgres.connect, which carries the timeouts")


def test_the_helper_passes_both_timeouts(monkeypatch):
    """The timeouts are the reason the helper exists."""
    seen = {}

    class _Conn:
        def close(self):
            pass

    def fake_connect(url, **kwargs):
        seen.update(kwargs)
        return _Conn()

    import psycopg
    monkeypatch.setattr(psycopg, "connect", fake_connect)
    with postgres.connect(Config.load()):
        pass

    assert seen["connect_timeout"] > 0
    assert "statement_timeout=" in seen["options"]


def test_a_caller_can_ask_for_a_tighter_bound(monkeypatch):
    """The session store sits at the front of every turn and wants less than a
    tool call does."""
    seen = {}

    class _Conn:
        def close(self):
            pass

    import psycopg
    monkeypatch.setattr(psycopg, "connect",
                        lambda url, **kwargs: (seen.update(kwargs), _Conn())[1])
    with postgres.connect(Config.load(), connect_timeout=2, statement_timeout_ms=500):
        pass

    assert seen["connect_timeout"] == 2
    assert seen["options"] == "-c statement_timeout=500"
