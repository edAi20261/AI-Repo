"""scripts/verify.py: one definition of done, and it cannot reach production.

CI, the developer agent's stop gate and an engineer before pushing all run it.
If any of them ran something else, "it passed" would mean different things.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location("verify", ROOT / "scripts" / "verify.py")
verify = importlib.util.module_from_spec(_spec)
sys.modules["verify"] = verify
_spec.loader.exec_module(verify)


def test_unit_tests_never_inherit_a_production_database_url():
    """shared/config.py loads .env without overriding the environment.

    On an engineer's machine that .env is production, so a unit run that merely
    defaulted DATABASE_URL would have talked to it.
    """
    env = verify.environment({"DATABASE_URL": "postgresql://prod.example/edu",
                              "QDRANT_URL": "https://prod.qdrant.example"})
    assert env["DATABASE_URL"] == verify.TEST_ENV["DATABASE_URL"]
    assert env["QDRANT_URL"] == ":memory:"


def test_a_focused_run_is_never_mistaken_for_done():
    """Only the bare form lints; a run of three tests must not read as a verify."""
    assert [name for name, _ in verify.steps([])] == ["lint", "tests"]
    assert [name for name, _ in verify.steps(["tests/unit/test_grade.py"])] == ["tests"]


def test_ci_runs_the_same_verify_as_everyone_else():
    """CLAUDE.md rule 3: "done" is decided in scripts/verify.py and nowhere else.

    Had CI kept its own lint paths, the developer's gate and the merge gate
    could disagree about the same commit.
    """
    workflow = (ROOT / ".github" / "workflows" / "deploy.yml").read_text()
    assert "scripts/verify.py" in workflow
    assert not re.search(r"-m (pytest|ruff)\b", workflow)
