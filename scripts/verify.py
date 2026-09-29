"""The definition of "done": lint and the unit suite, exactly as CI runs them.

    .venv/bin/python scripts/verify.py                      # done means this passes
    .venv/bin/python scripts/verify.py tests/unit/test_x.py::test_y   # just those tests

Exits non-zero on failure. CI, the developer agent's stop gate and a human
before pushing all run the first form, so "it passed" means the same thing to
each. The second form runs only the named tests, under the same environment, and
is never evidence that a change is done.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LINT_PATHS = ["agent", "shared", "system", "ingestion", "scripts", "tests", ".claude/hooks"]

# Forced, not defaulted. shared/config.py loads .env without overriding what is
# already set, and the .env on an engineer's machine points DATABASE_URL and
# Qdrant at production — so a unit test run that inherits the shell's
# environment can reach production.
TEST_ENV = {
    "DATABASE_URL": "postgresql://test:test@localhost:5432/test",
    "QDRANT_URL": ":memory:",
    "AI_WEBHOOK_SHARED_SECRET": "test-secret",
}


def steps(selectors: list[str], python: str = sys.executable) -> list[tuple[str, list[str]]]:
    tests = ("tests", [python, "-m", "pytest", "-q", *selectors])
    if selectors:
        return [tests]
    return [("lint", [python, "-m", "ruff", "check", *LINT_PATHS]), tests]


def environment(base: dict[str, str] | None = None) -> dict[str, str]:
    return {**(os.environ if base is None else base), **TEST_ENV}


def main(argv: list[str]) -> int:
    env = environment()
    failed = []
    for name, cmd in steps(argv):
        print(f"== {name}: {' '.join(cmd[1:])}", flush=True)
        if subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode != 0:
            failed.append(name)
    scope = "selected tests" if argv else "verify"
    if failed:
        print(f"== {scope} FAILED: {', '.join(failed)}", flush=True)
        return 1
    print(f"== {scope} passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
