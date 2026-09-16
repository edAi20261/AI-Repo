"""config.yaml is the only place a setting's value is decided.

``config.get("agent.min_score", 0.0)`` puts a second value in the source, and the
two drift silently: the shipped floor is 0.50, chosen from a measured sweep, while
five call sites carried 0.0 — the value that same sweep recorded as answering five
of eight questions the syllabus could not answer. Whichever wins depends on
whether an overlay happens to include the block.

Two rules, enforced here rather than in review:

  * a key the code reads must be declared in config.yaml
  * a literal fallback at a call site must equal the declared value

A fallback is still allowed — it is the safety net when a file is truncated or an
overlay is thin. It just may not disagree.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("agent", "shared", "system", "ingestion")

# Read from the environment or a URL, never from config.yaml.
NOT_SETTINGS = {"https://openrouter.ai/api/v1/key"}


def _declared() -> dict[str, object]:
    """Every dotted path in config.yaml, mapped to its value."""
    def walk(node, path=()):
        if isinstance(node, dict):
            for key, value in node.items():
                dotted = (*path, str(key))
                yield ".".join(dotted), value
                yield from walk(value, dotted)

    return dict(walk(yaml.safe_load((ROOT / "config.yaml").read_text())))


def _call_sites() -> list[tuple[str, int, str, object, bool]]:
    """``(file, line, key, default, has_default)`` for each ``.get("a.b", …)``."""
    found: list[tuple[str, int, str, object, bool]] = []
    for package in PACKAGES:
        for path in (ROOT / package).rglob("*.py"):
            if "__pycache__" in str(path):
                continue
            module = ast.parse(path.read_text())
            constants = {
                target.id: node.value.value
                for node in module.body
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            for node in ast.walk(module):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "get" and node.args):
                    continue
                key = node.args[0]
                if not (isinstance(key, ast.Constant) and isinstance(key.value, str)
                        and "." in key.value and key.value not in NOT_SETTINGS):
                    continue
                default, resolved = None, False
                if len(node.args) > 1:
                    argument = node.args[1]
                    if isinstance(argument, ast.Name) and argument.id in constants:
                        default, resolved = constants[argument.id], True
                    else:
                        try:
                            default, resolved = ast.literal_eval(argument), True
                        except ValueError:
                            resolved = False
                rel = str(path.relative_to(ROOT))
                found.append((rel, node.lineno, key.value, default,
                              len(node.args) > 1 and resolved))
    return found


def test_the_code_reads_no_setting_config_does_not_declare():
    """A key with no home in config.yaml can only ever be its Python fallback,
    so it is not configurable at all — however configurable it looks."""
    declared, undeclared = _declared(), {}
    for path, line, key, _default, _resolved in _call_sites():
        if key not in declared:
            undeclared.setdefault(key, f"{path}:{line}")
    assert not undeclared, (
        "read but not declared in config.yaml:\n  "
        + "\n  ".join(f"{k}  ({where})" for k, where in sorted(undeclared.items())))


def test_no_fallback_contradicts_the_shipped_value():
    """The dangerous shape: config.yaml says one thing, the code says another,
    and which applies depends on whether an overlay carried the block."""
    declared, conflicts = _declared(), []
    for path, line, key, default, resolved in _call_sites():
        if not resolved or key not in declared:
            continue
        # A block of overrides falling back to {} is the "no overrides" idiom:
        # the sub-values keep their own defaults inside the function. Only a
        # fallback that becomes the behaviour on its own is a contradiction.
        if isinstance(declared[key], dict) and default == {}:
            continue
        if declared[key] != default:
            conflicts.append(
                f"{key}: config.yaml={declared[key]!r} but {path}:{line} "
                f"falls back to {default!r}")
    assert not conflicts, "config.yaml and the code disagree:\n  " + "\n  ".join(conflicts)


@pytest.mark.parametrize("key", [
    "agent.min_score",
    "agent.topic_filter",
    "agent.resolve_topics_qdrant",
    "language.supported",
    "storage.key_prefix",
])
def test_the_settings_that_change_behaviour_are_declared(key):
    """Named individually because each of these once had a fallback that
    inverted it — retrieval that stopped declining, topic scoping that dropped
    the right material, the slow path, half the languages, the bucket root."""
    assert key in _declared()
