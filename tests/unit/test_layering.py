"""The layering in CLAUDE.md, enforced.

    shared/      <- agent/, ingestion/, system/     (leaf: depends on nothing above)
    agent/       <- system/
    ingestion/   <- system/
    agent/  <->  ingestion/                         never import each other

It was written down and nothing checked it. An import inside a function counts:
that is exactly where a shortcut across a layer hides.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = {
    "shared": {"agent", "ingestion", "system"},
    "agent": {"ingestion", "system"},
    "ingestion": {"agent", "system"},
}


def _imported_packages(path: Path, package: str) -> set[tuple[str, int]]:
    module_parts = path.relative_to(ROOT).with_suffix("").parts
    found: set[tuple[str, int]] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found |= {(alias.name.split(".")[0], node.lineno) for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = module_parts[:len(module_parts) - node.level]
                top = base[0] if base else package
            else:
                top = (node.module or "").split(".")[0]
            found.add((top, node.lineno))
    return found


def _violations() -> list[str]:
    out = []
    for package, banned in FORBIDDEN.items():
        for path in sorted((ROOT / package).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            for top, line in _imported_packages(path, package):
                if top in banned:
                    out.append(f"{path.relative_to(ROOT)}:{line} imports {top}")
    return out


def test_no_layer_imports_one_above_it_or_beside_it():
    assert _violations() == []


@pytest.mark.parametrize("package", sorted(FORBIDDEN))
def test_the_sweep_finds_the_files_it_checks(package):
    """A sweep that silently finds nothing protects nothing."""
    assert any((ROOT / package).rglob("*.py"))
