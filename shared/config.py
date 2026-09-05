"""Loads config.yaml (behaviour) + .env (secrets) and exposes them.

Shared by both deployable halves. The `environments:` block is deep-merged over
the base by APP_ENV. Values are read by dotted path (`get`), secrets from the
environment (`secret`), and per-task LLM settings via `llm_for`.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

# .env loading is optional; deployments may inject env vars directly.
try:  # pragma: no cover
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False


# shared/config.py -> repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _REPO_ROOT / "config.yaml"
_DEFAULT_ENV = _REPO_ROOT / ".env"


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` onto ``base`` without mutating either."""
    out: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if (
            key in out
            and isinstance(out[key], Mapping)
            and isinstance(value, Mapping)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class Config:
    """Read-only view over the merged configuration. Values by dotted path;
    secrets via ``secret`` (environment only)."""

    def __init__(self, data: Mapping[str, Any], app_env: str) -> None:
        self._data = dict(data)
        self.app_env = app_env

    # -- construction --------------------------------------------------------

    @classmethod
    def load(
        cls,
        config_path: str | os.PathLike[str] | None = None,
        env_path: str | os.PathLike[str] | None = None,
    ) -> "Config":
        """Load yaml + .env and apply the ``APP_ENV`` overlay."""
        env_file = Path(env_path) if env_path else _DEFAULT_ENV
        if env_file.exists():
            load_dotenv(env_file)

        app_env = os.environ.get("APP_ENV", "local")

        cfg_file = Path(config_path) if config_path else _DEFAULT_CONFIG
        with open(cfg_file, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        overlays = raw.pop("environments", {}) or {}
        overlay = overlays.get(app_env, {})
        merged = _deep_merge(raw, overlay)
        return cls(merged, app_env)

    # -- behavioural values (config.yaml) ------------------------------------

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Fetch a value by dotted path, e.g. ``ingestion.ocr.render_dpi``."""
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, dotted_key: str) -> dict[str, Any]:
        """A copy of a config sub-tree; ``{}`` if absent."""
        node = self.get(dotted_key, {})
        return copy.deepcopy(node) if isinstance(node, Mapping) else {}

    def llm_for(self, task: str | None = None) -> dict[str, Any]:
        """Base ``llm`` block with ``llm.overrides.<task>`` merged on top. ``None``
        values in an override are ignored (a null model means no LLM)."""
        base = {k: v for k, v in self.section("llm").items() if k != "overrides"}
        if task is None:
            return base
        override = self.get(f"llm.overrides.{task}", {}) or {}
        merged = _deep_merge(base, {k: v for k, v in override.items() if v is not None})
        return merged

    # -- secrets (.env / environment) ----------------------------------------

    @staticmethod
    def secret(name: str, default: str | None = None) -> str | None:
        """Read a secret/connection value from the environment."""
        return os.environ.get(name, default)

    # -- convenience accessors -----------------------------------------------

    @property
    def ingestion(self) -> dict[str, Any]:
        return self.section("ingestion")

    @property
    def embeddings(self) -> dict[str, Any]:
        return self.section("embeddings")

    @property
    def vector_store(self) -> dict[str, Any]:
        return self.section("vector_store")

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config(app_env={self.app_env!r})"


_INSTANCE: Config | None = None


def get_config(reload: bool = False) -> Config:
    """Process-wide Config singleton. ``reload=True`` rebuilds it."""
    global _INSTANCE
    if _INSTANCE is None or reload:
        _INSTANCE = Config.load()
    return _INSTANCE
