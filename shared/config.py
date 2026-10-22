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

DEV_ENVIRONMENTS = frozenset({"local", "test", "dev", "ci"})


def _as_pasted(name: str, value: str) -> str:
    """A secret's value, from the ways a person pastes one.

    The whole line of a .env file or a table — ``NAME=value``, ``NAME   value`` —
    and quotes around the value are the mistakes that reached the repository's
    secrets. Each one signed every S3 request with the wrong key and failed the
    ingest run with SignatureDoesNotMatch, an error that names nothing.
    """
    text = value.strip()
    if text.startswith(name):
        rest = text[len(name):]
        if rest[:1] in ("=", " ", "\t", ":"):
            text = rest.lstrip(" \t=:")
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


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

    @property
    def is_dev(self) -> bool:
        return self.app_env in DEV_ENVIRONMENTS

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
        # An overlay named with nothing under it parses as None, not {}.
        overlay = overlays.get(app_env) or {}
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

    def override(self, dotted_key: str, value: Any) -> None:
        """Set one value for this process, above the file and its overlay.

        For a switch an operator flips for a single run — reading one book with
        a parser the deployment does not use by default. It is not a second
        home for a default: the value in config.yaml is still the one that
        ships, and nothing that is not asked for on a command line lands here.
        """
        node: Any = self._data
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = value

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
        """Read a secret/connection value from the environment.

        Without the whitespace around it. A value pasted into a CI secrets form
        or a .env file arrives with a leading space or a trailing newline often
        enough, and nothing it names — a URL, a key, a bucket — contains either:
        a Qdrant URL saved as " https://…" failed every run of the ingest
        workflow with "Host ' https' contains invalid character".
        """
        value = os.environ.get(name)
        return default if value is None else _as_pasted(name, value)

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
