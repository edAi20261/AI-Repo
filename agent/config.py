"""Agent configuration — the shared Config loader.

Behavioural knobs live under the ``agent.*`` block in config.yaml; secrets in
.env. Re-exported so agent modules import config from their own package.
"""

from __future__ import annotations

from shared.config import Config, get_config

__all__ = ["Config", "get_config"]
