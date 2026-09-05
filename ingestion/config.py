"""Configuration loader. The implementation now lives in ``shared.config`` so both
deployable halves share one loader; this module re-exports it for the ingestion
package's existing imports."""

from shared.config import Config, get_config

__all__ = ["Config", "get_config"]
