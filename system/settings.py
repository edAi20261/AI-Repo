"""Service settings.

Behavioural values are read from ``config.yaml`` (``api.*``); secrets come from
the environment. This mirrors the split the rest of the codebase follows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shared.config import DEV_ENVIRONMENTS, Config

# The platform owns this contract: PATCH a lesson's content.ingestion_status.
DEFAULT_CALLBACK_PATH = "/lessons/{lesson_id}/ingestion"


@dataclass(frozen=True)
class IngestionSettings:
    max_workers: int = 2
    callback_enabled: bool = False
    callback_path: str = DEFAULT_CALLBACK_PATH
    job_retention: int = 500


@dataclass(frozen=True)
class Settings:
    app_env: str
    service_token: str | None
    backend_base_url: str | None
    max_concurrency: int
    request_timeout: float
    cors_origins: list[str] = field(default_factory=list)
    docs_enabled: bool = True
    ingestion: IngestionSettings = field(default_factory=IngestionSettings)

    @property
    def is_dev(self) -> bool:
        return self.app_env in DEV_ENVIRONMENTS

    @classmethod
    def from_config(cls, config: Config) -> Settings:
        api = config.section("api")
        ingestion = api.get("ingestion") or {}
        return cls(
            app_env=config.app_env,
            service_token=Config.secret("AI_WEBHOOK_SHARED_SECRET"),
            backend_base_url=Config.secret("BACKEND_API_BASE_URL"),
            max_concurrency=int(api.get("max_concurrency", 8)),
            request_timeout=float(api.get("request_timeout", 120)),
            cors_origins=list(api.get("cors_origins") or []),
            docs_enabled=bool(api.get("docs_enabled", True)),
            ingestion=IngestionSettings(
                max_workers=int(ingestion.get("max_workers", 2)),
                callback_enabled=bool(ingestion.get("callback_enabled", False)),
                callback_path=str(ingestion.get("callback_path", DEFAULT_CALLBACK_PATH)),
                job_retention=int(ingestion.get("job_retention", 500)),
            ),
        )
