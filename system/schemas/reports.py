"""The progress-report contract."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from .common import Schema, Visual


class ReportResponse(Schema):
    request_id: str
    scope: str = "student_progress"
    available: bool = Field(description="False when there is not yet enough graded work.")
    narrative: str = ""
    weak_topics: list[dict[str, Any]] = Field(default_factory=list)
    root_cause: Optional[dict[str, Any]] = None
    trend: dict[str, Any] = Field(default_factory=dict)
    visual: Optional[Visual] = Field(default=None, description="Rendered trend chart, when available.")
    degraded: bool = False

    @classmethod
    def from_result(cls, result: dict[str, Any], request_id: str) -> ReportResponse:
        known = {k: v for k, v in result.items() if k in cls.model_fields}
        return cls(request_id=request_id, **known)
