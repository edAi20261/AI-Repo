"""The progress-report contract."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from shared.metering import Meter

from .common import Schema, Usage, Visual


class ReportResponse(Schema):
    request_id: str
    scope: str = "student_progress"
    available: bool = Field(description="False when there is not yet enough graded work.")
    narrative: str = ""
    claims: list[dict[str, Any]] = Field(
        default_factory=list,
        description="The narrative sentence by sentence, each with the evidence it cites.")
    weak_topics: list[dict[str, Any]] = Field(default_factory=list)
    strengths: list[dict[str, Any]] = Field(default_factory=list)
    root_cause: Optional[dict[str, Any]] = None
    trend: dict[str, Any] = Field(default_factory=dict)
    visual: Optional[Visual] = Field(default=None, description="Rendered trend chart, when available.")
    degraded: bool = False
    report_id: Optional[str] = None
    generated_at: Optional[str] = None
    persisted: bool = Field(default=False, description="Whether this report was stored.")
    usage: Usage

    @classmethod
    def from_result(cls, result: dict[str, Any], request_id: str, meter: Meter) -> ReportResponse:
        known = {k: v for k, v in result.items() if k in cls.model_fields}
        return cls(request_id=request_id, usage=Usage.from_meter(meter), **known)


class StoredReportResponse(Schema):
    request_id: str
    report_id: str
    generated_at: str
    stale: Optional[bool] = Field(
        default=None,
        description="True when graded answers arrived after this report was written: "
                    "serve it, and ask for a fresh one in the background.")
    newer_evidence: Optional[int] = None
    report: dict[str, Any] = Field(description="The report as it was built.")

