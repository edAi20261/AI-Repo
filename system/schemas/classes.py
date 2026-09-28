"""The class-analytics contract."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from .common import Language, Schema


class ClassAnalyticsRequest(Schema):
    user_ids: list[str] = Field(
        min_length=1, max_length=500,
        description="The class roster. The platform decides who the teacher may see; "
                    "this service aggregates exactly these students.")
    subject_ids: Optional[list[str]] = Field(
        default=None, description="Scopes the analysis to the class's subject(s).")
    class_id: Optional[str] = Field(default=None, description="Logged, not used to look up.")

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class ClassAnalyticsResponse(Schema):
    request_id: str
    available: bool = Field(description="False until some student has graded work to judge.")
    degraded: bool = False
    students: dict[str, Any] = Field(default_factory=dict)
    topics: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per topic, how many students are strong, developing, weak, unknown and "
                    "not_practised, weakest topics first.")
    at_risk: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Students needing attention, each with `reasons` as facts with a `code`.")
    not_started: list[str] = Field(
        default_factory=list, description="Students with no graded work yet.")
    mistakes: list[dict[str, Any]] = Field(default_factory=list)
    most_missed: list[dict[str, Any]] = Field(default_factory=list)
    activity: list[dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def from_result(cls, result: dict[str, Any], request_id: str) -> ClassAnalyticsResponse:
        known = {k: v for k, v in result.items() if k in cls.model_fields}
        return cls(request_id=request_id, **known)


class ClassReportRequest(ClassAnalyticsRequest):
    class_id: str = Field(description="The class the report is stored under.")
    language: Optional[Language] = None


class ClassReportResponse(Schema):
    request_id: str
    scope: str = "class_analytics"
    available: bool
    narrative: str = ""
    claims: list[dict[str, Any]] = Field(default_factory=list)
    analysis: dict[str, Any] = Field(default_factory=dict)
    degraded: bool = False
    report_id: Optional[str] = None
    generated_at: Optional[str] = None
    persisted: bool = False

    @classmethod
    def from_result(cls, result: dict[str, Any], request_id: str) -> ClassReportResponse:
        known = {k: v for k, v in result.items() if k in cls.model_fields}
        return cls(request_id=request_id, **known)

