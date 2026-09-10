"""Shapes shared by more than one endpoint."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Pathway = Literal["student", "teacher", "parent"]
Language = Literal["ar", "en", "fr", "de"]


class Schema(BaseModel):
    """Base for request bodies: unknown fields are rejected rather than ignored,
    so a client typo surfaces as a 422 instead of silently doing nothing."""

    model_config = ConfigDict(extra="forbid")


class GradeReference(Schema):
    """Points a grading turn at a saved assessment so its answer key is used."""

    assessment_id: str
    kind: Literal["quiz", "homework"]


class Visual(BaseModel):
    """A rendered study visual. Only the fields relevant to ``kind`` are present."""

    kind: str
    artifact_id: Optional[str] = None
    s3_uri: Optional[str] = None
    cards: Optional[list[dict[str, Any]]] = None
    text: Optional[str] = None
    figure_id: Optional[str] = None
    caption: Optional[str] = None
    item_id: Optional[str] = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    """The body of every non-2xx response."""

    error: ErrorDetail
