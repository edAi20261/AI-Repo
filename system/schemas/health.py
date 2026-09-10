"""Liveness and readiness contracts."""

from __future__ import annotations

from .common import Schema


class HealthResponse(Schema):
    status: str = "ok"
    environment: str


class ReadinessResponse(Schema):
    status: str
    environment: str
    graph_ready: bool
    in_flight: int
