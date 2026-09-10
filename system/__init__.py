"""HTTP service exposing the EduAI runtime agent and ingestion pipeline.

This package is a transport layer only: it validates requests, enforces
service-to-service auth, and delegates to ``agent`` and ``ingestion``. No
teaching logic lives here.
"""

from .main import create_app

__all__ = ["create_app"]
