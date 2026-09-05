"""Object-storage upload for cropped figures.

The uploader backends now live in ``shared.storage`` (shared with the agent's render
artifacts); this module re-exports them and keeps the ingestion-specific ``figure_key``.
"""

from __future__ import annotations

import hashlib

from shared.storage import (  # re-exported for existing ingestion imports
    LocalUploader,
    S3Uploader,
    Uploader,
    get_uploader,
)

__all__ = ["Uploader", "S3Uploader", "LocalUploader", "get_uploader", "figure_key"]


def figure_key(lecture_id: str, page: int, image_bytes: bytes) -> str:
    """Deterministic object key: same crop re-ingested lands at the same key."""
    digest = hashlib.sha256(image_bytes).hexdigest()[:16]
    return f"figures/{lecture_id}/p{page:03d}_{digest}.png"
