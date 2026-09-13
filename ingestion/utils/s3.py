"""Object-storage keys for cropped figures.

The uploader backends live in ``shared.storage`` (shared with the agent's render
artifacts); this module re-exports them and owns the ingestion figure key layout.
"""

from __future__ import annotations

import hashlib

from shared.storage import (  # re-exported for existing ingestion imports
    LocalUploader,
    S3Uploader,
    StorageError,
    Uploader,
    get_uploader,
)

__all__ = [
    "LocalUploader",
    "S3Uploader",
    "StorageError",
    "Uploader",
    "figure_key",
    "figure_prefix",
    "get_uploader",
]


def figure_prefix(lecture_id: str) -> str:
    """Logical prefix holding every figure crop of one lecture."""
    return f"figures/{lecture_id}/"


def figure_key(lecture_id: str, page: int, image_bytes: bytes) -> str:
    """Deterministic object key: same crop re-ingested lands at the same key."""
    digest = hashlib.sha256(image_bytes).hexdigest()[:16]
    return f"{figure_prefix(lecture_id)}p{page:03d}_{digest}.png"
