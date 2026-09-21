"""What a hybrid collection calls its two vectors.

Shared because both halves of the system name them: ingestion writes a point
under these names and the agent searches under them. A point written as one and
searched as another is a point nobody finds, so the strings live in one place.
"""

from __future__ import annotations

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
