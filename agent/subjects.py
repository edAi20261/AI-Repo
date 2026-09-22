"""What the platform knows about a subject, read once and shared by every node.

A subject's language decides the language it is taught in, and its kind decides
the prompt addendum. Every adapter used to ask the backend separately, so one
turn made the same request up to five times; the answer changes when a teacher
edits the subject, not between the nodes of one turn.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from .models import SubjectFacts

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 300


class SubjectDirectory:
    """Subject facts from the backend, cached briefly, empty when unavailable."""

    def __init__(self, client, *, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        self._client = client
        self._ttl = ttl_seconds
        self._cache: dict[str, tuple[float, SubjectFacts]] = {}
        self._lock = threading.Lock()

    def facts(self, subject_id: Optional[str]) -> SubjectFacts:
        if self._client is None or not subject_id:
            return SubjectFacts()
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(subject_id)
        if hit and now - hit[0] < self._ttl:
            return hit[1]
        try:
            facts = SubjectFacts.model_validate(self._client.get_subject(subject_id) or {})
        except Exception:
            logger.warning("subject facts unavailable; using defaults",
                           extra={"subject_id": subject_id}, exc_info=True)
            return SubjectFacts()
        with self._lock:
            self._cache[subject_id] = (now, facts)
        return facts
