"""What a hybrid collection calls its two vectors.

Shared because both halves of the system name them: ingestion writes a point
under these names and the agent searches under them. A point written as one and
searched as another is a point nobody finds, so the strings live in one place.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"


@lru_cache(maxsize=8)
def _shape(client: Any, collection: str) -> bool:
    """Whether ``collection`` stores its vectors under names. Cached: a
    collection's shape does not change while a process runs, and asking Qdrant
    on every turn spends a round trip to learn something already known."""
    params = client.get_collection(collection).config.params
    return isinstance(params.vectors, dict)


def has_named_vectors(client: Any, collection: str) -> bool:
    """Whether a query against ``collection`` must name the vector it searches.

    Asked of the collection rather than read from config. A setting saying the
    index is hybrid while the index is not makes every query name a vector that
    does not exist, and Qdrant answers 400 — a 500 on every turn, where the
    honest outcome was "nothing found". The collection is the authority on its
    own shape; config only decides what shape a NEW one is built in.

    A collection that cannot be reached is treated as unnamed, because that is
    the form every collection written before this existed.
    """
    try:
        return _shape(client, collection)
    except Exception:  # noqa: BLE001 — a shape we cannot read is the old shape
        logger.warning("could not read the shape of %r; searching it as a "
                       "single unnamed vector", collection)
        return False


# The shape is cached for the life of the process; a test that changes it says so.
has_named_vectors.cache_clear = _shape.cache_clear
