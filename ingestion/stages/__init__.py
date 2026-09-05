"""Ingestion stages, in order: extract -> chunk -> embed -> topics ->
dependencies -> index.

Each stage is ``run(state, ctx) -> IngestionState``. The graph wires them; they
know nothing about each other beyond the state they read and write.
"""

from .chunk import run as chunk
from .dependencies import run as dependencies
from .embed import run as embed
from .extract import run as extract
from .index import run as index
from .topics import run as topics

__all__ = ["extract", "chunk", "embed", "topics", "dependencies", "index"]
