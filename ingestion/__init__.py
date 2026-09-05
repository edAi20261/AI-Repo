"""Offline ingestion pipeline.

One lesson PDF (already split — one file = one lesson) is turned into embedded,
indexed chunks so that at runtime the agent only does table reads and a vector
search. Public entry points:

    from ingestion import new_state, run_ingestion

    state = new_state(lecture_id=..., subject_id=..., source_uri=...)
    result = run_ingestion(state)
"""

from .backend import (
    BackendClient,
    FixtureBackendClient,
    HttpBackendClient,
    LessonSource,
    get_backend_client,
    resolve_lesson,
)
from .config import Config, get_config
from .context import PipelineContext
from .graph import build_graph, run_ingestion
from .state import (
    Chunk,
    Dependency,
    Discipline,
    Element,
    Figure,
    IngestionState,
    PageResult,
    new_state,
)

__all__ = [
    "Config",
    "get_config",
    "PipelineContext",
    "build_graph",
    "run_ingestion",
    "IngestionState",
    "new_state",
    "Chunk",
    "Dependency",
    "Discipline",
    "Element",
    "Figure",
    "PageResult",
    "BackendClient",
    "HttpBackendClient",
    "FixtureBackendClient",
    "LessonSource",
    "get_backend_client",
    "resolve_lesson",
]
