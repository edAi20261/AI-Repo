"""Shared dependencies (clients, stores) handed to every stage.

Built once per run and reused across stages. The embedder, topic extractor and
stores are lazy — stages that don't need them never trigger the load.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import Config
from .extractors import (
    DependencyDeriver,
    TopicExtractor,
    get_dependency_deriver,
    get_topic_extractor,
)
from .llm_provider import (
    Embedder,
    VisionClient,
    get_embedder,
    get_vision_client,
)
from .stores import (
    ChunkStore,
    VectorStore,
    get_chunk_store,
    get_vector_store,
)
from .utils.s3 import Uploader, get_uploader


@dataclass
class PipelineContext:
    config: Config
    vision: VisionClient
    uploader: Uploader
    _embedder: Optional[Embedder] = None
    _topic_extractor: Optional[TopicExtractor] = None
    _dependency_deriver: Optional[DependencyDeriver] = None
    _chunk_store: Optional[ChunkStore] = None
    _vector_store: Optional[VectorStore] = None

    @classmethod
    def build(cls, config: Config) -> "PipelineContext":
        return cls(
            config=config,
            vision=get_vision_client(config),
            uploader=get_uploader(config),
        )

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder(self.config)
        return self._embedder

    @property
    def topic_extractor(self) -> TopicExtractor:
        if self._topic_extractor is None:
            self._topic_extractor = get_topic_extractor(self.config)
        return self._topic_extractor

    @property
    def dependency_deriver(self) -> DependencyDeriver:
        if self._dependency_deriver is None:
            self._dependency_deriver = get_dependency_deriver(self.config)
        return self._dependency_deriver

    @property
    def chunk_store(self) -> ChunkStore:
        if self._chunk_store is None:
            self._chunk_store = get_chunk_store(self.config)
        return self._chunk_store

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = get_vector_store(self.config)
        return self._vector_store
