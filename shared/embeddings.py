"""The bilingual index embedder — shared by ingestion (writes the index) and the
agent (embeds queries), so a query vector and the chunk vectors are always produced
by the same model. Provider is a config change, not a code change; heavy imports
(torch / sentence-transformers) are lazy.
"""

from __future__ import annotations

import hashlib  # for fake embeddings (testing only)
import struct
import threading
from abc import ABC, abstractmethod
from typing import Sequence

from .config import Config


class Embedder(ABC):
    """Embedder behind a thin interface so callers never name a provider."""

    dimension: int

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts."""

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class LocalSentenceTransformerEmbedder(Embedder):
    """BGE-M3 (or any sentence-transformers model) run locally. torch is imported
    lazily on first construction."""

    def __init__(
        self,
        model_name: str,
        *,
        normalize: bool = True,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # lazy, heavy

        self._model = SentenceTransformer(model_name, device=device)
        self._normalize = normalize
        self._batch_size = batch_size
        self.dimension = self._model.get_embedding_dimension()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()


class HashEmbedder(Embedder):
    """Deterministic, dependency-free embedder for offline tests only — NO semantic
    meaning. Selected with ``embeddings.provider: hash``."""

    def __init__(self, dimension: int = 1024) -> None:
        self.dimension = dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        out: list[float] = []
        counter = 0
        while len(out) < self.dimension:
            digest = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
            # uint32s mapped to [-1, 1); ints (not a float32 view) stay finite.
            for i in range(0, 32, 4):
                u = struct.unpack("<I", digest[i : i + 4])[0]
                out.append(u / 2_147_483_648.0 - 1.0)
            counter += 1
        vec = out[: self.dimension]
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


# Cache embedders by their config signature: the local model load is heavy (~seconds)
# and the agent embeds a query on every retrieval, so it must load once per process.
_CACHE: dict[tuple, Embedder] = {}
_LOCK = threading.Lock()


def _build_embedder(provider: str, emb: dict) -> Embedder:
    if provider == "hash":
        return HashEmbedder(dimension=emb.get("dimensions") or 1024)
    if provider in ("huggingface", "local", "sentence-transformers"):
        return LocalSentenceTransformerEmbedder(
            emb.get("model") or "BAAI/bge-m3",
            normalize=emb.get("normalize", True),
            batch_size=emb.get("batch_size", 64),
        )
    raise NotImplementedError(
        f"Embedding provider {provider!r} not wired yet; use huggingface or hash."
    )


def get_embedder(config: Config) -> Embedder:
    """Return the embedder for ``config.embeddings`` (default BGE-M3 locally;
    ``provider: hash`` selects the test embedder). Cached per config signature so
    the model is loaded once per process, not once per query."""
    emb = config.embeddings
    provider = (emb.get("provider") or "huggingface").lower()
    key = (provider, emb.get("model"), bool(emb.get("normalize", True)),
           emb.get("batch_size", 64), emb.get("dimensions") or 1024)

    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is None:
            cached = _build_embedder(provider, emb)
            _CACHE[key] = cached
        return cached
