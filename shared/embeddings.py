"""The bilingual index embedder — shared by ingestion (writes the index) and the
agent (embeds queries), so a query vector and the chunk vectors are always produced
by the same model. Provider is a config change, not a code change; heavy imports
(torch / sentence-transformers) are lazy.
"""

from __future__ import annotations

import hashlib  # for fake embeddings (testing only)
import logging
import struct
import threading
import time
from abc import ABC, abstractmethod
from typing import Sequence

from .config import Config
from .matching import fold_for_match

logger = logging.getLogger(__name__)

# BM25's term-frequency saturation, applied per document rather than per corpus:
# a passage that repeats a word ten times is not ten times the answer. The
# length normalisation uses a fixed expected length because a document is
# encoded before the corpus it joins exists — the alternative is re-encoding
# every chunk whenever another is added.
BM25_K1 = 1.2
BM25_B = 0.75
EXPECTED_TOKENS = 220


class Embedder(ABC):
    """Embedder behind a thin interface so callers never name a provider.

    Every text is folded on its way in — see ``shared/matching.py``. Doing it
    here rather than at the call sites is the point: a document folded while a
    query is not is worse than folding neither, and there are six call sites
    across ingestion and the agent. None of them can get it wrong from here.
    """

    dimension: int

    @abstractmethod
    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of already-folded texts."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        return self._encode([fold_for_match(t) for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def _terms(self, text: str) -> list[int]:
        """The text as term ids. Subclasses that can do better than whitespace
        override this; the fold has already been applied."""
        return [abs(hash(word)) % 250_000 for word in text.split()]

    def embed_sparse(self, texts: Sequence[str]) -> list[dict[int, float]]:
        """Each text as ``{term_id: weight}``, folded the same way ``embed`` folds.

        The lexical half of a hybrid search. A dense vector finds a passage that
        means the same thing; this finds the passage that uses the same words —
        a name, a formula, a date, the things a student types exactly and that
        an average over 1024 dimensions loses.
        """
        return [self._weigh(self._terms(fold_for_match(t))) for t in texts]

    @staticmethod
    def _weigh(terms: Sequence[int]) -> dict[int, float]:
        if not terms:
            return {}
        counts: dict[int, int] = {}
        for term in terms:
            counts[term] = counts.get(term, 0) + 1
        norm = BM25_K1 * (1 - BM25_B + BM25_B * len(terms) / EXPECTED_TOKENS)
        return {term: (count * (BM25_K1 + 1)) / (count + norm)
                for term, count in counts.items()}


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
        self._tokenizer = self._model.tokenizer
        self._normalize = normalize
        self._batch_size = batch_size
        self.dimension = self._model.get_embedding_dimension()

    def _terms(self, text: str) -> list[int]:
        """Term ids from the embedding model's own tokenizer.

        Splitting on whitespace gives a Chinese sentence one term and an Arabic
        page far fewer than it has words. The model's tokenizer is trained
        across a hundred languages, so the same code covers a corpus in any of
        them, and the ids are stable for the life of the model.
        """
        return self._tokenizer(text, add_special_tokens=False)["input_ids"]

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
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

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
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


def _cache_key(config: Config) -> tuple:
    emb = config.embeddings
    return ((emb.get("provider") or "huggingface").lower(), emb.get("model"),
            bool(emb.get("normalize", True)), emb.get("batch_size", 64),
            emb.get("dimensions") or 1024)


def loaded_embedder(config: Config) -> Embedder | None:
    """The embedder for ``config``, but only if it is already in memory.

    For callers that can use embeddings when they happen to be available and do
    something else when they are not. Building one loads a multi-gigabyte model:
    that cost belongs to retrieval, which cannot work without it, and never to a
    caller that has a cheaper fallback.
    """
    return _CACHE.get(_cache_key(config))


def get_embedder(config: Config) -> Embedder:
    """Return the embedder for ``config.embeddings`` (default BGE-M3 locally;
    ``provider: hash`` selects the test embedder). Cached per config signature so
    the model is loaded once per process, not once per query."""
    emb = config.embeddings
    provider = (emb.get("provider") or "huggingface").lower()
    key = _cache_key(config)

    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is None:
            cached = _build_embedder(provider, emb)
            _CACHE[key] = cached
        return cached


def warm_embedder(config: Config) -> None:
    """Load the embedder and run it once, so no student's question pays for it."""
    started = time.monotonic()
    try:
        get_embedder(config).embed_query("warm-up")
    except Exception:
        logger.error("embedder warm-up failed; the first question will load it", exc_info=True)
        return
    logger.info("embedder ready", extra={"seconds": round(time.monotonic() - started, 1)})
