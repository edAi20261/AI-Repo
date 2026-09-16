"""Topic extraction and dependency derivation.

Domain logic that sits on top of the LLM seam: it names a lesson's skills and
proposes prerequisites, validates the model output through the schema contracts,
and clamps to the configured limits. It consumes ``llm_provider`` (``get_llm``)
but is not part of it — the provider layer stays free of pipeline knowledge.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from shared.matching import written_in
from shared.structured import call_structured, schema_hint

from .config import Config
from .llm_provider import get_llm, llm_key_available
from .prompts import DERIVE_DEPS_PROMPT, TOPIC_EXTRACTION_PROMPT
from .schemas import DependencyList, TopicList
from .utils.tokens import count_tokens

# --- shared helpers --------------------------------------------------------

class TopicExtractor(ABC):
    """Extract 3-8 topics (skills) from a lesson's chunk texts."""

    is_stub: bool = False

    @abstractmethod
    def extract(self, chunk_texts: Sequence[str], language: str = "en") -> list[dict]:
        """Return ``{label, description, bloom_level}`` dicts."""


class StubTopicExtractor(TopicExtractor):
    """Returns no topics when no chat-LLM key is configured. Chunks get empty
    ``topic_ids``, which RAG does not need."""

    is_stub = True

    def extract(self, chunk_texts: Sequence[str], language: str = "en") -> list[dict]:
        return []


class LLMTopicExtractor(TopicExtractor):
    """Extract topics with any chat LLM (an object with ``.invoke(str) -> message``).
    A lesson whose chunks exceed ``max_input_tokens`` is processed in batches and the
    topics unioned (label-deduped), so a large or mis-split lesson never sends one
    oversized prompt."""

    def __init__(
        self, llm, *, min_topics: int, max_topics: int, max_input_tokens: int = 0,
        config: Config | None = None,
    ) -> None:
        self._llm = llm
        self._min = min_topics
        self._max = max_topics
        self._max_input_tokens = max_input_tokens
        self._config = config

    def extract(self, chunk_texts: Sequence[str], language: str = "en") -> list[dict]:
        if not chunk_texts:
            return []
        seen: dict[str, dict] = {}
        for batch in self._batches(chunk_texts):
            for topic in self._extract_batch(batch, language):
                seen.setdefault(topic["label"].lower(), topic)
        return list(seen.values())[: self._max]

    def _batches(self, chunk_texts: Sequence[str]) -> list[list[str]]:
        """Pack chunks greedily under the token budget; one batch when unset."""
        if self._max_input_tokens <= 0:
            return [list(chunk_texts)]
        batches: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0
        for text in chunk_texts:
            tokens = count_tokens(text)
            if current and current_tokens + tokens > self._max_input_tokens:
                batches.append(current)
                current, current_tokens = [], 0
            current.append(text)
            current_tokens += tokens
        if current:
            batches.append(current)
        return batches or [list(chunk_texts)]

    def _extract_batch(self, batch: Sequence[str], language: str = "en") -> list[dict]:
        # A label is shown to the student, so it follows the lesson's language.
        # Without this the model defaults to English and an Arabic poetry lesson
        # comes back tagged "Evaluate poetic structure".
        prompt = TOPIC_EXTRACTION_PROMPT.render(
            language=language,
            min_topics=self._min,
            max_topics=self._max,
            chunks="\n\n---\n\n".join(batch),
            schema=schema_hint(TopicList),
        )
        result = call_structured(
            TopicList, prompt, task="extract_topics", config=self._config,
            llm=self._llm, label="topics",
            validate=lambda batch: _labels_in(batch, language),
        )
        return [t.model_dump() for t in result.topics if t.label]


def _labels_in(batch: TopicList, language: str) -> None:
    """A topic label is shown to the student under their answer, so it has to be
    in the lesson's language.

    The schema said so all along — "the skill, 2-6 words, in the lesson's main
    language" — but a description is only a hint, and pydantic checks that a
    label is a string, never which language it is in. So Arabic lessons came back
    tagged "Evaluate poetic structure" and the rows were written.

    Failing here hands the model the same feedback any other validation failure
    gets: what it sent, and what was required. It corrects itself, and nothing
    wrong reaches the table.
    """
    wrong = [t.label for t in batch.topics if t.label and not written_in(t.label, language)]
    if wrong:
        raise ValueError(
            f"every label must be written in {language!r}, the language of this "
            f"lesson. These are not: {wrong}")


def get_topic_extractor(config: Config) -> TopicExtractor:
    """The topic extractor. Stub unless a chat-LLM key is present for the provider
    configured on the ``extract_topics`` task."""
    spec = config.llm_for("extract_topics")
    if not llm_key_available(config, spec):
        return StubTopicExtractor()
    topics_cfg = config.get("ingestion.topics", {})
    return LLMTopicExtractor(
        get_llm(config, "extract_topics"),
        min_topics=topics_cfg.get("min_per_lecture", 3),
        max_topics=topics_cfg.get("max_per_lecture", 8),
        max_input_tokens=topics_cfg.get("max_input_tokens", 0),
        config=config,
    )


# --- dependencies ----------------------------------------------------------

class DependencyDeriver(ABC):
    """Choose a topic's prerequisites from a list of candidate topic labels."""

    is_stub: bool = False

    @abstractmethod
    def derive(self, topic_label: str, candidate_labels: Sequence[str]) -> list[dict]:
        """Return ``{prereq, strength}`` dicts (prereq is a candidate label)."""


class StubDependencyDeriver(DependencyDeriver):
    """Returns no prerequisites when no chat-LLM key is configured."""

    is_stub = True

    def derive(self, topic_label: str, candidate_labels: Sequence[str]) -> list[dict]:
        return []


class LLMDependencyDeriver(DependencyDeriver):
    """Derive prerequisites with any chat LLM. One call per topic."""

    def __init__(self, llm, config: Config | None = None) -> None:
        self._llm = llm
        self._config = config

    def derive(self, topic_label: str, candidate_labels: Sequence[str]) -> list[dict]:
        if not candidate_labels:
            return []
        candidates = set(candidate_labels)
        prompt = DERIVE_DEPS_PROMPT.render(
            topic=topic_label,
            candidates="\n".join(f"- {c}" for c in candidate_labels),
            schema=schema_hint(DependencyList),
        )
        result = call_structured(
            DependencyList, prompt, task="derive_deps", config=self._config,
            llm=self._llm, label="dependencies",
        )
        # Only labels from the candidate list survive: a prerequisite the model
        # invented has no topic to point at.
        return [d.model_dump() for d in result.dependencies if d.prereq in candidates]


def get_dependency_deriver(config: Config) -> DependencyDeriver:
    """The dependency deriver. Stub unless a key is present for the ``derive_deps``
    task."""
    spec = config.llm_for("derive_deps")
    if not llm_key_available(config, spec):
        return StubDependencyDeriver()
    return LLMDependencyDeriver(get_llm(config, "derive_deps"), config)
