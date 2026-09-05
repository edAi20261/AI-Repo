"""Topic extraction and dependency derivation.

Domain logic that sits on top of the LLM seam: it names a lesson's skills and
proposes prerequisites, validates the model output through the schema contracts,
and clamps to the configured limits. It consumes ``llm_provider`` (``get_llm``)
but is not part of it — the provider layer stays free of pipeline knowledge.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from .config import Config
from .llm_provider import get_llm, llm_key_available
from .prompts import DERIVE_DEPS_PROMPT, TOPIC_EXTRACTION_PROMPT
from .schemas import DependencyItem, DependencyList, TopicItem, TopicList
from .utils.tokens import count_tokens


# --- shared helpers --------------------------------------------------------

def _message_content(message) -> str:
    """Text of a LangChain message or a plain string."""
    if isinstance(message, str):
        return message
    content = getattr(message, "content", message)
    if isinstance(content, list):  # some providers return content parts
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _structured_output(llm, prompt: str, model_cls):
    """Provider-native structured output, or ``None`` when unsupported/failed.

    Kept optional and defensive: gateways (e.g. Qwen/Kimi over OpenRouter) that do not
    honour ``json_schema`` fall through to the tolerant text parser rather than error."""
    factory = getattr(llm, "with_structured_output", None)
    if factory is None:
        return None
    try:
        result = factory(model_cls).invoke(prompt)
    except Exception:
        return None
    return result if isinstance(result, model_cls) else None


def _json_array(content: str) -> list:
    """Recover a JSON array from a response that may wrap it in a fence or prose."""
    import json
    import re

    text = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        raw = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return raw if isinstance(raw, list) else []


# --- topics ----------------------------------------------------------------

class TopicExtractor(ABC):
    """Extract 3-8 topics (skills) from a lesson's chunk texts."""

    is_stub: bool = False

    @abstractmethod
    def extract(self, chunk_texts: Sequence[str]) -> list[dict]:
        """Return ``{label, description, bloom_level}`` dicts."""


class StubTopicExtractor(TopicExtractor):
    """Returns no topics when no chat-LLM key is configured. Chunks get empty
    ``topic_ids``, which RAG does not need."""

    is_stub = True

    def extract(self, chunk_texts: Sequence[str]) -> list[dict]:
        return []


class LLMTopicExtractor(TopicExtractor):
    """Extract topics with any chat LLM (an object with ``.invoke(str) -> message``).
    A lesson whose chunks exceed ``max_input_tokens`` is processed in batches and the
    topics unioned (label-deduped), so a large or mis-split lesson never sends one
    oversized prompt. Uses provider-native structured output when available, falling
    back to tolerant JSON parsing."""

    def __init__(
        self, llm, *, min_topics: int, max_topics: int, max_input_tokens: int = 0
    ) -> None:
        self._llm = llm
        self._min = min_topics
        self._max = max_topics
        self._max_input_tokens = max_input_tokens

    def extract(self, chunk_texts: Sequence[str]) -> list[dict]:
        if not chunk_texts:
            return []
        seen: dict[str, dict] = {}
        for batch in self._batches(chunk_texts):
            for topic in self._extract_batch(batch):
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

    def _extract_batch(self, batch: Sequence[str]) -> list[dict]:
        prompt = TOPIC_EXTRACTION_PROMPT.format(
            min_topics=self._min,
            max_topics=self._max,
            chunks="\n\n---\n\n".join(batch),
        )
        structured = _structured_output(self._llm, prompt, TopicList)
        if structured is not None:
            return [t.model_dump() for t in structured.topics if t.label]
        return _parse_topics(_message_content(self._llm.invoke(prompt)))


def _parse_topics(content: str) -> list[dict]:
    """Validate each item through :class:`TopicItem`; drop empty-label items."""
    out: list[dict] = []
    for item in _json_array(content):
        if not isinstance(item, dict):
            continue
        topic = TopicItem.model_validate(item)
        if topic.label:
            out.append(topic.model_dump())
    return out


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
    )


# --- dependencies ----------------------------------------------------------

class DependencyDeriver(ABC):
    """Choose a topic's prerequisites from a list of candidate topic labels."""

    is_stub: bool = False

    @abstractmethod
    def derive(self, topic_label: str, candidate_labels: Sequence[str]) -> list[dict]:
        """Return ``{prereq, strength, rationale}`` dicts (prereq is a candidate label)."""


class StubDependencyDeriver(DependencyDeriver):
    """Returns no prerequisites when no chat-LLM key is configured."""

    is_stub = True

    def derive(self, topic_label: str, candidate_labels: Sequence[str]) -> list[dict]:
        return []


class LLMDependencyDeriver(DependencyDeriver):
    """Derive prerequisites with any chat LLM. One call per topic."""

    def __init__(self, llm) -> None:
        self._llm = llm

    def derive(self, topic_label: str, candidate_labels: Sequence[str]) -> list[dict]:
        if not candidate_labels:
            return []
        candidates = set(candidate_labels)
        prompt = DERIVE_DEPS_PROMPT.format(
            topic=topic_label,
            candidates="\n".join(f"- {c}" for c in candidate_labels),
        )
        structured = _structured_output(self._llm, prompt, DependencyList)
        if structured is not None:
            items = (d.model_dump() for d in structured.dependencies)
        else:
            items = _parse_deps(_message_content(self._llm.invoke(prompt)))
        # Only labels from the candidate list survive.
        return [d for d in items if d["prereq"] in candidates]


def _parse_deps(content: str) -> list[dict]:
    """Validate each item through :class:`DependencyItem` (candidate filtering is the
    caller's job)."""
    out: list[dict] = []
    for item in _json_array(content):
        if not isinstance(item, dict):
            continue
        out.append(DependencyItem.model_validate(item).model_dump())
    return out


def get_dependency_deriver(config: Config) -> DependencyDeriver:
    """The dependency deriver. Stub unless a key is present for the ``derive_deps``
    task."""
    spec = config.llm_for("derive_deps")
    if not llm_key_available(config, spec):
        return StubDependencyDeriver()
    return LLMDependencyDeriver(get_llm(config, "derive_deps"))
