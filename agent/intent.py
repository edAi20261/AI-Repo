"""What a student's message is, before deciding what to do about it.

Not every turn is a question about the syllabus. "Hi", "thanks", "what can you
do?" were being planned as though they were: the planner's only task types are
explain/summarize/assess/grade/visual/report, so a greeting became an explain
task, retrieval ran over the textbook, and the model dutifully reported that the
syllabus does not cover the word "Hi".

This decides whether the syllabus is needed at all — not whether the answer is in
it. That second question still belongs to retrieval, which can see the material.

Two passes, cheapest first: an exact phrase match over short messages, then
similarity against seed utterances using the embedder the service already loads
for retrieval. A message that clears neither is treated as a real question.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from shared.language.request import language_instruction_only
from shared.matching import fold_for_match
from shared.language.script import ARABIC

logger = logging.getLogger(__name__)

Intent = Literal["greeting", "thanks", "meta", "chitchat", "language", "question"]

CONVERSATIONAL: tuple[Intent, ...] = (
    "greeting", "thanks", "meta", "chitchat", "language")

DEFAULT_MAX_WORDS = 6
DEFAULT_SIMILARITY = 0.72

# Seeds are utterances, not keywords: the second pass compares meaning, so
# "how's it going?" matches without being listed. Arabizi is included because
# Egyptian students type it constantly and no detector handles it.
SEEDS: dict[Intent, tuple[str, ...]] = {
    "greeting": (
        "hi", "hello", "hey", "good morning", "good evening", "welcome",
        "مرحبا", "اهلا", "السلام عليكم", "صباح الخير", "مساء الخير",
        "salam", "ahlan", "izayak", "ezayak",
        "bonjour", "salut", "hallo", "guten morgen",
    ),
    "thanks": (
        "thank you", "thanks", "thanks a lot", "that helped", "appreciate it",
        "شكرا", "شكرا جزيلا", "تسلم", "جزاك الله خيرا",
        "shukran", "merci", "danke",
    ),
    "meta": (
        "what can you do", "who are you", "what are you", "how do you work",
        "what do you know", "can you help me", "how can you help",
        "ماذا تستطيع أن تفعل", "من أنت", "كيف تعمل", "بماذا يمكنك مساعدتي",
        "que peux-tu faire", "was kannst du",
    ),
    "chitchat": (
        "how are you", "how is it going", "are you ok", "good night", "bye",
        "goodbye", "see you", "nice to meet you",
        "كيف حالك", "عامل ايه", "تصبح على خير", "مع السلامة", "إلى اللقاء",
        "comment ca va", "wie geht es dir",
    ),
}

_PUNCTUATION = re.compile(r"[!?.,؛،:؟\-_*\"'()\[\]]+")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class IntentDecision:
    """What the message is, and what decided it."""

    intent: Intent
    source: Literal["phrase", "similarity", "default"]
    confidence: float = 1.0
    evidence: str | None = None

    @property
    def needs_syllabus(self) -> bool:
        return self.intent == "question"

    def log_fields(self) -> dict[str, object]:
        return {"intent": self.intent, "intent_source": self.source,
                "intent_confidence": round(self.confidence, 3)}


def normalise(text: str) -> str:
    """Fold a message to the form the seed phrases are matched in.

    The same fold retrieval uses, plus the two things an exact dictionary lookup
    needs and a vector comparison does not: case and punctuation. It used to be a
    second hand-written copy that folded less — it missed ٱ, ئ, ة and ؤ — so a
    student writing "مدرسة" and one writing "مدرسه" took different paths here
    than they did in search.
    """
    folded = fold_for_match(text).lower()
    return _WHITESPACE.sub(" ", _PUNCTUATION.sub(" ", folded)).strip()


def word_count(text: str) -> int:
    return len(normalise(text).split())


class IntentRouter:
    """Classifies a turn as conversational or a real question."""

    def __init__(
        self,
        *,
        embedder=None,
        max_words: int = DEFAULT_MAX_WORDS,
        similarity: float = DEFAULT_SIMILARITY,
        enabled: bool = True,
        acknowledge: bool = True,
    ) -> None:
        self._embedder = embedder
        self._acknowledge = acknowledge
        self._max_words = max_words
        self._similarity = similarity
        self._enabled = enabled
        self._phrases = {
            normalise(seed): intent
            for intent, seeds in SEEDS.items()
            for seed in seeds
        }
        self._vectors: list[tuple[Intent, list[float]]] | None = None

    @classmethod
    def from_config(cls, config, embedder=None) -> IntentRouter:
        return cls(
            embedder=embedder,
            max_words=config.get("agent.intent.max_words", DEFAULT_MAX_WORDS),
            similarity=config.get("agent.intent.similarity", DEFAULT_SIMILARITY),
            enabled=config.get("agent.intent.enabled", True),
            acknowledge=config.get("language.sticky.acknowledge", True),
        )

    def classify(self, message: str) -> IntentDecision:
        """What this message is. Anything unrecognised is a question."""
        text = message or ""

        # "reply in Arabic" is an instruction, not a question about the word
        # "reply". Deterministic and ahead of the enabled gate, because an
        # instruction must be honoured even where similarity routing is off.
        if self._acknowledge:
            instruction = language_instruction_only(text)
            if instruction:
                return IntentDecision("language", "phrase", evidence=instruction[0])

        if not self._enabled or not text.strip():
            return IntentDecision("question", "default")

        folded = normalise(text)
        matched = self._phrases.get(folded)
        if matched:
            return IntentDecision(matched, "phrase", evidence=folded)

        # Only short messages are considered conversational. "How are plants
        # able to make food?" opens like small talk and is a real question.
        if word_count(text) > self._max_words:
            return IntentDecision("question", "default")

        similar = self._by_similarity(folded)
        if similar is not None:
            return similar
        return IntentDecision("question", "default")

    def _by_similarity(self, folded: str) -> IntentDecision | None:
        vectors = self._seed_vectors()
        if not vectors or not folded:
            return None
        embedder = self._resolve_embedder()
        if embedder is None:
            return None
        try:
            query = embedder.embed_query(folded)
        except Exception:
            logger.exception("intent embedding failed; treating the turn as a question")
            return None

        best_intent, best_score = None, 0.0
        for intent, vector in vectors:
            score = _cosine(query, vector)
            if score > best_score:
                best_intent, best_score = intent, score
        if best_intent is None or best_score < self._similarity:
            return None
        return IntentDecision(best_intent, "similarity", confidence=best_score, evidence=folded)

    def _resolve_embedder(self):
        """The embedder, if one is available right now.

        ``embedder`` may be a model or a callable that returns one — or returns
        ``None`` when it is not loaded yet. Routing must never be what triggers a
        model load: it has a phrase pass that answers without one, and blocking
        here would block a greeting behind a multi-gigabyte download.
        """
        source = self._embedder
        if source is None or hasattr(source, "embed"):
            return source
        try:
            return source()
        except Exception:
            logger.exception("intent embedder unavailable; falling back to phrases")
            return None

    def _seed_vectors(self) -> list[tuple[Intent, list[float]]]:
        """Seed embeddings, computed once the embedder is available.

        Not cached when it is unavailable: the model usually loads on the first
        retrieval, so a later turn can still get the similarity pass.
        """
        if self._vectors is not None:
            return self._vectors
        embedder = self._resolve_embedder()
        if embedder is None:
            return []
        seeds = [(intent, seed) for intent, group in SEEDS.items() for seed in group]
        try:
            vectors = embedder.embed([normalise(seed) for _, seed in seeds])
        except Exception:
            logger.exception("seed embedding failed; intent routing falls back to phrases")
            self._vectors = []
            return self._vectors
        self._vectors = [(intent, vector) for (intent, _), vector in zip(seeds, vectors)]
        return self._vectors


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def is_arabic_script(text: str) -> bool:
    return bool(ARABIC.search(text or ""))
