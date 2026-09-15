"""Intent routing: what reaches retrieval, and what never should.

The reported bug: "Hi" was planned as an explain task, retrieval ran over the
textbook, and the tutor reported that the syllabus does not cover the word "Hi".
The router decides whether the syllabus is needed at all; whether the answer is
*in* it stays retrieval's question.
"""

from __future__ import annotations

import pytest

from agent.intent import CONVERSATIONAL, IntentRouter, normalise, word_count


@pytest.fixture
def router() -> IntentRouter:
    """Phrase pass only — no embedder, so nothing here depends on a model."""
    return IntentRouter()


# --- conversational ---------------------------------------------------------

@pytest.mark.parametrize("message,expected", [
    ("Hi", "greeting"),
    ("hello", "greeting"),
    ("Good morning!", "greeting"),
    ("مرحبا", "greeting"),
    ("السلام عليكم", "greeting"),
    ("صباح الخير", "greeting"),
    ("ezayak", "greeting"),
    ("bonjour", "greeting"),
    ("thank you", "thanks"),
    ("Thanks!", "thanks"),
    ("شكراً", "thanks"),
    ("شكرا جزيلا", "thanks"),
    ("merci", "thanks"),
    ("who are you", "meta"),
    ("what can you do?", "meta"),
    ("من أنت", "meta"),
    ("how are you", "chitchat"),
    ("كيف حالك", "chitchat"),
    ("bye", "chitchat"),
])
def test_conversational_turns_are_recognised(router, message, expected):
    decision = router.classify(message)
    assert decision.intent == expected
    assert decision.needs_syllabus is False


# --- questions --------------------------------------------------------------

@pytest.mark.parametrize("message", [
    "explain column subtraction",
    "ما هي عملية الطرح؟",
    "how are plants able to make their own food?",
    "hi, can you explain subtraction to me?",
    "what can you do with fractions in this lesson",
    "who was the first president of Egypt?",
    "thanks to photosynthesis, what do plants produce?",
])
def test_real_questions_reach_the_syllabus(router, message):
    """Each of these opens like small talk and is a question."""
    decision = router.classify(message)
    assert decision.intent == "question"
    assert decision.needs_syllabus is True


def test_an_empty_message_is_not_conversational(router):
    assert router.classify("").intent == "question"
    assert router.classify("   ").intent == "question"


def test_diacritics_and_alef_variants_do_not_change_the_verdict(router):
    assert router.classify("شكراً").intent == router.classify("شكرا").intent
    assert router.classify("أهلا").intent == "greeting"


def test_routing_can_be_switched_off():
    off = IntentRouter(enabled=False)
    assert off.classify("hello").intent == "question"


# --- the similarity pass ----------------------------------------------------

class _Embedder:
    """Matches a seed only when the normalised text is identical, so the test
    exercises the similarity path without depending on a real model."""

    def embed(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)

    @staticmethod
    def _vector(text):
        return [float(sum(ord(c) for c in text) % 97), float(len(text)), 1.0]


def test_the_similarity_pass_only_sees_short_messages():
    """A long message is a question whatever it resembles."""
    router = IntentRouter(embedder=_Embedder(), max_words=3)
    long_greeting = "hello there my friend how is the lesson going today"
    assert router.classify(long_greeting).intent == "question"


def test_an_embedder_failure_falls_back_to_a_question():
    class Broken:
        def embed(self, texts):
            raise RuntimeError("model unavailable")

        def embed_query(self, text):
            raise RuntimeError("model unavailable")

    router = IntentRouter(embedder=Broken())
    assert router.classify("howdy").intent == "question"
    assert router.classify("hello").intent == "greeting"      # phrase pass still works


# --- the guarantee ----------------------------------------------------------

def test_a_greeting_never_reaches_retrieval(monkeypatch):
    """The point of the whole layer: no vector search, no LLM, no 'not in the
    syllabus' for someone saying hello."""
    from agent import retrieval
    from agent.graph import nodes

    calls: list[str] = []
    monkeypatch.setattr(retrieval, "rag_search",
                        lambda *a, **k: calls.append("search") or [])
    monkeypatch.setattr(retrieval, "get_topic_deps",
                        lambda *a, **k: calls.append("deps") or [])

    from agent.config import Config
    route = nodes.build_route(Config.load(), embedder=None)
    for message in ("Hi", "شكراً", "who are you", "bye"):
        out = route({"message": message, "language": "en"})
        assert out["tasks"] == []
        assert out["response"]
        assert out["intent"] in CONVERSATIONAL
    assert calls == []


def test_a_conversational_reply_is_in_the_student_s_language():
    from agent.config import Config
    from agent.graph import nodes

    route = nodes.build_route(Config.load(), embedder=None)
    english = route({"message": "hello", "language": "en"})["response"]
    arabic = route({"message": "hello", "language": "ar"})["response"]
    assert english and arabic and english != arabic


def test_a_question_is_left_for_the_planner():
    from agent.config import Config
    from agent.graph import nodes

    route = nodes.build_route(Config.load(), embedder=None)
    out = route({"message": "explain column subtraction", "language": "en"})
    assert out["intent"] == "question"
    assert "response" not in out and "tasks" not in out


# --- helpers ----------------------------------------------------------------

def test_normalise_folds_punctuation_and_case():
    assert normalise("  HELLO!!  ") == "hello"
    assert word_count("how are you?") == 3


# --- startup cost -----------------------------------------------------------

def test_routing_never_builds_the_embedder():
    """The deploy regression: resolving the embedder while building the graph
    blocked startup on a multi-gigabyte model download, so the health check
    failed and ECS killed the task in a loop."""
    built: list[str] = []

    def factory():
        built.append("built")               # a cold process: nothing loaded yet

    router = IntentRouter(embedder=factory)
    assert router.classify("hello").intent == "greeting"     # phrase pass
    assert built == [], "the phrase pass asked for an embedder"

    assert router.classify("howdy partner").intent == "question"
    assert built == ["built"], "the similarity pass should have asked once"


def test_the_similarity_pass_warms_up_later():
    """A cold turn must not permanently disable the similarity pass — the model
    usually loads on the first retrieval, after routing has already run."""
    state = {"loaded": False}

    def factory():
        return _Embedder() if state["loaded"] else None

    router = IntentRouter(embedder=factory, max_words=6)
    router.classify("howdy")              # cold: phrase pass only
    state["loaded"] = True
    assert router._seed_vectors(), "seeds should compute once the model is there"


def test_building_the_graph_does_not_load_a_model(monkeypatch):
    from shared import embeddings

    def explode(*args, **kwargs):
        raise AssertionError("graph construction must not build an embedder")

    monkeypatch.setattr(embeddings, "_build_embedder", explode)

    from agent.config import Config
    from agent.graph.graph import build_graph
    build_graph(Config.load())
