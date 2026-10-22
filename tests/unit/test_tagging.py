"""Topic suggestions for questions a teacher writes: the link that lets a quiz answer
move the student's mastery of the right topic."""

from __future__ import annotations

from agent import tagging
from agent.config import get_config
from agent.models import ResolvedTopic


def test_each_question_gets_its_nearest_topics_with_their_lesson(monkeypatch):
    """A teacher-made question carried no topic, so its graded answer could not be
    attributed and moved no mastery at all."""
    by_text = {
        "What is 3/4 + 1/8?": [ResolvedTopic("t-frac", "Adding fractions", 0.82)],
        "Solve x^2 = 9": [ResolvedTopic("t-quad", "Quadratic equations", 0.77),
                          ResolvedTopic("t-root", "Square roots", 0.61)],
    }
    monkeypatch.setattr(tagging, "resolve_topics_many",
                        lambda qs, sid=None, **k: [by_text.get(q, []) for q in qs])
    monkeypatch.setattr(tagging, "get_topic_lectures",
                        lambda ids, **k: {"t-frac": "l1", "t-quad": "l2", "t-root": "l2"})

    out = tagging.suggest_topics(
        [{"text": "What is 3/4 + 1/8?", "question_id": "q1"}, {"text": "Solve x^2 = 9"}],
        "math", config=get_config())

    assert out["degraded"] is False
    first, second = out["suggestions"]
    assert first["question_id"] == "q1" and first["resolved"] is True
    assert first["topics"] == [{"topic_id": "t-frac", "label": "Adding fractions",
                                "score": 0.82, "lecture_id": "l1"}]
    assert [t["topic_id"] for t in second["topics"]] == ["t-quad", "t-root"]


def test_a_question_nothing_matches_is_told_apart_from_a_failed_lookup(monkeypatch):
    """An empty topic list is also what a failed lookup looks like; a teacher shown
    "no matching topic" for an outage would tag the question by hand for nothing."""
    monkeypatch.setattr(tagging, "resolve_topics_many", lambda qs, sid=None, **k: [[]])
    monkeypatch.setattr(tagging, "get_topic_lectures", lambda ids, **k: {})
    matched_nothing = tagging.suggest_topics([{"text": "unrelated"}], "math",
                                             config=get_config())

    def _unreachable(qs, sid=None, **k):
        raise RuntimeError("vector store unreachable")

    monkeypatch.setattr(tagging, "resolve_topics_many", _unreachable)
    failed = tagging.suggest_topics([{"text": "unrelated"}, {"text": "other"}], "math",
                                    config=get_config())

    assert matched_nothing["degraded"] is False
    assert matched_nothing["suggestions"][0]["resolved"] is True
    assert failed["degraded"] is True
    assert [s["resolved"] for s in failed["suggestions"]] == [False, False]
    assert all(s["topics"] == [] for s in failed["suggestions"])


def test_a_whole_quiz_is_resolved_in_one_lookup(monkeypatch):
    """Twenty questions made twenty embeddings and twenty searches one after
    another; under load the teacher's request passed CloudFront's 30 s limit."""
    calls = []

    def _resolve(qs, sid=None, **k):
        calls.append(list(qs))
        return [[] for _ in qs]

    monkeypatch.setattr(tagging, "resolve_topics_many", _resolve)
    monkeypatch.setattr(tagging, "get_topic_lectures", lambda ids, **k: {})

    tagging.suggest_topics([{"text": f"question {i}"} for i in range(20)], "math",
                           config=get_config())

    assert len(calls) == 1 and len(calls[0]) == 20


def test_a_failed_lesson_lookup_marks_the_suggestions_degraded(monkeypatch):
    """Suggestions without their lesson look complete, but the grader then has a topic
    and no lesson, which is not enough to record mastery."""
    def _unreachable(*a, **k):
        raise RuntimeError("database unreachable")

    monkeypatch.setattr(tagging, "resolve_topics_many",
                        lambda qs, sid=None, **k: [[ResolvedTopic("t1", "Fractions", 0.8)]])
    monkeypatch.setattr(tagging, "get_topic_lectures", _unreachable)

    out = tagging.suggest_topics([{"text": "3/4 + 1/8"}], "math", config=get_config())

    assert out["degraded"] is True
    assert out["suggestions"][0]["topics"][0]["lecture_id"] is None


def test_many_queries_share_one_embedding_batch_and_one_search(monkeypatch):
    """resolve_topics embedded and searched one query at a time; a batch of them
    is one model call and one Qdrant round trip, with each query's hits kept
    apart and in order."""
    from agent import tools

    class _Embedder:
        calls = 0

        def embed(self, texts):
            _Embedder.calls += 1
            return [[float(i)] for i, _ in enumerate(texts)]

    class _Point:
        def __init__(self, topic_id):
            self.id, self.score = topic_id, 0.9
            self.payload = {"topic_id": topic_id, "label": topic_id, "subject_id": "math"}

    class _Client:
        batches = []

        def query_batch_points(self, collection, requests):
            _Client.batches.append(requests)
            return [type("R", (), {"points": [_Point(f"t{int(r.query[0])}")]})()
                    for r in requests]

    monkeypatch.setattr(tools, "get_embedder", lambda config: _Embedder())
    monkeypatch.setattr(tools, "query_client", lambda config: _Client())

    found = tools.resolve_topics_many(["a", "b", "c"], "math", config=get_config())

    assert _Embedder.calls == 1 and len(_Client.batches) == 1
    assert [[t.topic_id for t in hits] for hits in found] == [["t0"], ["t1"], ["t2"]]


def test_a_topic_found_twice_for_one_question_is_suggested_once(monkeypatch):
    """Two hits for the same topic showed the teacher the same skill twice, and the
    save sent its id twice in the question's topic list."""
    monkeypatch.setattr(tagging, "resolve_topics_many", lambda qs, sid=None, **k: [[
        ResolvedTopic("t-sub", "Subtraction", 0.71),
        ResolvedTopic("t-borrow", "Borrowing", 0.64),
        ResolvedTopic("t-sub", "Subtraction", 0.69)]])
    monkeypatch.setattr(tagging, "get_topic_lectures", lambda ids, **k: {})

    out = tagging.suggest_topics([{"text": "15 - 7"}], "math", config=get_config())

    topics = out["suggestions"][0]["topics"]
    assert [t["topic_id"] for t in topics] == ["t-sub", "t-borrow"]
    assert topics[0]["score"] == 0.71
