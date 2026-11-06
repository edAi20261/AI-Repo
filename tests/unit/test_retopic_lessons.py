"""Lessons split out of a book whose topics were found once, for the whole book.

Every lesson of a subject carried its first lessons' eight topics: "Lesson Two
Chemical Equations" was tagged with states of matter, and a quiz on it asked
about heating curves."""

import math

from scripts.retopic_lessons import PlanStore, payload_groups, plan_lesson

LESSON = "6f1b7d3e-0000-4000-8000-000000000002"
FIRST = "6f1b7d3e-0000-4000-8000-000000000001"
SUBJECT = "6f1b7d3e-0000-4000-8000-0000000000aa"
BORROWED = "6f1b7d3e-0000-4000-8000-0000000000b1"


def _unit(*xs):
    norm = math.sqrt(sum(x * x for x in xs))
    return [x / norm for x in xs]


# Close enough to a chemistry label (0.7) for the 0.60 reuse of a lesson's own
# topics, far from the 0.92 a shared skill needs.
STATES = _unit(0.7, 0.71414, 0.0)
EQUATIONS = _unit(1.0, 0.0, 0.0)
VECTORS = {"Explain heating curves": STATES, "Balance chemical equations": EQUATIONS}


def _database(sql, params):
    if "WHERE lecture_id = %s" in sql:          # owned by the lesson
        return []
    if "WHERE subject_id = %s" in sql:          # the subject's catalog
        return [(BORROWED, "Explain heating curves", STATES)]
    if "topic_id = ANY" in sql:                 # owners
        return [(BORROWED, FIRST)] if BORROWED in params[0] else []
    raise AssertionError(sql)


class _Extractor:
    is_stub = False

    def extract(self, texts, language):
        return [{"label": "Balance chemical equations", "bloom_level": "apply"}]


class _Embedder:
    def embed(self, labels):
        return [VECTORS[label] for label in labels]


class _NoDeriver:
    is_stub = True


def _context(store):
    from ingestion.context import PipelineContext
    from shared.config import get_config
    from scripts.retopic_lessons import PlanVectors

    config = get_config()
    config.app_env = "production"
    return PipelineContext(config=config, vision=None, uploader=None, _embedder=_Embedder(),
                           _topic_extractor=_Extractor(), _dependency_deriver=_NoDeriver(),
                           _chunk_store=store, _vector_store=PlanVectors())


def test_a_lesson_gets_topics_found_in_its_own_text_not_the_ones_it_carries():
    store = PlanStore(_database)
    passages = [("c1", "Balance the equation", "", 55, 55, "en", 0, 1)]
    planned = plan_lesson((LESSON, SUBJECT, "Lesson Two Chemical Equations", 17, "syllabus"),
                          passages, {"c1": EQUATIONS}, _context(store), store)

    assert planned["status"] == "planned"
    assert planned["labels"] == ["Balance chemical equations"]
    [topic] = store.planned
    assert topic.lecture_id == LESSON and topic.topic_id != BORROWED
    assert planned["passages"] == {"c1": [topic.topic_id]}


def test_a_lessons_own_topics_are_those_it_owns_not_those_its_passages_carry():
    """Its passages carry the borrowed topics; reusing them at 0.60 handed the
    same eight ids straight back."""
    calls = []
    store = PlanStore(lambda sql, params: calls.append(sql) or [])
    store.read_lecture_topics(LESSON)
    assert "unnest" not in calls[0] and "ai.chunks" not in calls[0]


def test_a_skill_planned_for_an_earlier_lesson_is_offered_to_the_next():
    """One skill taught in two lessons stays one topic."""
    from ingestion.state import Topic

    store = PlanStore(lambda sql, params: [])
    store.write_topics([Topic(label="Balance chemical equations", subject_id=SUBJECT,
                              lecture_id=FIRST, topic_id="t1", label_embedding=EQUATIONS)])
    assert [t[0] for t in store.read_topic_catalog(SUBJECT)] == ["t1"]
    assert store.read_topic_owners(["t1"]) == {"t1": FIRST}


def test_passages_tagged_alike_are_written_in_one_payload_update():
    groups = payload_groups({"c1": ["a"], "c2": ["a"], "c3": ["b", "a"]})
    assert groups == {("a",): ["c1", "c2"], ("b", "a"): ["c3"]}
