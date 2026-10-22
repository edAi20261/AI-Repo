"""A turn is bounded by what the student selected, and says so when it refuses.

A lesson id has always been able to narrow retrieval. What it could not do is
change the answer when nothing was found: a student sitting in "Photosynthesis"
who asked about the Second World War was told "there isn't enough syllabus
material" — which reads as a hole in the product rather than as the boundary
doing its job. The tutor knew it had been scoped and had no way to say so.
"""

from __future__ import annotations

import pytest

from agent.scope import StudyScope, lecture_filter, scope_of

LESSON = "11111111-1111-1111-1111-111111111111"
OTHER = "22222222-2222-2222-2222-222222222222"


# --- reading the selection --------------------------------------------------

def test_a_declared_scope_is_used_as_given():
    scope = scope_of({"study_scope": {"lecture_id": LESSON, "lesson_title": "Photosynthesis"}})
    assert scope.lecture_id == LESSON
    assert scope.is_narrow
    assert scope.label() == "Photosynthesis"


def test_a_single_legacy_lecture_id_still_reads_as_a_lesson_selection():
    """The platform sent lecture_ids long before this field existed, and still
    does. Requiring every caller to move at once would have meant no scoping at
    all until they did."""
    scope = scope_of({"lecture_ids": [LESSON]})
    assert scope.lecture_id == LESSON
    assert scope.is_narrow


def test_several_lessons_are_not_a_lesson_selection():
    """"Search these three lessons" is a filter, not a place the student is
    sitting — there is no single lesson to name in a refusal."""
    scope = scope_of({"lecture_ids": [LESSON, OTHER]})
    assert scope.lecture_id is None
    assert not scope.is_narrow


def test_a_multi_lesson_filter_survives_the_scope():
    """The regression this guards: reading the selection as one lesson must not
    quietly discard the other lessons a caller asked to search."""
    assert lecture_filter({"lecture_ids": [LESSON, OTHER]}) == [LESSON, OTHER]
    assert lecture_filter({"study_scope": {"lecture_id": LESSON}}) == [LESSON]
    assert lecture_filter({}) is None


def test_a_declared_scope_outranks_the_legacy_field():
    """Two sources for one decision disagree eventually; the declared one wins."""
    assert lecture_filter({
        "study_scope": {"lecture_id": LESSON},
        "lecture_ids": [OTHER],
    }) == [LESSON]


def test_a_blank_lesson_title_is_not_a_lesson_title():
    assert StudyScope(lecture_id=LESSON, lesson_title="   ").label() is None


# --- what the student is told -----------------------------------------------

def _refusal(scope: StudyScope) -> str:
    from agent.graph.subgraphs.explain_summarize_sg import _decline_without_model

    return _decline_without_model(scope)


def test_without_a_model_the_refusal_still_names_the_lesson():
    text = _refusal(StudyScope(lecture_id=LESSON, lesson_title="Photosynthesis"))
    assert "Photosynthesis" in text


def test_without_a_lesson_the_refusal_names_no_boundary():
    """Outside a lesson there is no boundary to explain, and claiming one would
    be a lie: the material genuinely is not in the syllabus."""
    assert _refusal(StudyScope(subject_id="s")) == "∅"
    assert _refusal(StudyScope(lecture_id=LESSON)) == "∅"


# --- does the selected lesson actually answer this? -------------------------

class _Hit:
    def __init__(self, score: float):
        self.score = score


def _within(monkeypatch, scoped: float, unscoped: float | None, ratio: float = 0.90) -> bool:
    from agent import retrieval

    monkeypatch.setattr(
        retrieval, "rag_search",
        lambda *a, **k: [_Hit(unscoped)] if unscoped is not None else [])
    return retrieval.answers_within_scope("q", ["syllabus"], scoped, ratio=ratio)


def test_a_question_the_lesson_owns_scores_the_same_with_or_without_the_filter(monkeypatch):
    """Measured: in-scope questions score 0.952 to 1.000 of the unscoped best."""
    assert _within(monkeypatch, scoped=0.600, unscoped=0.600)
    assert _within(monkeypatch, scoped=0.472, unscoped=0.496)


def test_a_question_another_lesson_answers_better_is_out_of_scope(monkeypatch):
    """Measured: out-of-scope questions reach at most 0.872 of the unscoped best.

    Inside one lesson the similarity floor cannot make this call — with the rest
    of the corpus filtered out there is nothing to lose to, so the lesson's own
    best chunk clears the floor whatever was asked."""
    assert not _within(monkeypatch, scoped=0.503, unscoped=0.650)   # the Youth poem
    assert not _within(monkeypatch, scoped=0.312, unscoped=0.652)   # electronic games


def test_nothing_retrieved_in_scope_is_not_in_scope(monkeypatch):
    assert not _within(monkeypatch, scoped=0.0, unscoped=0.650)


def test_a_corpus_with_nowhere_better_keeps_the_answer(monkeypatch):
    """If the unscoped search finds nothing at all there is no elsewhere to
    prefer, so the lesson is as good an answer as exists."""
    assert _within(monkeypatch, scoped=0.45, unscoped=None)


# --- the plan node must obey the declared scope too -------------------------

class _NoSubjects:
    def facts(self, subject_id):
        from agent.models import SubjectFacts
        return SubjectFacts()


def _quiz(query):
    return {"type": "assessment", "query": query, "level": "intermediate", "length": "medium"}


class _PlanCfg:
    """Minimal config for build_plan with no default subject."""

    def get(self, key, default=None):
        if key == "agent.default_tasks":
            return ["assessment"]
        if key == "language.supported":
            return ["ar", "en", "fr", "de"]
        return default


def test_planning_resolves_topics_inside_the_declared_lesson(monkeypatch):
    """A quiz inside a lesson came back empty because plan ignored study_scope.

    ``plan`` scoped topic resolution to the legacy ``lecture_ids`` only, so a turn
    carrying a declared ``study_scope`` resolved topics across the whole subject.
    The assessment adapter then narrowed retrieval to the selected lesson, the
    two filters intersected to nothing, and the student got "there isn't enough
    syllabus material" instead of their quiz.
    """
    from agent.graph import nodes as nmod

    seen: dict = {}

    def _spy(query, subject_id=None, *, lecture_ids=None, **kwargs):
        seen["lecture_ids"] = lecture_ids
        return []

    monkeypatch.setattr(nmod, "resolve_topics", _spy)
    plan = nmod.build_plan(_PlanCfg(), _NoSubjects())
    plan({
        "pathway": "student",
        "message": "اختبرني في هذا الدرس",
        "planned_tasks": [_quiz("اختبرني في هذا الدرس")],
        "subject_id": "subjX",
        "study_scope": {"subject_id": "subjX", "lecture_id": LESSON,
                        "lesson_title": "Photosynthesis"},
    })
    assert seen["lecture_ids"] == [LESSON]


def test_planning_still_honours_a_multi_lesson_legacy_filter(monkeypatch):
    """Several lecture_ids and no declared scope stays a multi-lesson filter."""
    from agent.graph import nodes as nmod

    seen: dict = {}

    def _spy(query, subject_id=None, *, lecture_ids=None, **kwargs):
        seen["lecture_ids"] = lecture_ids
        return []

    monkeypatch.setattr(nmod, "resolve_topics", _spy)
    plan = nmod.build_plan(_PlanCfg(), _NoSubjects())
    plan({"pathway": "student", "message": "quiz me", "planned_tasks": [_quiz("quiz me")],
          "subject_id": "subjX", "lecture_ids": [LESSON, OTHER]})
    assert seen["lecture_ids"] == [LESSON, OTHER]


class _Writer:
    def __init__(self, reply="", error=False):
        self.reply, self.error, self.prompt = reply, error, None

    def invoke(self, messages):
        self.prompt = messages[0].content
        if self.error:
            raise RuntimeError("model unavailable")

        class _Msg:
            content = self.reply
        return _Msg()


def test_the_tutor_writes_its_own_refusal_and_names_the_lesson():
    """A fixed sentence per language answered every miss the same way; the model
    writes it now, told which lesson bounds the turn and forbidden to answer."""
    from agent.graph.subgraphs.explain_summarize_sg import _declined

    writer = _Writer("That's outside Photosynthesis, so I can't answer it here.")
    scope = StudyScope(lecture_id=LESSON, lesson_title="Photosynthesis")
    text = _declined(scope, {"language": "en", "message": "who was Napoleon?",
                             "history": ""}, writer, "friendly")
    assert text == "That's outside Photosynthesis, so I can't answer it here."
    assert "Photosynthesis" in writer.prompt and "who was Napoleon?" in writer.prompt


def test_a_refusal_the_model_cannot_write_fails_the_turn():

    from agent.errors import TutorUnavailable
    from agent.graph.subgraphs.explain_summarize_sg import _declined

    scope = StudyScope(lecture_id=LESSON, lesson_title="Photosynthesis")
    with pytest.raises(TutorUnavailable):
        _declined(scope, {"language": "en", "message": "q"}, _Writer(error=True), "friendly")


def _offers(monkeypatch, topics):
    from agent.graph.subgraphs import explain_summarize_sg as sg

    asked = {}

    def offer(**kw):
        asked.update(kw)
        return topics

    monkeypatch.setattr(sg.retrieval, "topics_on_offer", offer)
    return asked


def test_a_refusal_suggests_only_topics_the_lessons_really_cover(monkeypatch):
    """Told to suggest what to ask instead and never told what the lessons hold, the
    model made topics up: a maths student was offered "The Water Cycle", an Arabic
    one "the history of the internet"."""
    from agent.graph.subgraphs.explain_summarize_sg import _declined

    asked = _offers(monkeypatch, ["Column subtraction", "Borrowing"])
    writer = _Writer("Your lessons don't cover that.")
    _declined(StudyScope(subject_id="maths"), {"language": "en", "message": "spider legs?",
                                               "subject_id": "maths"}, writer, "friendly")
    assert "Column subtraction; Borrowing" in writer.prompt
    assert asked == {"lecture_id": None, "subject_id": "maths", "limit": 5}


def test_inside_a_lesson_the_refusal_offers_that_lessons_topics(monkeypatch):
    from agent.graph.subgraphs.explain_summarize_sg import _declined

    asked = _offers(monkeypatch, ["Borrowing"])
    _declined(StudyScope(subject_id="maths", lecture_id=LESSON, lesson_title="Subtraction"),
              {"language": "en", "message": "q"}, _Writer("x"), "friendly")
    assert asked["lecture_id"] == LESSON


def test_a_refusal_that_knows_no_topics_is_told_to_name_none(monkeypatch):
    from agent.graph.subgraphs.explain_summarize_sg import _declined

    _offers(monkeypatch, [])
    writer = _Writer("Your lessons don't cover that.")
    _declined(StudyScope(), {"language": "en", "message": "q"}, writer, "friendly")
    assert "Do not name any topic" in writer.prompt
    assert "suggest what they could ask about instead" not in writer.prompt
# --- the selected subject is the turn's subject ------------------------------

def test_the_subject_a_student_selected_is_the_subject_the_turn_is_answered_in():
    """A turn that carried its subject only inside study_scope had it ignored: the
    subject was guessed from the words of the message, and a student sitting in
    the Arabic subject's Entrepreneurship lesson was answered as if in Arabic
    mathematics."""
    from system.schemas.chat import ChatRequest

    state = ChatRequest(user_id="u1", user_type="student", message="لخص لي هذا الدرس",
                        study_scope={"subject_id": "arabic", "lecture_id": LESSON}).to_state()
    assert state["subject_id"] == "arabic"


def test_a_subject_sent_on_the_turn_itself_still_wins():
    from system.schemas.chat import ChatRequest

    state = ChatRequest(user_id="u1", user_type="student", message="q", subject_id="maths",
                        study_scope={"subject_id": "arabic"}).to_state()
    assert state["subject_id"] == "maths"


# --- telling the client the question was outside the lesson ----------------

def _explained(monkeypatch, *, study_scope, whole_lesson=False, within=False, found=True):
    from agent.config import get_config
    from agent.graph.subgraphs import explain_summarize_sg as sg
    from agent.models import RetrievedChunk

    chunk = RetrievedChunk(chunk_id="c1", score=0.5, text="Napoleon was ...", topic_ids=[],
                           page_from=1, page_to=1, heading_path=None, language="en",
                           lecture_id=OTHER)
    monkeypatch.setattr(sg.retrieval, "retrieve", lambda *a, **k: [chunk] if found else [])
    monkeypatch.setattr(sg.retrieval, "whole_lesson", lambda *a, **k: [])
    monkeypatch.setattr(sg.retrieval, "answers_within_scope", lambda *a, **k: within)
    monkeypatch.setattr(sg.retrieval, "with_neighbours", lambda found, **k: found)
    graph = sg.build_explain_sg(config=get_config(), llm=None)
    return graph.invoke({
        "query": "who was Napoleon?", "message": "who was Napoleon?", "language": "en",
        "namespaces": ["syllabus"], "k": 8, "min_score": 0.47, "subject_id": "history",
        "study_scope": study_scope, "whole_lesson": whole_lesson})


def test_a_question_outside_the_selected_lesson_is_flagged_for_the_client(monkeypatch):
    """The reply named the lesson, but nothing a client could read said the
    question had fallen outside it, so no notice could be shown."""
    out = _explained(monkeypatch, study_scope={"subject_id": "history", "lecture_id": LESSON,
                                               "lesson_title": "Photosynthesis"})
    assert out["basis"] == "declined" and out["outside_lesson"] is True


def test_a_refusal_with_no_lesson_selected_is_not_outside_a_lesson(monkeypatch):
    out = _explained(monkeypatch, found=False, study_scope={"subject_id": "history"})
    assert out["basis"] == "declined" and not out.get("outside_lesson")


def test_an_empty_lesson_read_as_a_whole_is_not_outside_itself(monkeypatch):
    out = _explained(monkeypatch, whole_lesson=True,
                     study_scope={"subject_id": "history", "lecture_id": LESSON})
    assert out["basis"] == "declined" and not out.get("outside_lesson")


def test_an_answer_inside_the_lesson_is_not_flagged(monkeypatch):
    out = _explained(monkeypatch, within=True,
                     study_scope={"subject_id": "history", "lecture_id": LESSON})
    assert out["basis"] == "lesson" and not out.get("outside_lesson")


def test_the_api_says_when_a_question_was_outside_the_lesson():
    from shared.metering import Meter
    from system.schemas.chat import ChatResponse

    state = {"response": "That is outside Photosynthesis.", "language": "en",
             "answers": [{"type": "explain", "text": "x", "basis": "declined",
                          "outside_lesson": True}]}
    assert ChatResponse.from_state(state, "req-1", Meter()).outside_lesson is True
    state["answers"][0]["outside_lesson"] = False
    assert ChatResponse.from_state(state, "req-1", Meter()).outside_lesson is False
