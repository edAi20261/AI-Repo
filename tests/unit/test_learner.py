"""The tutor reads what the student has already shown before it explains.

``ai.mastery`` was written by grading from the first day and read only by the
parts that produce a number — assessment weighting, grading, the report. A
student could miss three questions on a topic and have the next explanation
pitched exactly as if they had never been assessed, because the part that talks
to them read the conversation and nothing else.
"""

from __future__ import annotations

import uuid

from agent.config import get_config
from agent.learner import NOTHING_RECORDED, LearnerStanding, standing
from agent.graph import nodes as nmod

USER = str(uuid.uuid4())
SUBJECT = str(uuid.uuid4())


def _topic(name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))

# (topic_id, score, attempts, last_seen, trend, confidence) — ai.mastery's shape.
WEAK = (_topic("t-weak"), 0.20, 4, None, -0.1, 1.0)
WEAKER = (_topic("t-weaker"), 0.05, 4, None, -0.2, 1.0)
STRONG = (_topic("t-strong"), 0.90, 4, None, 0.1, 1.0)
THIN = (_topic("t-thin"), 0.10, 1, None, None, 0.5)      # one attempt: not yet evidence


def _db(fake_postgres, mastery, labels):
    return fake_postgres(fetchall=[list(mastery), [(k, v) for k, v in labels.items()]])


def test_a_topic_the_student_got_wrong_reaches_the_tutor(fake_postgres):
    _db(fake_postgres, [WEAK], {_topic("t-weak"): "Fractions"})
    read = standing(USER, SUBJECT, config=get_config())
    assert read.weak == ["Fractions"] and not read.strong
    assert "Fractions" in read.render()


def test_a_thinly_sampled_topic_is_not_called_a_weakness(fake_postgres):
    """One wrong answer is not evidence. A tutor told a student is weak on
    something it has a single data point for teaches to a fact that isn't true."""
    _db(fake_postgres, [THIN], {_topic("t-thin"): "Photosynthesis"})
    assert not standing(USER, SUBJECT, config=get_config())


def test_the_weakest_topics_come_first_and_the_list_is_bounded(fake_postgres):
    rows = [WEAK, WEAKER] + [(_topic(f"t{i}"), 0.3, 4, None, None, 1.0) for i in range(8)]
    labels = {_topic("t-weak"): "Fractions", _topic("t-weaker"): "Long division",
              **{_topic(f"t{i}"): f"Topic {i}" for i in range(8)}}
    _db(fake_postgres, rows, labels)
    read = standing(USER, SUBJECT, config=get_config())
    assert read.weak[0] == "Long division"      # 0.05 before 0.20
    assert len(read.weak) == 5


def test_a_topic_with_no_label_is_dropped(fake_postgres):
    """A bare uuid in a prompt is noise the model has to ignore, or quote."""
    _db(fake_postgres, [WEAK, STRONG], {_topic("t-strong"): "Place value"})
    read = standing(USER, SUBJECT, config=get_config())
    assert read.weak == [] and read.strong == ["Place value"]


def test_an_unreadable_learner_model_does_not_cost_the_student_an_answer(fake_postgres):
    fake_postgres(explode=True)
    read = standing(USER, SUBJECT, config=get_config())
    assert not read and read.render() == NOTHING_RECORDED


def test_nothing_is_read_for_a_turn_with_no_student_or_no_subject():
    assert not standing(None, SUBJECT) and not standing(USER, None)


def test_a_standing_with_nothing_in_it_says_so():
    assert LearnerStanding().render() == NOTHING_RECORDED


def test_the_mastery_read_is_scoped_to_the_subject_in_play(fake_postgres):
    db = _db(fake_postgres, [WEAK], {_topic("t-weak"): "Fractions"})
    standing(USER, SUBJECT, topic_ids=[_topic("t-weak")], config=get_config())
    mastery_sql, params = db.statements[0]
    assert "t.subject_id = %s" in mastery_sql
    assert SUBJECT in params and [_topic("t-weak")] in params


# --- the turn -----------------------------------------------------------------

def test_the_planner_puts_the_standing_where_the_answer_can_read_it(monkeypatch):
    monkeypatch.setattr(nmod, "standing",
                        lambda *a, **k: LearnerStanding(weak=["Fractions"]))
    monkeypatch.setattr(nmod, "resolve_topics", lambda *a, **k: [])
    plan = nmod.build_plan(get_config(), nmod.SubjectDirectory(get_config()))
    out = plan({"pathway": "student", "user_id": USER, "subject_id": SUBJECT,
                "message": "explain fractions", "understanding": {"needs_syllabus": True},
                "planned_tasks": [{"type": "explain", "query": "fractions",
                                   "level": "intermediate", "length": "medium"}],
                "conversation": {}})
    assert "Fractions" in out["learner"]


def test_the_standing_is_carried_to_every_task_of_the_turn():
    sends = nmod.dispatch({"tasks": [{"type": "explain", "params": {}}],
                           "learner": "Answered incorrectly before: Fractions."})
    assert sends[0].arg["learner"] == "Answered incorrectly before: Fractions."


def test_the_compiled_graph_hands_the_standing_to_the_explain_prompt(monkeypatch):
    """plan returned the standing under a key MainState did not declare, so LangGraph
    dropped it, and the explain adapter never forwarded it either. Every explanation
    read "nothing recorded" while the node-level tests passed."""
    from types import SimpleNamespace

    from agent.config import get_config
    from agent.graph import build_graph
    from agent.graph import graph as gmod
    from agent.graph import nodes as nmod
    from agent import retrieval as rmod
    from agent.models import PrereqTopic, ResolvedTopic, RetrievedChunk

    prompts: list[str] = []

    class _RecordingLLM:
        def invoke(self, messages):
            prompts.append(messages[-1].content)
            return SimpleNamespace(content="Plants turn light into energy [1].")

    monkeypatch.setattr(gmod, "maybe_llm",
                        lambda config, task: _RecordingLLM() if task == "explain" else None)
    monkeypatch.setattr(nmod, "resolve_topics",
                        lambda q, sid=None, **k: [ResolvedTopic("t1", "Photosynthesis", 0.9)])
    monkeypatch.setattr(nmod, "standing",
                        lambda *a, **k: LearnerStanding(weak=["Fractions"]))
    chunk = RetrievedChunk(
        chunk_id="c1", score=0.7, text="Plants convert light into energy.",
        lecture_id="lec1", topic_ids=["t1"], page_from=3, page_to=3,
        heading_path="Ch1 > Photosynthesis", language="en")
    monkeypatch.setattr(rmod, "rag_search", lambda *a, **k: [chunk])
    monkeypatch.setattr(rmod, "get_topic_deps",
                        lambda ids, **k: [PrereqTopic("t0", "Cells", "required", 1)])

    build_graph(get_config()).invoke({"message": "explain photosynthesis",
                                      "user_type": "student", "user_id": "u1",
                                      "subject_id": "sub1"})

    assert prompts, "the explain model was never called"
    assert "Answered incorrectly before: Fractions." in prompts[0]
    assert NOTHING_RECORDED not in prompts[0]
