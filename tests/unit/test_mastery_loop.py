"""2B — the mastery loop, end to end: grade_sg writes mastery → assessment_sg reads it
→ the next assessment weights the weak topic first. Both subgraphs run against a shared
in-memory mastery store (no DB, no LLM)."""

from __future__ import annotations

from agent.config import get_config
from agent.graph.subgraphs import assessment_sg as qmod
from agent.graph.subgraphs import grade_sg as gmod
from agent.graph.subgraphs import build_assessment_sg, build_grade_sg
from agent.mastery import Mastery, MasteryParams, status
from agent.models import RetrievedChunk


def _chunk(topic):
    return RetrievedChunk(
        chunk_id=f"c-{topic}", score=0.5, text="Grounding text for the item.",
        lecture_id="lec1", topic_ids=[topic], page_from=1, page_to=1,
        heading_path="Ch1", language="en")


def _row(tid, m, params):
    return {"topic_id": tid, "score": m.score, "attempts": m.attempts,
            "trend": m.trend, "confidence": m.confidence, "status": status(m, params)}


def test_mastery_loop_grade_feeds_next_assessment(monkeypatch):
    cfg = get_config()
    params = MasteryParams.from_config(cfg)
    # tStrong is already mastered; tWeak has no history yet.
    store: dict[str, Mastery] = {"tStrong": Mastery(score=0.9, attempts=5)}

    def read(uid, ids=None, **_):
        ids = ids or list(store)
        return [_row(t, store[t], params) for t in ids if t in store]

    # --- grade side: the student answers tWeak WRONG; mastery gets written ---
    monkeypatch.setattr(gmod, "get_mastery", lambda uid, ids, config=None: read(uid, ids))
    monkeypatch.setattr(gmod, "update_mastery",
                        lambda uid, tid, m, config=None: store.__setitem__(tid, m))
    monkeypatch.setattr(gmod, "write_mastery_events", lambda events, config=None: ["e1"])

    grade = build_grade_sg(config=cfg, llm=None)
    grade.invoke({
        "submissions": [{"question": "q", "student_answer": "wrong", "correct_answer": "right",
                         "topic_id": "tWeak", "lecture_id": "lec1", "difficulty": "easy"}],
        "user_id": "u1", "source": "quiz",
    })
    assert "tWeak" in store                       # grade_sg wrote mastery for the weak topic

    # --- assessment side: the next quiz reads that mastery and adapts ---
    monkeypatch.setattr(qmod, "get_mastery", read)
    monkeypatch.setattr(qmod.retrieval, "retrieve",
                        lambda q, ns, **k: [_chunk((k.get("topic_ids") or ["x"])[0])])
    assessment = build_assessment_sg(config=cfg, llm=None)
    out = assessment.invoke({
        "query": "review", "subject_id": "sub1",
        "topic_ids": ["tStrong", "tWeak"], "namespaces": ["syllabus"], "language": "en",
        "item_source": "generated", "type_mix": {"mcq": 2}, "k_per_topic": 1,
        "adaptive": True, "user_id": "u1",
    })
    # The weak/unknown topic gets the first item — the loop closed.
    assert out["assessment"]["items"][0]["topic_id"] == "tWeak"
