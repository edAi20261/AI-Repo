"""An archived lesson is gone for a student, whatever the index still holds.

Archiving is done by the platform, which tells this service nothing: 162
passages of 19 archived lessons (a retired copy of a book, old demo lessons)
and 1,457 of 25 archived external lessons were still being served."""

import pytest

from agent.models import RetrievedChunk

LESSON = "6f1b7d3e-0000-4000-8000-000000000001"
CHUNK = "6f1b7d3e-0000-4000-8000-0000000000c1"
TOPIC = "6f1b7d3e-0000-4000-8000-0000000000f1"
ROW = (CHUNK, "a passage", None, 1, 1, LESSON, "en", [], 0)


def _reads(tools, config, ready_only):
    anchor = RetrievedChunk(chunk_id=CHUNK, score=0.0, text="", heading_path=None,
                            page_from=1, page_to=1, lecture_id=LESSON, language="en",
                            topic_ids=[], order_index=0)
    return {
        "rag_search": lambda: tools.rag_search(
            "photosynthesis", ["syllabus"], ready_only=ready_only, config=config),
        "get_chunks_by_id": lambda: tools.get_chunks_by_id(
            [CHUNK], ready_only=ready_only, config=config),
        "get_neighbours": lambda: tools.get_neighbours(
            [anchor], window=1, ready_only=ready_only, config=config),
        "get_lesson_chunks": lambda: tools.get_lesson_chunks(
            LESSON, ["syllabus"], limit=4, ready_only=ready_only, config=config),
        "get_topic_chunks": lambda: tools.get_topic_chunks(
            [TOPIC], ["syllabus"], ready_only=ready_only, config=config),
    }


@pytest.mark.parametrize("read", ["rag_search", "get_chunks_by_id", "get_neighbours",
                                  "get_lesson_chunks", "get_topic_chunks"])
@pytest.mark.parametrize("ready_only", [True, False])
def test_an_archived_lessons_passages_are_never_served(read, ready_only, fake_postgres, monkeypatch):
    from agent import tools
    from agent.config import get_config

    monkeypatch.setattr(tools, "_candidate_scores", lambda *a, **k: ({CHUNK: 0.9}, {CHUNK: {}}))
    monkeypatch.setattr(tools, "_citation_labels", lambda *a, **k: {})
    monkeypatch.setattr(tools, "get_embedder", lambda config: _Embedder())
    db = fake_postgres(fetchall=[[ROW]] * 4)

    _reads(tools, get_config(), ready_only)[read]()

    passages = [sql for sql in db.sql if "FROM ai.chunks c" in sql]
    assert passages and all("status = 'archived'" in sql for sql in passages), db.sql


class _Embedder:
    def embed_query(self, text):
        return [0.0] * 4
