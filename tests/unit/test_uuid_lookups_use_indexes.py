"""A lookup by id reads the index on that id.

``c.chunk_id::text = ANY(%s)`` compares a value Postgres has to compute for every
row, so neither ``chunks_pkey`` nor ``idx_chunks_lecture`` can answer it. Every
chunk lookup on a tutor turn parallel-scanned all ~148k chunks: 835 ms for three
chunks by id, 729 ms for one lesson's passages, against 13 ms and 8 ms once the
comparison is uuid to uuid. Idle, that is slow; under concurrent turns it reached
the 5000 ms statement timeout, and eleven tutor turns failed on 10-25.
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path

from agent.models import RetrievedChunk

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("agent", "ingestion", "system", "shared", "scripts")

CAST_COMPARISON = re.compile(
    r"::text\s*(=|<>|!=|\bIN\b)|(=|<>|!=)\s*[\w.]+::text\b", re.IGNORECASE)

LESSON = "11111111-1111-1111-1111-111111111111"
CHUNK = "22222222-2222-2222-2222-222222222222"


def _string_literals(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value


def _cast_comparisons(sql: str) -> list[str]:
    return [m.group(0) for m in CAST_COMPARISON.finditer(sql)]


def test_no_query_in_the_repo_compares_an_id_cast_to_text():
    """The cast that cost every tutor turn a full scan of ai.chunks was in three
    queries and had been copied into twenty more; a guard over one function would
    have left the other nineteen."""
    found = [
        f"{path.relative_to(ROOT)}:{line}: {hit}"
        for package in PACKAGES
        for path in sorted((ROOT / package).rglob("*.py"))
        for line, text in _string_literals(path)
        for hit in _cast_comparisons(text)
    ]
    assert found == [], "compare uuid to uuid so the index applies:\n" + "\n".join(found)


def _config():
    from agent.config import get_config

    return get_config()


def _no_labels(monkeypatch):
    from agent import tools

    monkeypatch.setattr(tools, "_citation_labels", lambda ids, config: {})


def _row(chunk_id: str = CHUNK, order: int = 3):
    return (chunk_id, "passage", None, 1, 1, LESSON, "en", [], order)


def test_the_tutors_chunk_lookups_send_uuids_the_index_can_match(fake_postgres, monkeypatch):
    """The three lookups on every tutor turn: by id, by neighbour, by lesson."""
    from agent import tools

    _no_labels(monkeypatch)
    db = fake_postgres(rows=[_row()])
    anchor = RetrievedChunk(chunk_id=CHUNK, score=0.9, text="passage", topic_ids=[],
                            page_from=1, page_to=1, heading_path=None, language="en",
                            lecture_id=LESSON, order_index=3)

    tools.get_chunks_by_id([CHUNK], config=_config())
    tools.get_neighbours([anchor], window=2, config=_config())
    tools.get_lesson_chunks(LESSON, ["syllabus"], limit=5, config=_config())

    assert len(db.sql) == 3
    assert [hit for sql in db.sql for hit in _cast_comparisons(sql)] == []
    assert "c.chunk_id = ANY(%s::uuid[])" in db.sql[0]
    assert "c.lecture_id = %s::uuid" in db.sql[1]
    assert "c.lecture_id = %s::uuid" in db.sql[2]


def test_an_id_that_is_not_a_uuid_is_left_out_of_a_lookup_and_logged(
        fake_postgres, monkeypatch, caplog):
    """With the comparison on uuids, one malformed id would make Postgres reject
    the whole statement and fail the turn; matched as text, it found nothing and
    nobody knew. It is dropped before the query, and the drop is logged."""
    from agent import tools

    _no_labels(monkeypatch)
    db = fake_postgres(rows=[_row()])

    with caplog.at_level(logging.WARNING):
        chunks = tools.get_chunks_by_id(["not-a-uuid", CHUNK], config=_config())

    assert [c.chunk_id for c in chunks] == [CHUNK]
    assert db.statements[0][1] == ([CHUNK],)
    assert any("not uuids" in r.getMessage() for r in caplog.records)


def test_a_lesson_id_that_is_not_a_uuid_reads_nothing_and_says_so(
        fake_postgres, monkeypatch, caplog):
    from agent import tools

    _no_labels(monkeypatch)
    db = fake_postgres(rows=[_row()])

    with caplog.at_level(logging.WARNING):
        chunks = tools.get_lesson_chunks("lesson-7", ["syllabus"], limit=5, config=_config())

    assert chunks == [] and db.sql == []
    assert any("not uuids" in r.getMessage() for r in caplog.records)
