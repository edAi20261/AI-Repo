from __future__ import annotations

import pytest

from scripts.merge_duplicate_topics import (
    MERGE_STATEMENTS,
    group_topics,
    mapping,
    merge_subject,
    update_index,
)


def _topic(tid, vec, lecture="lec-1"):
    return {"topic_id": tid, "label": tid, "lecture_id": lecture, "embedding": vec}


def test_a_skill_minted_in_three_lessons_becomes_the_oldest_of_them():
    """Each lesson minted its own "Subtraction", and a student's mastery of it was
    split between three topics."""
    topics = [_topic("sub-1", [1.0, 0.0], "lec-1"), _topic("add", [0.0, 1.0], "lec-1"),
              _topic("sub-2", [0.99, 0.05], "lec-2"), _topic("sub-3", [0.98, 0.1], "lec-3")]
    groups = group_topics(topics, 0.9)

    assert len(groups) == 1
    assert groups[0]["canonical"]["topic_id"] == "sub-1"
    assert [m["topic_id"] for m in groups[0]["merged"]] == ["sub-2", "sub-3"]
    assert mapping(groups) == {"old": ["sub-2", "sub-3"], "new": ["sub-1", "sub-1"]}


def test_a_chain_of_near_neighbours_does_not_merge_unrelated_skills():
    """Similarity is taken to the kept topic: A~B and B~C must not put C under A
    when C is not itself close to A."""
    topics = [_topic("a", [1.0, 0.0]), _topic("b", [0.8, 0.6]), _topic("c", [0.28, 0.96])]
    groups = group_topics(topics, 0.75)

    assert [(g["canonical"]["topic_id"], [m["topic_id"] for m in g["merged"]])
            for g in groups] == [("a", ["b"])]


def test_distinct_skills_are_left_alone():
    assert group_topics([_topic("a", [1.0, 0.0]), _topic("b", [0.0, 1.0])], 0.6) == []
    assert group_topics([], 0.6) == []


class _Cursor:
    def __init__(self, present):
        self.present = present
        self.executed = []
        self.rowcount = 1
        self._rows = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        if "SELECT topic_id::text FROM ai.topics" in sql:
            self._rows = [(t,) for t in self.present]
        elif sql.startswith("UPDATE ai.chunks"):
            self._rows = [("c1", ["sub-1", "add"])]
        else:
            self._rows = []

    def fetchall(self):
        return self._rows


_GROUPS = [{"canonical": {"topic_id": "sub-1"}, "merged": [{"topic_id": "sub-2"}]}]


def test_every_reference_moves_before_the_merged_topics_are_deleted():
    cur = _Cursor(present=["sub-1"])
    counts, passages = merge_subject(cur, _GROUPS, min_attempts=2)

    statements = [sql for sql, _ in cur.executed[1:]]
    assert statements == [sql for _, sql in MERGE_STATEMENTS]
    assert statements[-1] == "DELETE FROM ai.topics WHERE topic_id = ANY(%(old)s::uuid[])"
    for table in ("ai.chunks", "ai.figures", "ai.vector_index", "quizzes.quiz_questions",
                  "homework.homework_questions", "ai.mastery", "ai.mastery_events",
                  "ai.topic_dependencies"):
        assert any(table in sql for sql in statements), table
    assert all(params == {"old": ["sub-2"], "new": ["sub-1"], "min_attempts": 2}
               for _, params in cur.executed)
    assert passages == [("c1", ["sub-1", "add"])]
    assert counts["ai.topics"] == 1


def test_merged_mastery_sums_evidence_and_weights_the_score_by_it():
    sql = dict(MERGE_STATEMENTS)["ai.mastery"]
    assert "sum(x.attempts)" in sql and "sum(x.score * x.attempts) / sum(x.attempts)" in sql
    assert "max(x.last_seen)" in sql and "ON CONFLICT (user_id, topic_id) DO UPDATE" in sql


def test_a_merged_edge_that_would_point_at_itself_is_dropped():
    sql = dict(MERGE_STATEMENTS)["ai.topic_dependencies"]
    assert "e.topic_id <> e.prereq_topic_id" in sql and "DO NOTHING" in sql


def test_a_plan_whose_kept_topic_is_gone_changes_nothing():
    cur = _Cursor(present=[])
    with pytest.raises(ValueError, match="no longer exist"):
        merge_subject(cur, _GROUPS, min_attempts=2)
    assert len(cur.executed) == 1


def test_the_index_follows_the_database():
    class _Client:
        def __init__(self):
            self.payloads, self.deleted = [], []

        def set_payload(self, collection, payload, points, wait):
            self.payloads.append((collection, payload, list(points)))

        def delete(self, collection_name, points_selector, wait):
            self.deleted.append((collection_name, list(points_selector.points)))

    class _Config:
        vector_store = {"collection": "chunks", "topics_collection": "topics"}

    client = _Client()
    update_index(client, _Config(), [("c1", ["t1"]), ("c2", ["t1"]), ("c3", ["t1", "t2"])],
                 ["11111111-1111-1111-1111-111111111111"])

    assert client.payloads == [("chunks", {"topic_ids": ["t1"]}, ["c1", "c2"]),
                               ("chunks", {"topic_ids": ["t1", "t2"]}, ["c3"])]
    assert client.deleted == [("topics", ["11111111-1111-1111-1111-111111111111"])]
