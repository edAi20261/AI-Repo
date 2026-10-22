"""Merge the topics that name one skill twice within a subject.

Every lesson minted its own topics, so a skill taught in three lessons was three
topics, and a student's mastery of it was split three ways. Topics whose labels
are as close as ingestion's merge threshold are one skill: the oldest is kept,
and every passage, figure, index row, question, prerequisite edge and mastery
record that named another is moved onto it before the others are deleted.

    python -m scripts.merge_duplicate_topics [--subject <id> ...] --plan plan.json           # dry run
    python -m scripts.merge_duplicate_topics --plan plan.json --apply

The dry run writes the groups to ``plan.json``; ``--apply`` merges exactly
those groups, so what is applied is what was reviewed. Each subject is merged
in one transaction, and the index is updated after it commits.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from shared.config import get_config
from shared.namespaces import NOT_BACKGROUND_TOPIC
from shared.postgres import connect_once

_SUBJECTS = "SELECT DISTINCT subject_id::text FROM ai.topics ORDER BY 1"
_TOPICS = (
    "SELECT topic_id::text, label, lecture_id::text, created_at, label_embedding "
    "FROM ai.topics WHERE subject_id = %s AND label_embedding IS NOT NULL AND "
    + NOT_BACKGROUND_TOPIC + " ORDER BY created_at, topic_id"
)

_MAP = "unnest(%(old)s::uuid[], %(new)s::uuid[]) AS m(old, new)"

_TOPIC_ARRAY_TABLES = ("ai.chunks", "ai.figures", "ai.vector_index",
                       "quizzes.quiz_questions", "homework.homework_questions")


def _remap_array(table: str) -> str:
    returning = " RETURNING chunk_id::text, topic_ids::text[]" if table == "ai.chunks" else ""
    return (
        f"UPDATE {table} SET topic_ids = (SELECT array_agg(s.topic_id ORDER BY s.first) FROM ("
        "SELECT coalesce(m.new, a.topic_id) AS topic_id, min(a.ord) AS first "
        "FROM unnest(topic_ids) WITH ORDINALITY AS a(topic_id, ord) "
        f"LEFT JOIN {_MAP} ON m.old = a.topic_id GROUP BY 1) s) "
        "WHERE topic_ids && %(old)s::uuid[]" + returning
    )


_MERGE_MASTERY = (
    "INSERT INTO ai.mastery (user_id, topic_id, score, attempts, last_seen, trend, confidence) "
    "SELECT x.user_id, x.canonical, "
    "CASE WHEN sum(x.attempts) > 0 THEN sum(x.score * x.attempts) / sum(x.attempts) "
    "ELSE max(x.score) END, "
    "sum(x.attempts), max(x.last_seen), "
    "(array_agg(x.trend ORDER BY x.last_seen DESC NULLS LAST))[1], "
    "least(1.0, sum(x.attempts)::float8 / greatest(1, %(min_attempts)s)) "
    "FROM (SELECT ms.*, coalesce(m.new, ms.topic_id) AS canonical FROM ai.mastery ms "
    f"LEFT JOIN {_MAP} ON m.old = ms.topic_id "
    "WHERE (ms.topic_id = ANY(%(old)s::uuid[]) OR ms.topic_id = ANY(%(new)s::uuid[])) "
    "AND ms.user_id IN (SELECT user_id FROM ai.mastery WHERE topic_id = ANY(%(old)s::uuid[]))"
    ") x GROUP BY x.user_id, x.canonical "
    "ON CONFLICT (user_id, topic_id) DO UPDATE SET score = EXCLUDED.score, "
    "attempts = EXCLUDED.attempts, last_seen = EXCLUDED.last_seen, trend = EXCLUDED.trend, "
    "confidence = EXCLUDED.confidence"
)
_MERGE_DEPENDENCIES = (
    "INSERT INTO ai.topic_dependencies (topic_id, prereq_topic_id, strength, rationale) "
    "SELECT DISTINCT ON (e.topic_id, e.prereq_topic_id) e.topic_id, e.prereq_topic_id, "
    "e.strength, e.rationale FROM ("
    "SELECT coalesce(a.new, d.topic_id) AS topic_id, "
    "coalesce(b.new, d.prereq_topic_id) AS prereq_topic_id, d.strength, d.rationale "
    "FROM ai.topic_dependencies d "
    "LEFT JOIN unnest(%(old)s::uuid[], %(new)s::uuid[]) AS a(old, new) ON a.old = d.topic_id "
    "LEFT JOIN unnest(%(old)s::uuid[], %(new)s::uuid[]) AS b(old, new) "
    "ON b.old = d.prereq_topic_id "
    "WHERE d.topic_id = ANY(%(old)s::uuid[]) OR d.prereq_topic_id = ANY(%(old)s::uuid[])"
    ") e WHERE e.topic_id <> e.prereq_topic_id "
    "ORDER BY e.topic_id, e.prereq_topic_id, e.strength "
    "ON CONFLICT (topic_id, prereq_topic_id) DO NOTHING"
)

MERGE_STATEMENTS: tuple[tuple[str, str], ...] = (
    *((table, _remap_array(table)) for table in _TOPIC_ARRAY_TABLES),
    ("ai.mastery", _MERGE_MASTERY),
    ("ai.mastery (merged rows)", "DELETE FROM ai.mastery WHERE topic_id = ANY(%(old)s::uuid[])"),
    ("ai.mastery_events", "UPDATE ai.mastery_events e SET topic_id = m.new FROM "
                          f"{_MAP} WHERE e.topic_id = m.old"),
    ("ai.topic_dependencies", _MERGE_DEPENDENCIES),
    ("ai.topic_dependencies (merged edges)",
     "DELETE FROM ai.topic_dependencies WHERE topic_id = ANY(%(old)s::uuid[]) "
     "OR prereq_topic_id = ANY(%(old)s::uuid[])"),
    ("ai.topics", "DELETE FROM ai.topics WHERE topic_id = ANY(%(old)s::uuid[])"),
)

_KEPT_PRESENT = "SELECT topic_id::text FROM ai.topics WHERE topic_id = ANY(%(new)s::uuid[])"

_SET_PAYLOAD_BATCH = 500


def group_topics(topics: Sequence[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    """Topics in creation order, each folded into the most similar older topic
    kept so far when their labels are within ``threshold``. Similarity is taken
    to a kept topic, never through a merged one, so a chain of near neighbours
    does not drift into one group of unrelated skills."""
    import numpy as np

    if not topics:
        return []
    vectors = np.array([t["embedding"] for t in topics], dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    similarity = vectors @ vectors.T
    kept: list[int] = []
    merged: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for i in range(len(topics)):
        if kept:
            scores = similarity[i, kept]
            best = int(np.argmax(scores))
            if scores[best] >= threshold:
                merged[kept[best]].append((i, float(scores[best])))
                continue
        kept.append(i)
    return [{"canonical": _entry(topics[k]),
             "merged": [{**_entry(topics[i]), "similarity": round(score, 4)}
                        for i, score in merged[k]]}
            for k in kept if merged[k]]


def _entry(topic: dict[str, Any]) -> dict[str, Any]:
    return {"topic_id": topic["topic_id"], "label": topic["label"],
            "lecture_id": topic["lecture_id"]}


def mapping(groups: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    old = [m["topic_id"] for g in groups for m in g["merged"]]
    new = [g["canonical"]["topic_id"] for g in groups for _ in g["merged"]]
    return {"old": old, "new": new}


def merge_subject(cur, groups: Sequence[dict[str, Any]], min_attempts: int,
                  ) -> tuple[dict[str, int], list[tuple[str, list[str]]]]:
    """Every reference moved onto its group's kept topic, then the rest deleted.
    Returns rows touched per table and the passages whose topics changed."""
    params = {**mapping(groups), "min_attempts": min_attempts}
    cur.execute(_KEPT_PRESENT, params)
    missing = set(params["new"]) - {row[0] for row in cur.fetchall()}
    if missing:
        raise ValueError(f"the plan keeps topics that no longer exist: {sorted(missing)}; "
                         "run the dry run again")
    counts: dict[str, int] = {}
    passages: list[tuple[str, list[str]]] = []
    for table, statement in MERGE_STATEMENTS:
        cur.execute(statement, params)
        counts[table] = cur.rowcount
        if table == "ai.chunks":
            passages = [(chunk_id, list(topic_ids or [])) for chunk_id, topic_ids in cur.fetchall()]
    return counts, passages


def update_index(client, config, passages: Sequence[tuple[str, list[str]]],
                 merged_ids: Sequence[str]) -> None:
    from qdrant_client import models

    by_topics: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for chunk_id, topic_ids in passages:
        by_topics[tuple(topic_ids)].append(chunk_id)
    collection = config.vector_store.get("collection")
    for topic_ids, chunk_ids in by_topics.items():
        for start in range(0, len(chunk_ids), _SET_PAYLOAD_BATCH):
            client.set_payload(collection, payload={"topic_ids": list(topic_ids)},
                               points=chunk_ids[start:start + _SET_PAYLOAD_BATCH], wait=True)
    if merged_ids:
        client.delete(collection_name=config.vector_store.get("topics_collection"),
                      points_selector=models.PointIdsList(points=list(merged_ids)), wait=True)


def _read_topics(cur, subject_id: str) -> list[dict[str, Any]]:
    cur.execute(_TOPICS, (subject_id,))
    return [{"topic_id": tid, "label": label, "lecture_id": lecture_id,
             "created_at": created_at.isoformat() if created_at else None, "embedding": emb}
            for tid, label, lecture_id, created_at, emb in cur.fetchall() if emb]


def _plan(config, subjects: Sequence[str], threshold: float) -> dict[str, Any]:
    with connect_once(config) as conn, conn.cursor() as cur:
        if not subjects:
            cur.execute(_SUBJECTS)
            subjects = [row[0] for row in cur.fetchall()]
        planned = [{"subject_id": subject_id,
                    "groups": group_topics(_read_topics(cur, subject_id), threshold)}
                   for subject_id in subjects]
    return {"threshold": threshold, "subjects": planned}


def _apply(config, plan: dict[str, Any]) -> None:
    from shared.vector_store import upsert_client

    client = upsert_client(config, config.secret("QDRANT_URL"))
    min_attempts = config.get("mastery.min_attempts")
    for subject in plan["subjects"]:
        groups = subject["groups"]
        if not groups:
            continue
        with connect_once(config) as conn, conn.cursor() as cur:
            counts, passages = merge_subject(cur, groups, min_attempts)
            conn.commit()
        update_index(client, config, passages, mapping(groups)["old"])
        print(f"{subject['subject_id']}: " + ", ".join(f"{t}={n}" for t, n in counts.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subject", action="append", default=[], dest="subjects",
                        help="a subject to merge; every subject when none is given")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--threshold", type=float,
                        help="label similarity to merge at, for this dry run; "
                             "ingestion.topics.merge_threshold when not given")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    config = get_config()

    if not args.apply:
        threshold = (args.threshold if args.threshold is not None
                     else config.get("ingestion.topics.merge_threshold"))
        plan = _plan(config, args.subjects, threshold)
        args.plan.write_text(json.dumps(plan, ensure_ascii=False, indent=1))
        merged = 0
        for subject in plan["subjects"]:
            for group in subject["groups"]:
                merged += len(group["merged"])
                print(f"{subject['subject_id']}  {group['canonical']['label']}  <-  "
                      + "; ".join(f"{m['label']} ({m['similarity']})" for m in group["merged"]))
        print(f"{merged} topics would merge into "
              f"{sum(len(s['groups']) for s in plan['subjects'])}; plan written to {args.plan}")
        print("dry run: nothing changed; pass --apply to merge them")
        return 0

    plan = json.loads(args.plan.read_text())
    if args.subjects:
        plan["subjects"] = [s for s in plan["subjects"] if s["subject_id"] in args.subjects]
    _apply(config, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
