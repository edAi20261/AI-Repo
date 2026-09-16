"""Backfill ai.lecture_ingestion for lectures already ingested, without re-running
the pipeline. Status and counts are derived from the current ai.* rows: a lecture
with untagged or stub-text chunks is marked needs_review, else ready.

    python scripts/backfill_ingestion_records.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.config import Config
from ingestion.llm_provider import VISION_STUB_SENTINEL
from shared.postgres import connect

_UPSERT = """
INSERT INTO ai.lecture_ingestion
    (lecture_id, subject_id, version, status, chunk_count, figure_count,
     topic_count, needs_review_reason, ingested_at)
VALUES (%(lecture_id)s, %(subject_id)s, %(version)s, %(status)s, %(chunk_count)s,
        %(figure_count)s, %(topic_count)s, %(needs_review_reason)s, CURRENT_TIMESTAMP)
ON CONFLICT (lecture_id) DO UPDATE SET
    subject_id = EXCLUDED.subject_id, version = EXCLUDED.version,
    status = EXCLUDED.status, chunk_count = EXCLUDED.chunk_count,
    figure_count = EXCLUDED.figure_count, topic_count = EXCLUDED.topic_count,
    needs_review_reason = EXCLUDED.needs_review_reason, ingested_at = CURRENT_TIMESTAMP
"""


def main() -> int:
    cfg = Config.load()
    with connect(cfg, connect_timeout=15, statement_timeout_ms=0) as conn:
        return _backfill(conn)


def _backfill(conn) -> int:
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT lecture_id::text FROM ai.chunks ORDER BY 1")
    lectures = [r[0] for r in cur.fetchall()]

    for lid in lectures:
        cur.execute(
            "SELECT count(*), "
            "count(*) FILTER (WHERE topic_ids IS NULL OR cardinality(topic_ids) = 0), "
            "count(*) FILTER (WHERE text LIKE %s) "
            "FROM ai.chunks WHERE lecture_id = %s",
            (f"%{VISION_STUB_SENTINEL}%", lid),
        )
        n_chunks, untagged, stubbed = cur.fetchone()
        cur.execute("SELECT count(*) FROM ai.figures WHERE lecture_id = %s", (lid,))
        n_fig = cur.fetchone()[0]
        cur.execute("SELECT count(DISTINCT t) FROM "
                    "(SELECT unnest(topic_ids) t FROM ai.chunks WHERE lecture_id = %s) s", (lid,))
        n_topics = cur.fetchone()[0]
        cur.execute("SELECT subject_id::text FROM ai.vector_index WHERE lecture_id = %s LIMIT 1", (lid,))
        row = cur.fetchone()
        subject_id = row[0] if row else None
        cur.execute("SELECT coalesce(max(version), 1) FROM ai.vector_index WHERE lecture_id = %s", (lid,))
        version = cur.fetchone()[0]

        problems = []
        if untagged:
            problems.append(f"{untagged} chunk(s) untagged")
        if stubbed:
            problems.append(f"{stubbed} chunk(s) stub text")
        status = "needs_review" if problems else "ready"

        cur.execute(_UPSERT, {
            "lecture_id": lid, "subject_id": subject_id, "version": version,
            "status": status, "chunk_count": n_chunks, "figure_count": n_fig,
            "topic_count": n_topics, "needs_review_reason": "; ".join(problems) or None,
        })
        print(f"  {status:12s} {lid}  chunks={n_chunks} figures={n_fig} topics={n_topics} v{version}")

    conn.commit()
    print(f"\nbackfilled {len(lectures)} lecture(s) into ai.lecture_ingestion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
