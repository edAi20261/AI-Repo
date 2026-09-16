"""Write every topic label in the language of the lesson it belongs to.

The extraction prompt did not name a language, so it defaulted to English, and
46 of 79 topics on Arabic lessons were written in English. A student asking a
question in Arabic got an Arabic answer that ended in English prerequisites.

The producer is fixed — the prompt states the language and ``_labels_in``
validates it at creation — but the rows that were already written stay wrong
until something rewrites them. This is that something.

It relabels rather than re-extracts. A topic_id is what a student's mastery
history hangs from, and re-extraction mints new ones; worse, topic reuse matches
an old label against a new one by cosine at 0.60, and a correct Arabic
translation of an English label measures 0.54-0.64 against it — so a re-ingest
would leave the English row live beside a new Arabic one rather than replacing
it.

    python scripts/relabel_topics.py                 # show what would change
    python scripts/relabel_topics.py --apply         # write it
    python scripts/relabel_topics.py --apply --lecture-id <uuid>
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psycopg.types.json import Json  # noqa: E402

from ingestion.prompts import TOPIC_RELABEL_PROMPT  # noqa: E402
from ingestion.schemas import RelabelList  # noqa: E402
from shared.config import Config  # noqa: E402
from shared.embeddings import get_embedder  # noqa: E402
from shared.language import written_in  # noqa: E402
from shared.llm import get_llm  # noqa: E402
from shared.postgres import connect  # noqa: E402
from shared.structured import call_structured, schema_hint  # noqa: E402
from shared.vector_store import upsert_client  # noqa: E402

_LESSONS = """
SELECT t.topic_id::text, t.label, t.description, t.lecture_id::text,
       t.subject_id::text,
       (SELECT mode() WITHIN GROUP (ORDER BY c.language)
          FROM ai.chunks c WHERE c.lecture_id = t.lecture_id) AS language
FROM ai.topics t
WHERE t.label IS NOT NULL
"""


def _mislabelled(config: Config, lecture_id: str | None) -> dict[str, list[dict]]:
    """``{lecture_id: [topic, ...]}`` for topics not in their lesson's language."""
    sql = _LESSONS + ("AND t.lecture_id = %s::uuid" if lecture_id else "")
    with connect(config, connect_timeout=15, statement_timeout_ms=0) as conn, \
            conn.cursor() as cur:
        cur.execute(sql, (lecture_id,) if lecture_id else None)
        rows = cur.fetchall()

    lessons: dict[str, list[dict]] = {}
    for topic_id, label, description, lecture, subject, language in rows:
        if not language or written_in(label, language):
            continue
        lessons.setdefault(lecture, []).append({
            "topic_id": topic_id, "label": label, "description": description,
            "subject_id": subject, "language": language,
        })
    return lessons


def _same_topics(batch: RelabelList, wanted: set[str], language: str) -> None:
    """A relabel that loses, invents or renames an id is not a relabel.

    The whole point of this script over a re-extraction is that topic_ids
    survive, so an id that changes here costs exactly what re-extracting would
    have cost — a student's mastery history, detached from the skill it was
    earned on.
    """
    returned = [t.topic_id for t in batch.topics]
    if sorted(returned) != sorted(wanted):
        missing = sorted(wanted - set(returned))
        unknown = sorted(set(returned) - wanted)
        raise ValueError(
            f"return every id exactly once. Missing: {missing}. Not asked for: "
            f"{unknown}. Duplicated: {sorted({i for i in returned if returned.count(i) > 1})}")
    wrong = [t.label for t in batch.topics if not written_in(t.label, language)]
    if wrong:
        raise ValueError(
            f"every label must be written in {language!r}, the language of this "
            f"lesson. These are not: {wrong}")


def _relabel(config: Config, topics: list[dict], language: str) -> list[dict]:
    listing = "\n".join(
        f'- id {t["topic_id"]}: {t["label"]}'
        + (f' — {t["description"]}' if t["description"] else "")
        for t in topics)
    prompt = TOPIC_RELABEL_PROMPT.render(
        language=language, schema=schema_hint(RelabelList), topics=listing)
    wanted = {t["topic_id"] for t in topics}
    result = call_structured(
        RelabelList, prompt, task="extract_topics", config=config,
        llm=get_llm(config, "extract_topics"), label="relabel_topics",
        validate=lambda batch: _same_topics(batch, wanted, language),
    )
    return [t.model_dump() for t in result.topics]


def _write(config: Config, relabelled: list[dict], rows: dict[str, dict]) -> None:
    """Update the labels, then the vectors that are derived from them."""
    embedder = get_embedder(config)
    vectors = embedder.embed([t["label"] for t in relabelled])

    with connect(config, connect_timeout=15, statement_timeout_ms=0) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE ai.topics SET label = %s, description = %s, "
                "label_embedding = %s WHERE topic_id = %s::uuid",
                [(t["label"], t["description"], Json(vector), t["topic_id"])
                 for t, vector in zip(relabelled, vectors)])
        conn.commit()

    # A label vector that still says "Identify main idea" would keep matching
    # English queries to a lesson whose label now reads تحديد الفكرة الرئيسية.
    from qdrant_client import models

    collection = config.vector_store.get("topics_collection", "eduai_topics")
    client = upsert_client(config, config.secret("QDRANT_URL"))
    client.upsert(collection, wait=True, points=[
        models.PointStruct(
            id=t["topic_id"], vector=vector,
            payload={k: v for k, v in {
                "topic_id": t["topic_id"], "label": t["label"],
                "subject_id": rows[t["topic_id"]]["subject_id"],
                "lecture_id": rows[t["topic_id"]]["lecture_id"],
            }.items() if v is not None},
        )
        for t, vector in zip(relabelled, vectors)
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="write the new labels; without it, only show them")
    parser.add_argument("--lecture-id", default=None)
    args = parser.parse_args()

    config = Config.load()
    lessons = _mislabelled(config, args.lecture_id)
    if not lessons:
        print("every topic label is already in its lesson's language.")
        return 0

    total = sum(len(t) for t in lessons.values())
    print(f"{total} label(s) across {len(lessons)} lesson(s) "
          f"not in the lesson's language\n")

    changed: list[dict] = []
    rows: dict[str, dict] = {}
    for lecture_id, topics in lessons.items():
        language = topics[0]["language"]
        for topic in topics:
            topic["lecture_id"] = lecture_id
            rows[topic["topic_id"]] = topic
        print(f"lecture {lecture_id}  ({language}, {len(topics)} topics)")
        try:
            relabelled = _relabel(config, topics, language)
        except Exception as error:      # noqa: BLE001 — one lesson must not stop the rest
            print(f"  FAILED: {type(error).__name__}: {str(error)[:160]}")
            continue
        for new in relabelled:
            print(f"  {rows[new['topic_id']]['label']}\n    -> {new['label']}")
        changed.extend(relabelled)
        print()

    if not args.apply:
        print(f"{len(changed)} label(s) would be rewritten. "
              "Re-run with --apply to write them.")
        return 0

    _write(config, changed, rows)
    print(f"rewrote {len(changed)} label(s) in ai.topics, their label_embedding, "
          "and the topics collection in Qdrant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
