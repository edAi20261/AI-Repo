from __future__ import annotations

from scripts.reingest_untagged import _UNTAGGED, reingest


def test_the_listing_is_the_ready_lessons_with_no_topics_or_no_passages():
    """196 lessons were recorded ready with 0 topics and 3 with 0 chunks."""
    assert "i.status = 'ready'" in _UNTAGGED
    assert "i.topic_count = 0 OR i.chunk_count = 0" in _UNTAGGED
    assert "l.status <> 'archived'" in _UNTAGGED


def test_each_lesson_is_reingested_and_one_failure_does_not_stop_the_rest():
    ingested = []

    def resolve(config, lesson_id):
        if lesson_id == "book-lesson":
            raise ValueError("lesson 'book-lesson' has no ingestible source")
        return {"lecture_id": lesson_id}

    def ingest(state, *, config):
        ingested.append(state["lecture_id"])
        if state["lecture_id"] == "broken":
            raise RuntimeError("vision down")
        return {"ingestion_status": "ready",
                "stats": {"index": {"chunks_written": 1}, "topics": {"extracted": 3}}}

    outcome = reingest(object(), ["subtraction", "book-lesson", "broken"],
                       resolve=resolve, ingest=ingest)

    assert ingested == ["subtraction", "broken"]
    assert outcome["ready"] == ["subtraction"]
    assert outcome["skipped"] == ["book-lesson"]
    assert outcome["failed"] == ["broken"]
