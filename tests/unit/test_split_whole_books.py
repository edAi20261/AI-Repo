from __future__ import annotations

from ingestion.books import Lesson, lesson_id_for
from ingestion.layout.bridge import parsed_pages
from scripts.split_whole_books import _filing, _pages

BOOK = "9d3f1b20-0002-4a00-9000-000000000006"
TOPIC_A, TOPIC_B, TOPIC_UNUSED = (f"00000000-0000-4000-8000-00000000000{n}" for n in (1, 2, 3))


def _lessons() -> list[Lesson]:
    return [Lesson(lesson_id=lesson_id_for(BOOK, n), ordinal=n, title=f"Lesson {n}",
                   unit_number=1, unit_title="", page_from=start, page_to=end)
            for n, (start, end) in enumerate([(1, 10), (11, 20)])]


def test_a_passage_is_filed_under_the_lesson_its_page_is_in():
    """A book ingested whole had every passage under the book; split in place,
    each must land under the lesson a student on that page is answered from."""
    chunks = [("c1", 3, "", "text", [TOPIC_A]), ("c2", 12, "", "text", [TOPIC_B]),
              ("c3", 20, "", "text", [])]
    passages, _ = _filing(_lessons(), chunks, [])
    first, second = (lesson.lesson_id for lesson in _lessons())
    assert passages == {first: ["c1"], second: ["c2", "c3"]}


def test_a_topic_goes_to_the_lesson_most_of_its_passages_went_to():
    """A topic is owned by one lesson; one that no passage names still needs an
    owner, or it would keep naming the archived whole book."""
    chunks = [("c1", 2, "", "t", [TOPIC_A]), ("c2", 14, "", "t", [TOPIC_A]),
              ("c3", 15, "", "t", [TOPIC_A])]
    _, topics = _filing(_lessons(), chunks, [TOPIC_A, TOPIC_UNUSED])
    first, second = (lesson.lesson_id for lesson in _lessons())
    assert topics == {second: [TOPIC_A], first: [TOPIC_UNUSED]}


def test_the_stored_passages_are_read_back_as_the_pages_the_breakdown_reads():
    """The breakdown reads parsed pages; a heading repeated on every passage of
    a page is one heading of that page, not one per passage."""
    chunks = [("c1", 1, "Unit 1", "first", []), ("c2", 1, "Unit 1", "second", []),
              ("c3", 2, "", "third", [])]
    pages = parsed_pages(_pages(BOOK, chunks))
    assert [p.meta.page for p in pages] == [1, 2]
    assert [(b.type, b.text) for b in pages[0].blocks] == [
        ("heading", "Unit 1"), ("body", "first"), ("body", "second")]
    assert [(b.type, b.text) for b in pages[1].blocks] == [("body", "third")]
