from __future__ import annotations

from ingestion.lesson_pages import file_name as _file_name
from scripts.attach_lesson_pages import _spans


def test_a_lesson_runs_to_the_page_before_the_next_and_the_last_to_the_end():
    """A lesson was shown with no pages; the pages cut for it must cover its
    figure-only pages too, which have no passage to say where they belong."""
    rows = [("b", "s", "Two", 17, False), ("a", "s", "One", 1, False),
            ("c", "s", "Three", 25, True)]
    assert [(row[0], first, last) for row, first, last in _spans(rows, 32)] == [
        ("a", 1, 16), ("b", 17, 24), ("c", 25, 32)]


def test_a_lesson_with_no_passages_is_not_given_pages():
    rows = [("a", "s", "One", 1, False), ("b", "s", "Empty", None, False)]
    assert [row[0] for row, _, _ in _spans(rows, 10)] == ["a"]


def test_the_attachment_is_named_after_the_lesson_in_its_own_script():
    assert _file_name("الدرس الأول: تركيب الذرة", "id") == "الدرس الأول تركيب الذرة.pdf"
    assert _file_name("???", "lesson-id") == "lesson-id.pdf"
