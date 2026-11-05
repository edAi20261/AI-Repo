from __future__ import annotations

from ingestion.lesson_pages import file_name as _file_name
from scripts.attach_lesson_pages import _spans


def test_a_lesson_runs_to_the_page_before_the_next_and_the_last_to_the_end():
    """A lesson was shown with no pages; the pages cut for it must cover its
    figure-only pages too, which have no passage to say where they belong."""
    rows = [("b", "s", "Two", 17, False, None), ("a", "s", "One", 1, False, None),
            ("c", "s", "Three", 25, True, None)]
    assert [(row[0], first, last) for row, first, last in _spans(rows, 32)] == [
        ("a", 1, 16), ("b", 17, 24), ("c", 25, 32)]


def test_a_lesson_with_no_passages_is_not_given_pages():
    rows = [("a", "s", "One", 1, False, 1), ("b", "s", "Empty", None, False, None)]
    assert [row[0] for row, _, _ in _spans(rows, 10)] == ["a"]


def test_the_attachment_is_named_after_the_lesson_in_its_own_script():
    assert _file_name("الدرس الأول: تركيب الذرة", "id") == "الدرس الأول تركيب الذرة.pdf"
    assert _file_name("???", "lesson-id") == "lesson-id.pdf"


def test_a_title_with_vowel_marks_keeps_its_words_in_the_file_name():
    """A vowel mark is not a letter, so each one became a space: the attachment
    of "لَا تَلُمْنِي فِي هَوَاهَا" was named "ل ا ت ل م ن ي ف ي ه و اه ا.pdf"."""
    assert _file_name("لَا تَلُمْنِي فِي هَوَاهَا", "id") == "لَا تَلُمْنِي فِي هَوَاهَا.pdf"
    assert _file_name("Café crème: déjà vu", "id") == "Café crème déjà vu.pdf"


def test_a_lesson_followed_by_back_matter_with_no_passages_ends_at_its_own_last_page():
    """The glossary and answer pages after the last lesson have no passages, so
    they could not bound it: a 14-page lesson was cut as a 133-page PDF."""
    rows = [("a", "s", "One", 1, False, 9), ("b", "s", "Last", 10, False, 23)]
    assert [(row[0], first, last) for row, first, last in _spans(rows, 140)] == [
        ("a", 1, 9), ("b", 10, 23)]
