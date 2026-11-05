from scripts.check_integrity import (
    Result,
    out_of_order,
    pdf_suspects,
    shared_passages,
)


def test_two_lessons_filed_the_same_section_are_reported_as_copies():
    """21 paired readings each held both texts and the unit's writing task: the
    split gave both selections the same pages, so every passage was filed twice."""
    rows = [("s", "a", "Thank You, M'am", h) for h in "xyzw"]
    rows += [("s", "b", "A Police Stop", h) for h in "xyzv"]
    assert shared_passages(rows) == [("Thank You, M'am", "A Police Stop", 3)]


def test_a_passage_two_subjects_share_is_not_a_copy():
    """The same book is filed under more than one grade on purpose."""
    rows = [("s1", "a", "A", h) for h in "xyz"] + [("s2", "b", "B", h) for h in "xyz"]
    assert shared_passages(rows) == []


def test_front_matter_listed_after_the_last_unit_is_out_of_order():
    """Two front-matter sections of Into Literature G7 (pp.22 and 28) were
    listed after the unit 6 selections that start on p.506."""
    rows = [("s", "book", 1, 40, "Unit 1"), ("s", "book", 2, 506, "Triangle Factory Fire"),
            ("s", "book", 3, 22, "Must Read Essay")]
    assert out_of_order(rows) == [("Must Read Essay", 22, "Triangle Factory Fire", 506)]


def test_lessons_of_two_books_in_one_subject_are_ordered_per_book():
    """A subject made of two editions starts over at page 1 with the second."""
    rows = [("s", "vol 1", 1, 10, "1-1"), ("s", "vol 1", 2, 60, "1-6"),
            ("s", "vol 2", 3, 14, "5-1"), ("s", "vol 2", 4, 30, "5-2")]
    assert out_of_order(rows) == []


def test_paired_selections_starting_on_one_page_are_in_order():
    rows = [("s", "book", 1, 88, "Thank You, M'am"), ("s", "book", 2, 88, "A Police Stop")]
    assert out_of_order(rows) == []


def test_a_pdf_far_heavier_per_page_than_its_subject_is_opened():
    """A 14-page lesson was shown as a 133-page PDF: 9x its subject's bytes per page."""
    rows = [("s", f"l{i}", f"L{i}", 100_000, 10, "b", f"k{i}") for i in range(5)]
    rows.append(("s", "big", "14-7", 950_000, 10, "b", "kbig"))
    assert [r[1] for r in pdf_suspects(rows)] == ["big"]


def test_a_warning_never_fails_the_check():
    assert not Result("R9", "", ["empty subject"], warning=True).failed
    assert Result("R2", "", ["archived lesson"]).failed
    assert not Result("R2", "", []).failed
