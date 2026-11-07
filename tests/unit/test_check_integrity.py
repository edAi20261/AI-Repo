from scripts.check_integrity import (
    Result,
    copy_lessons,
    lesson_pages_span,
    out_of_order,
    pdf_suspects,
    shared_passages,
    stale_sizes,
)


def test_two_lessons_filed_the_same_section_are_reported_as_copies():
    """21 paired readings each held both texts and the unit's writing task: the
    split gave both selections the same pages, so every passage was filed twice."""
    rows = [("s", "a", "Thank You, M'am", h, p) for h, p in zip("xyzw", (88, 89, 90, 91))]
    rows += [("s", "b", "A Police Stop", h, p) for h, p in zip("xyzv", (88, 89, 90, 100))]
    assert shared_passages(rows) == [("Thank You, M'am", "A Police Stop", 3)]


def test_a_passage_two_subjects_share_is_not_a_copy():
    """The same book is filed under more than one grade on purpose."""
    rows = [("s1", "a", "A", h, 1) for h in "xyz"] + [("s2", "b", "B", h, 2) for h in "xyz"]
    assert shared_passages(rows) == []


def test_two_sections_that_start_on_one_page_may_both_carry_it():
    """Ingestion gives a page two sections share to both (shared_pages: copy):
    StudySync's library guide puts CATALOG and CARD SERVICES on one page."""
    rows = [("s", "a", "CATALOG", h, 766) for h in "xyz"]
    rows += [("s", "b", "CARD SERVICES", h, 766) for h in "xyz"]
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


def test_exercise_pages_holding_only_figures_are_the_lessons_own_pages():
    """McGraw G8 25.2 Line Graphs: text on pp.352-354, its graph exercises on pp.355-358."""
    assert lesson_pages_span(352, 354, 358) == 7


def test_a_lesson_without_figures_spans_its_passages():
    assert lesson_pages_span(149, 157, None) == 9


def test_a_pdf_record_whose_size_is_not_its_file_is_reported():
    """Holt C2 1-12: the record said 312,877 bytes for a 2,392,165-byte file."""
    records = [("1-12", 312_877, 2_392_165), ("1-13", 400_000, 400_000)]
    assert stale_sizes(records) == ["1-12 | recorded 312877 bytes, file 2392165"]


def test_a_pdf_missing_from_storage_is_reported():
    assert stale_sizes([("1-14", 500, None)]) == ["1-14 | recorded 500 bytes, file missing"]


def test_a_lesson_holding_nothing_another_lesson_does_not_is_a_copy():
    """Holt's Skills Bank filed "Factors" and "Roman Numerals" from one page, each
    holding the whole page: 88 shown lessons had no passage of their own."""
    rows = [("s", "a", "Factors", "p808"), ("s", "b", "Roman Numerals", "p808"),
            ("s", "c", "Divisibility Rules", "p807"), ("s", "c", "Divisibility Rules", "x")]
    assert copy_lessons(rows) == ["Factors", "Roman Numerals"]


def test_a_lesson_repeating_some_of_another_lessons_text_is_not_a_copy():
    """A pretest and a posttest print the same instructions; each has its own questions."""
    rows = [("s", "pre", "Pretest", "rules"), ("s", "pre", "Pretest", "q1"),
            ("s", "post", "Posttest", "rules"), ("s", "post", "Posttest", "q2"),
            ("t", "other", "Other subject", "q1")]
    assert copy_lessons(rows) == []
