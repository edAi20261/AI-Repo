"""Paired readings both held the pair's whole span: each lesson of 21 pairs held
both texts and the unit's writing task, about 150 passages filed twice."""

import pytest

from scripts.refile_selections import groups_of, headed, outside, spans


def test_each_selection_keeps_from_its_start_to_the_page_before_the_next():
    """'Thank You, M'am' is pp.88-99; 'A Police Stop' runs on from p.100 into the
    unit's writing task, as the book does."""
    found = spans({"police": 100, "maam": 88})
    assert found["maam"] == (88, 99)
    assert found["police"][0] == 100 and found["police"][1] == float("inf")


def test_two_selections_cannot_start_on_one_page():
    with pytest.raises(ValueError):
        spans({"a": 88, "b": 88})


def test_a_lesson_loses_only_what_lies_outside_its_own_pages():
    passages = [("c1", 88), ("c2", 99), ("c3", 100), ("c4", 125)]
    assert outside(passages, (88, 99)) == ["c3", "c4"]
    assert outside(passages, (100, float("inf"))) == ["c1", "c2"]


def test_lessons_sharing_passages_are_one_group():
    assert groups_of([("a", "b"), ("b", "c"), ("x", "y")]) == [["a", "b", "c"], ["x", "y"]]


def test_a_title_is_found_where_the_book_prints_it_as_a_heading():
    """The scanned books carry no text layer; the stored reading of each page
    says which lines are headings."""
    pages = [(258, {"blocks": [{"type": "heading", "text": "ODE TO ENCHANTED LIGHT"}]}),
             (266, {"blocks": [{"type": "body", "text": "Sleeping in the Forest"},
                               {"type": "heading", "text": "SLEEPING IN THE FOREST"}]}),
             (267, {"blocks": [{"type": "body", "text": "Sleeping in the Forest"}]})]
    assert headed(pages, "Sleeping in the Forest") == [266]
    assert headed(pages, "Ode to enchanted light") == [258]
