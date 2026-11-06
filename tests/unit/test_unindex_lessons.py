from scripts.unindex_lessons import by_namespace, orphans
from shared.namespaces import ARCHIVED, BACKGROUND, SYLLABUS


def test_lessons_are_put_back_in_the_namespace_each_came_from():
    """An archived study guide is background, an archived lesson is syllabus;
    putting both back as one would show the study guide as a lesson."""
    rows = [("c1", "guide", BACKGROUND), ("c2", "guide", BACKGROUND), ("c3", "lesson", SYLLABUS)]
    assert by_namespace(rows) == {BACKGROUND: ["guide"], SYLLABUS: ["lesson"]}


def test_a_point_with_no_passage_behind_it_is_an_orphan():
    """Three points of deleted test lessons stayed searchable for weeks."""
    assert orphans({"p1", "p2", "p3"}, {"p1", "p2"}) == ["p3"]


def test_the_archived_namespace_is_not_one_any_search_reads():
    assert ARCHIVED not in (SYLLABUS, BACKGROUND)
