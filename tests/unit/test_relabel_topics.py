"""Relabelling a lesson's topics must not change which skill an id names.

46 of 79 topics carried an English label on an Arabic lesson, so an Arabic answer
ended in English prerequisites. The producer is fixed; these are the rows that
were already written.

The repair relabels rather than re-extracts, and that choice is the whole reason
these checks exist: a topic_id is what a student's mastery history hangs from, so
an id that goes missing, gets invented, or moves to a different skill costs
exactly what re-extraction would have cost.
"""

from __future__ import annotations

import pytest

from ingestion.schemas import RelabelList

from scripts.relabel_topics import _same_topics

A, B = "11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222"


def _batch(*pairs) -> RelabelList:
    return RelabelList(topics=[{"topic_id": i, "label": label} for i, label in pairs])


def test_a_faithful_relabel_passes():
    _same_topics(_batch((A, "تحديد الفكرة الرئيسية"), (B, "صياغة الجمل")), {A, B}, "ar")


def test_a_dropped_topic_is_refused():
    """The model summarising eight skills into five loses three topic_ids, and
    with them every mastery event earned on those skills."""
    with pytest.raises(ValueError, match="Missing"):
        _same_topics(_batch((A, "تحديد الفكرة الرئيسية")), {A, B}, "ar")


def test_an_invented_topic_is_refused():
    with pytest.raises(ValueError, match="Not asked for"):
        _same_topics(_batch((A, "تحديد"), (B, "صياغة"), ("33333333-3333-3333-3333-333333333333", "جديد")),
                     {A, B}, "ar")


def test_a_repeated_id_is_refused():
    """Two labels for one id: whichever is written last silently wins."""
    with pytest.raises(ValueError, match="Duplicated"):
        _same_topics(_batch((A, "تحديد"), (A, "صياغة")), {A, B}, "ar")


def test_a_label_still_in_the_wrong_language_is_refused():
    """The failure this whole script exists to undo, refused at creation rather
    than written and hidden at display."""
    with pytest.raises(ValueError, match="must be written in 'ar'"):
        _same_topics(_batch((A, "Identify main idea"), (B, "صياغة")), {A, B}, "ar")
