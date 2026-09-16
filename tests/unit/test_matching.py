"""Folding text to the form matching compares it in.

The named regression: a student asked about a lesson that exists and was
answered from an unrelated one. The lesson PDFs are printed fully vocalised and
students type plain Arabic, so the query and the document never shared a token.
"""

from __future__ import annotations

import pytest

from shared.matching import fold_for_match


@pytest.mark.parametrize("vocalised,typed", [
    ("جَابِرُ بْنُ حَيَّانَ", "جابر بن حيان"),
    ("الْمَدْرَسَةُ", "المدرسه"),
    ("الطَّرْحُ", "الطرح"),
    ("رِيَادَةُ الْأَعْمَالِ", "رياده الاعمال"),
])
def test_a_textbook_and_a_student_write_the_same_word(vocalised, typed):
    """+0.15 similarity on real questions, measured — see scripts/eval_retrieval.py."""
    assert fold_for_match(vocalised) == fold_for_match(typed)


@pytest.mark.parametrize("a,b", [
    ("أحمد", "احمد"),      # alef with hamza
    ("إبراهيم", "ابراهيم"),
    ("مصطفى", "مصطفي"),    # alef maqsura
    ("مدرسة", "مدرسه"),    # ta marbuta
    ("مسؤول", "مسوول"),
])
def test_interchangeable_letters_fold_together(a, b):
    assert fold_for_match(a) == fold_for_match(b)


@pytest.mark.parametrize("text", [
    "explain subtraction",
    "What is column subtraction?",
    "Newton's first law",
])
def test_latin_text_is_left_alone(text):
    """English retrieval was never the problem and must not become one."""
    assert fold_for_match(text) == text


def test_tatweel_and_spacing_are_cleaned():
    assert fold_for_match("الــــدرس") == "الدرس"
    assert fold_for_match("  a   b  ") == "a b"


def test_empty_input_is_safe():
    assert fold_for_match("") == ""
    assert fold_for_match(None) == ""


def test_the_fold_is_idempotent():
    once = fold_for_match("جَابِرُ بْنُ حَيَّانَ")
    assert fold_for_match(once) == once


def test_the_embedder_folds_both_sides():
    """The asymmetry is the whole risk: a folded document and an unfolded query
    match worse than folding neither. Doing it in the base class is what makes
    that impossible."""
    from shared.embeddings import HashEmbedder

    embedder = HashEmbedder(dimension=64)
    vocalised = embedder.embed(["جَابِرُ بْنُ حَيَّانَ"])[0]
    typed = embedder.embed_query("جابر بن حيان")
    assert vocalised == typed
