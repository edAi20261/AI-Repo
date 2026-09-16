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


def test_intent_matching_folds_a_message_the_way_retrieval_does():
    """There were three hand-written folds. Two of them missed ٱ, ئ, ة and ؤ,
    so a student typing "مدرسه" was matched one way by search and another way
    by the intent router — and the router is what decides whether the question
    reaches search at all."""
    from agent.intent import normalise

    assert normalise("مدرسة") == normalise("مدرسه")
    assert normalise("مسئول") == normalise("مسيول")
    assert normalise("ٱلدرس") == normalise("الدرس")


def test_courtesy_matching_folds_a_message_the_way_retrieval_does():
    """The same fold, in the other place that had its own copy."""
    from shared.language.request import _fold

    assert _fold("شكراً") == _fold("شكرا")
    assert _fold("مدرسة") == _fold("مدرسه")


# --- lexical ranking --------------------------------------------------------

def test_a_name_in_the_question_outranks_a_paraphrase_of_it():
    """Why retrieval is two-stage. A dense vector summarises meaning, and the
    summary loses the name a question turns on: five Grade 7 lessons about
    success and youth scored within 0.04 of each other, so the lesson that
    actually names the poet came second."""
    from shared.lexical import bm25_scores

    query = "من هو إيليا أبو ماضي؟"
    named = "قال إيليا أبو ماضي في قصيدته عن الشباب"
    paraphrase = "الشاعر يتحدث عن الأمل والتفاؤل في الحياة"

    scores = bm25_scores(query, [paraphrase, named])
    assert scores[1] > scores[0]


def test_the_lexical_stage_folds_the_way_retrieval_does():
    """The textbook prints التَّسْوِيفُ and the student types التسويف. Comparing
    them unfolded compares two different strings."""
    from shared.lexical import bm25_scores

    assert bm25_scores("التسويف", ["الكسل، التَّسْوِيفُ، الفوضى"])[0] > 0


def test_a_word_carrying_an_attached_conjunction_is_a_different_word():
    """A known limit, recorded rather than guessed at. Arabic attaches و/ب/ل/ك
    to the front of a word, so وَالتَّسْوِيف and التسويف do not match on tokens.
    Stripping those prefixes was measured against the golden set and moved
    precision@1 by nothing at all (11/12 either way), so the stemmer is not
    here: an unmeasured transformation that can only create false matches is
    worse than a miss you know about."""
    from shared.lexical import bm25_scores

    assert bm25_scores("التسويف", ["الكسل وَالتَّسْوِيفُ وَالفَوْضَى"])[0] == 0


def test_a_document_matching_no_query_word_does_not_place():
    """Fusion adds a rank to a rank. A document that matches nothing must not be
    lifted merely for being in the candidate pool — that is how a lexical stage
    turns a decline into a confident wrong answer."""
    from shared.lexical import ranks_from_scores

    assert ranks_from_scores([2.0, 0.0, 1.0], skip_zero=True) == [1, 0, 2]


def test_fusion_uses_rank_not_score():
    """A cosine and a BM25 score are not on the same scale, and no normalisation
    makes them comparable; reciprocal rank fusion never compares them."""
    from shared.lexical import reciprocal_rank_fusion

    fused = reciprocal_rank_fusion([1, 2, 3], [3, 1, 2], k=60)
    assert fused[1] > fused[0] > fused[2]
