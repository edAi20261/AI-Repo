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


# --- folding is not only for Arabic -----------------------------------------

def test_hebrew_points_fold_the_way_arabic_vowels_do():
    """A pointed Hebrew textbook against an unpointed query is the same
    mismatch tashkeel was: the page carries the marks, nobody types them."""
    from shared.matching import fold_for_match

    assert fold_for_match("מַחֲזוֹר הַמַּיִם") == fold_for_match("מחזור המים")


def test_latin_accents_are_not_folded_away():
    """French and Vietnamese marks carry meaning — folding them would merge
    words the writer distinguished."""
    from shared.matching import fold_for_match

    assert "é" in fold_for_match("café")
    assert fold_for_match("café") != fold_for_match("cafe")


# --- the parts of a book that are not content -------------------------------

def test_a_back_of_book_index_is_apparatus():
    """The last pages of a textbook are terms and page numbers. They share a
    word with almost any question and answer none of them. Measured over a
    371-page module, index entries per word: 0.066 through the back matter
    against 0.0003 through the body."""
    from shared.matching import looks_like_apparatus

    assert looks_like_apparatus(
        "adhesion, 10 climate and, 232 cohesion, 10 conduction and, 125 "
        "conservation, 2-3 temperate zone, 234 temperature, 116 atmosphere "
        "and, 107, 108, 110 climate and, 226, 231 subduction, 61")


def test_a_contents_page_of_dotted_leaders_is_apparatus():
    """Rows of dots carry no meaning and inflate every full-stop count."""
    from shared.matching import looks_like_apparatus

    assert looks_like_apparatus(
        "Program Overview . . . . . . . . . . . . . . . . 1\n"
        "Earth's Water . . . . . . . . . . . . . . . . . . 5\n"
        "The Water Cycle . . . . . . . . . . . . . . . . 20")


def test_prose_is_never_apparatus():
    """Whatever it is about, and in whatever script."""
    from shared.matching import looks_like_apparatus

    assert not looks_like_apparatus(
        "A watershed is the area of land that is drained by a river system. "
        "Watersheds are separated from one another by divides, which are ridges "
        "of higher land. Water flows downhill from the divide into the river.")
    assert not looks_like_apparatus(
        "جابر بن حيان عالم عربي بارز اشتهر في مجال الكيمياء، وقد لقبه علماء "
        "الغرب بأبي الكيمياء. عاش في القرن الثاني الهجري وترك إرثا علميا كبيرا.")
    assert not looks_like_apparatus(
        "水循环在海洋和天空之间移动水。太阳提供能量。水蒸发后形成云，然后降雨回到地面。")


def test_prose_carrying_figures_is_not_apparatus():
    """A science page is full of numbers. Digit density was the first signal
    tried and it flagged two pages of real bathymetric map exercises, whose
    depth readings run 0.129 digits per character."""
    from shared.matching import looks_like_apparatus

    assert not looks_like_apparatus(
        "The highest temperature ever recorded on Earth was 58 degrees Celsius, "
        "measured in Libya in 1922. The lowest was minus 89 degrees Celsius at "
        "Vostok Station in Antarctica in 1983, some 3488 metres above sea level.")


def test_a_chart_axis_is_not_an_index():
    """A thousands separator sits between two digits; an index entry sits
    between a word and a digit. That is the whole difference."""
    from shared.matching import looks_like_apparatus

    assert not looks_like_apparatus(
        "You Try It! Now follow the steps below to draw the profile for this "
        "bathymetric map of a region near Monterey Bay, California. Depth in "
        "metres 0 10 20 30 40 50 -1,000 -2,000 -3,000 -4,000 -2,500 Area of "
        "profile NORTH Use the grid to plot each depth reading in turn.")


def test_an_empty_chunk_is_not_apparatus():
    from shared.matching import looks_like_apparatus

    assert not looks_like_apparatus("")
