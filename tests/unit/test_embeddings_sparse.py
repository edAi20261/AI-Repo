"""Sparse vectors: the lexical half of a hybrid search, in any script.

Dense retrieval finds a passage that means the same thing; sparse retrieval
finds the passage that uses the same words. A syllabus is full of terms a
student types exactly — a name, a formula, a date — and averaging them into a
1024-dimensional vector is where they go missing.
"""

from __future__ import annotations

from shared.config import get_config
from shared.embeddings import get_embedder


def _sparse(text: str) -> dict[int, float]:
    return get_embedder(get_config()).embed_sparse([text])[0]


def test_every_script_produces_terms_to_match_on():
    """Whitespace tokenisation gives Chinese one token for a whole sentence.
    The terms come from the model's own multilingual tokenizer instead, so a
    Chinese, Arabic, Hindi or German page is as searchable as an English one."""
    for text in ("The water cycle moves water between sea and sky.",
                 "دورة الماء تنقل الماء بين البحر والسماء",
                 "水循环在海洋和天空之间移动水",
                 "जल चक्र समुद्र और आकाश के बीच जल ले जाता है",
                 "Der Wasserkreislauf bewegt Wasser zwischen Meer und Himmel"):
        terms = _sparse(text)
        assert len(terms) >= 4, text
        assert all(weight > 0 for weight in terms.values())


def test_a_word_used_twice_weighs_more_than_one_used_once_but_not_double():
    """Term frequency saturates: a passage repeating a word ten times is not ten
    times the answer, which is the whole point of BM25's tf component."""
    once = _sparse("photosynthesis")
    twice = _sparse("photosynthesis photosynthesis")
    key = max(once, key=once.get)
    assert twice[key] > once[key]
    assert twice[key] < 2 * once[key]


def test_a_query_and_a_passage_are_folded_the_same_way():
    """The fold that makes a vocalized page match an unvocalized query has to
    apply to the sparse half too, or half of retrieval sees a different corpus."""
    vocalized = _sparse("رِيَادَةُ الأَعْمَالِ")
    plain = _sparse("ريادة الاعمال")
    assert set(vocalized) & set(plain)


def test_a_page_with_no_words_has_no_terms():
    assert _sparse("") == {}
