"""Ranking text by the words it actually contains.

A dense vector is a summary of meaning, and a summary loses exactly the thing a
student's question often turns on: a name. "قصيدة الشباب لإيليا أبو ماضي" and
"ما معنى التسويف؟" both name something the corpus contains once, and both were
answered from the wrong lesson — five Grade 7 lessons about success and youth
occupy almost the same region of the embedding space, and their top scores sat
within 0.04 of each other.

Measured on the 14-query golden set, dense alone puts the right lesson first 10
times. Reciprocal-rank fusion with BM25 over the same candidates puts it first
13 times, and the three it fixes are exactly the title and proper-name queries.
A cross-encoder reranker was measured against the same set for comparison: 11 of
14, for 1.8 seconds a query.

Both halves are folded the same way (``shared/matching``) before they are
compared, because the textbook prints ``التَّسْوِيفِ`` and the student types
``التسويف``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

from .matching import fold_for_match

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

# Robertson's defaults. Nothing in this corpus argues for moving them, and a
# number that has not been measured should be the one everyone else measured.
K1 = 1.5
B = 0.75

# The rank-fusion constant from Cormack et al. Its job is to stop rank 1 from
# dominating rank 2 outright; 60 is the published value.
RRF_K = 60


def tokenise(text: str) -> list[str]:
    """Words, folded to the form two pieces of text are compared in."""
    return _WORD.findall(fold_for_match(text))


def bm25_scores(query: str, documents: Sequence[str]) -> list[float]:
    """BM25 of ``query`` against each document, scored within this set.

    Document frequency comes from the candidate set rather than the whole
    corpus. That is a deliberate narrowing: these documents are already the ones
    a dense search considered close, so a term that is rare *among them* is
    precisely the term that tells them apart — which is the discrimination the
    dense stage failed to make.

    It also means no corpus-wide index has to exist, be kept in memory, or be
    rebuilt when a lesson is re-ingested.
    """
    if not documents:
        return []

    tokenised = [tokenise(d) for d in documents]
    count = len(tokenised)
    lengths = [len(d) for d in tokenised]
    average_length = (sum(lengths) / count) or 1.0

    frequency: Counter[str] = Counter()
    for document in tokenised:
        frequency.update(set(document))

    terms = tokenise(query)
    scores: list[float] = []
    for document, length in zip(tokenised, lengths):
        counts = Counter(document)
        score = 0.0
        for term in terms:
            occurrences = counts.get(term)
            if not occurrences:
                continue
            appearances = frequency[term]
            idf = math.log(1 + (count - appearances + 0.5) / (appearances + 0.5))
            score += idf * occurrences * (K1 + 1) / (
                occurrences + K1 * (1 - B + B * length / average_length))
        scores.append(score)
    return scores


def reciprocal_rank_fusion(*rankings: Sequence[int], k: int = RRF_K) -> list[float]:
    """Fuse several rankings of the same items into one score per item.

    Each ranking is a list of 1-based positions, one per item, in the same order
    as the items; a position of 0 means the ranking did not place the item at
    all. Ranks rather than scores, because a cosine and a BM25 score are not on
    the same scale and no amount of normalisation makes them comparable.
    """
    if not rankings:
        return []
    size = len(rankings[0])
    fused = [0.0] * size
    for ranking in rankings:
        for index, position in enumerate(ranking):
            if position:
                fused[index] += 1.0 / (k + position)
    return fused


def ranks_from_scores(scores: Sequence[float], *, skip_zero: bool = False) -> list[int]:
    """1-based positions, highest score first; 0 for an item that does not place."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    positions = [0] * len(scores)
    for position, index in enumerate(order, start=1):
        if skip_zero and not scores[index]:
            continue
        positions[index] = position
    return positions
