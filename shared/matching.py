"""Folding text to the form two pieces of it should be compared in.

Egyptian prep textbooks are printed fully vocalised — ``جَابِرُ بْنُ حَيَّانَ``
— and students type ``جابر بن حيان``. To a tokeniser those are different
strings, and the gap is expensive: a correct Arabic lesson scored 0.42 against
its own question while an unrelated lesson scored 0.48 against a question the
syllabus could not answer at all. The right answer ranked below the wrong one,
and no threshold could separate them.

Folding the diacritics away before embedding is worth about +0.15 similarity on
real questions, and it pushes unrelated matches down at the same time. Latin
text is untouched, which is why the symptom looked specific to Arabic.

This is a *matching* form, never a display form. The original text is what the
student reads and what the model is given as context; only the vector is built
from the fold.
"""

from __future__ import annotations

import re
import unicodedata

# Marks a writer sets above and below a letter that a reader supplies from
# knowledge and a searcher never types: Arabic tashkeel (fathatan..sukun,
# superscript alef, the hamza and madda marks) and Hebrew niqqud and
# cantillation. Latin, Vietnamese and Cyrillic marks are deliberately absent —
# there they separate words rather than decorate them, so folding café into
# cafe would merge what the writer distinguished.
DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u0653-\u0655"
                        r"\u0591-\u05BD\u05BF\u05C1\u05C2\u05C4\u05C5\u05C7]")
TATWEEL = "ـ"

# Letters an Arabic writer treats as interchangeable and a tokeniser does not.
# Folding them costs nothing in meaning and buys every spelling of a word.
_LETTERS = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ئ": "ي",
    "ة": "ه",
    "ؤ": "و",
})

_WHITESPACE = re.compile(r"\s+")


def fold_for_match(text: str) -> str:
    """The form of ``text`` that matching compares, never the form we show."""
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", text)
    folded = folded.replace(TATWEEL, "")
    folded = DIACRITICS.sub("", folded)
    folded = folded.translate(_LETTERS)
    return _WHITESPACE.sub(" ", folded).strip()


# A contents page repeats "الدرس الأول / الدرس الثاني / الوحدة" down the page.
# A lesson body names its own lesson at most once. Measured on the indexed
# corpus the gap is clean: contents pages carry five such markers, no real
# lesson chunk carries more than one.
_NAVIGATION_MARKER = re.compile(
    r"الدَّرْسُ|الدرس|الوَحْدَةُ|الوحدة|الوحده|\blesson\s+\d|\bunit\s+\d|"
    r"\btable\s+of\s+contents\b|\bcontents\b|فهرس|محتويات",
    re.IGNORECASE,
)

MIN_NAVIGATION_MARKERS = 3


def looks_like_contents(text: str) -> bool:
    """Whether a chunk is a contents page rather than something to answer from.

    A contents page is a dense list of lesson titles, so it matches any
    title-shaped question better than the lesson itself does — a student asking
    to be taught a lesson was handed the index page instead. It is navigation,
    and never an answer.
    """
    if not text:
        return False
    return len(_NAVIGATION_MARKER.findall(text)) >= MIN_NAVIGATION_MARKERS


# What separates the apparatus of a book from its content, measured over a
# 371-page Grade 6 science module. The signal is the shape of an index entry —
# a term, a comma, a page number — per word of the chunk:
#
#                         chunks   entries/word mean   median    max
#   front matter              11         0.0002        0.0000   0.0022
#   body                     473         0.0003        0.0000   0.0381
#   back-of-book index        61         0.0661        0.0000   0.4444
#
# Digit density was the first thing tried and it is wrong: a science page is
# full of figures, and two pages of bathymetric map exercises — real content,
# with depth readings down an axis — came out looking exactly like an index at
# 0.129 digits per character. They score 0.0000 on this one, because a
# thousands separator sits between two digits and an index entry sits between a
# word and a digit.
#
# Dotted leaders are their own signal: rows of full stops carry no meaning and
# nothing else produces them. Both are structural, so they read the same in any
# script.
INDEX_ENTRIES_PER_WORD = 0.04
LEADER_STOPS_PER_WORD = 0.5
LEADER_MIN_WORDS = 4
INDEX_MIN_WORDS = 12

# A term, a comma, a page number: "adhesion, 10". Never "-1,000".
_INDEX_ENTRY = re.compile(r"[^\W\d_]\s*,\s*\d")
_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)
_STOP = re.compile(r"[.!?]")


def looks_like_apparatus(text: str) -> bool:
    """Whether a chunk is part of finding the content rather than being it.

    A back-of-book index or a table of contents: a dense list of terms and page
    numbers that shares a word with almost any question and answers none of
    them. The same failure as a contents page, arriving by a different route.

    Deliberately narrow. A glossary is not apparatus — it defines the terms a
    student asks about — and neither is a page of equations. Losing content
    costs more than carrying a little apparatus.
    """
    words = _TOKEN.findall(text)
    if len(words) < LEADER_MIN_WORDS:
        return False
    # Two signals, two minimums. Rows of dots are unambiguous and a contents
    # line is short by nature, so a handful of words is enough to judge one. An
    # entry rate over a few words is noise, so that one waits for a chunk worth
    # measuring.
    if len(_STOP.findall(text)) / len(words) >= LEADER_STOPS_PER_WORD:
        return True
    if len(words) < INDEX_MIN_WORDS:
        return False
    return len(_INDEX_ENTRY.findall(text)) / len(words) >= INDEX_ENTRIES_PER_WORD


def chunk_kind(text: str) -> str | None:
    """What a chunk is, when it is not something to answer from.

    ``None`` means content. Anything else is navigation or apparatus, and
    retrieval excludes it: a contents page or an index shares a word with
    almost any question and answers none of them.

    One function because three places decide this — the index stage, the
    re-embed script and the query filter — and a chunk written as one kind and
    excluded as another is excluded by accident.
    """
    if looks_like_contents(text):
        return "contents"
    if looks_like_apparatus(text):
        return "apparatus"
    return None


# The kinds retrieval will not answer from.
NOT_CONTENT = ("contents", "apparatus")
