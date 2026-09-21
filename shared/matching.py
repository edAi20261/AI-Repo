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
