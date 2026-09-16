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

# Fathatan..sukun, superscript alef, and the hamza/madda marks above and below.
DIACRITICS = re.compile(r"[ً-ٰٕ]")
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
