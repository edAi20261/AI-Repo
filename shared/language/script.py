"""Writing-system checks that decide a language without statistics.

Script is evidence a detector cannot improve on: Arabic letters mean Arabic.
The same pass identifies the two shapes no detector handles — Arabizi (Arabic
typed in Latin letters and digits) and balanced code-switching — so the resolver
can decline instead of guessing at them.
"""

from __future__ import annotations

import re

from .types import Script

ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")

# Every right-to-left script, not only the one the pilot corpus is written in:
# Hebrew, Syriac, Thaana, N'Ko, Samaritan and Mandaic alongside Arabic and its
# supplements. Direction is a property of the script, not of the language, so a
# Persian and an Urdu page are covered by the Arabic ranges already.
RIGHT_TO_LEFT = re.compile(
    r"[\u0590-\u05FF\u0600-\u06FF\u0700-\u074F\u0750-\u077F\u0780-\u07BF"
    r"\u07C0-\u07FF\u0800-\u083F\u0840-\u085F\u0860-\u086F\u0870-\u089F"
    r"\u08A0-\u08FF\uFB1D-\uFB4F\uFB50-\uFDFF\uFE70-\uFEFF]")
LATIN = re.compile(r"[A-Za-z]")

# 3 -> ع, 7 -> ح, 2 -> ء: a digit standing in for a letter inside a word is the
# signature of Arabizi, and never occurs in French, German or English prose.
ARABIZI_DIGIT_LETTER = re.compile(r"[a-z][2356789][a-z]", re.IGNORECASE)


MIN_MIXED_SHARE = 0.30


def arabic_ratio(text: str) -> float:
    """Share of the letters in ``text`` written in Arabic script."""
    letters = sum(1 for char in text if char.isalpha())
    if not letters:
        return 0.0
    return len(ARABIC.findall(text)) / letters


def looks_arabizi(text: str) -> bool:
    """Whether Latin-script ``text`` is Arabic typed in Latin letters and digits."""
    return bool(ARABIZI_DIGIT_LETTER.search(text))


def classify(text: str, *, arabic_threshold: float = MIN_MIXED_SHARE) -> Script:
    """The writing system of ``text``, or the reason it cannot be decided."""
    if not text or not text.strip():
        return "none"

    ratio = arabic_ratio(text)
    has_latin = bool(LATIN.search(text))

    if ratio >= arabic_threshold:
        return "arabic"
    if ratio > 0 and has_latin and ratio >= MIN_MIXED_SHARE / 2:
        return "mixed"
    if not has_latin:
        return "none"
    return "arabizi" if looks_arabizi(text) else "latin"


def written_in(text: str, language: str) -> bool:
    """Whether ``text`` is written in the script ``language`` uses.

    Coarse on purpose — it separates Arabic from Latin and nothing finer, which
    is all that can be told from characters alone. It is a guard against a model
    answering in the wrong language entirely, not a language detector.
    """
    if not text or not text.strip():
        return False
    arabic = bool(ARABIC.search(text))
    return arabic if language == "ar" else not arabic


def rtl_ratio(text: str) -> float:
    """Share of the letters in ``text`` written in a right-to-left script."""
    letters = sum(1 for char in text if char.isalpha())
    if not letters:
        return 0.0
    return len(RIGHT_TO_LEFT.findall(text)) / letters


def reading_direction(text: str) -> str:
    """Which side of a column ``text`` is read from.

    Decided by what the text is actually written in, never by a setting: one
    global direction reads every English, French and German page backwards the
    moment the corpus stops being Arabic. A page with no letters at all — a
    table of sums, a diagram — falls back to left to right, which is what every
    Latin, Cyrillic and CJK corpus needs and what a page with nothing to go on
    should not be the exception to.
    """
    return "rtl" if rtl_ratio(text) >= 0.5 else "ltr"
