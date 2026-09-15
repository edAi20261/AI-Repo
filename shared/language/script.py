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
LATIN = re.compile(r"[A-Za-z]")

# 3 -> ع, 7 -> ح, 2 -> ء: a digit standing in for a letter inside a word is the
# signature of Arabizi, and never occurs in French, German or English prose.
ARABIZI_DIGIT_LETTER = re.compile(r"[a-z][2356789][a-z]", re.IGNORECASE)

ARABIZI_WORDS = frozenset({
    "momken", "mumkin", "ezay", "izzay", "eh", "da", "di", "dah", "dol",
    "el", "ana", "enta", "enti", "3ayez", "3awez", "3ashan", "mesh",
    "kda", "keda", "bardo", "khalas", "yalla", "shukran", "dars", "mudarris",
})

MIN_MIXED_SHARE = 0.30


def arabic_ratio(text: str) -> float:
    """Share of the letters in ``text`` written in Arabic script."""
    letters = sum(1 for char in text if char.isalpha())
    if not letters:
        return 0.0
    return len(ARABIC.findall(text)) / letters


def looks_arabizi(text: str) -> bool:
    """Whether Latin-script ``text`` is Arabic typed in Latin letters and digits."""
    if ARABIZI_DIGIT_LETTER.search(text):
        return True
    words = re.findall(r"[a-z0-9]+", text.lower())
    return sum(1 for word in words if word in ARABIZI_WORDS) >= 2


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
