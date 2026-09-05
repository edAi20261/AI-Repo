"""Born-digital legibility gate.

A page whose fonts are subset Identity-H with no ToUnicode CMap extracts to glyph-id
noise, not characters, yet carries plenty of characters — so the char-count scanned
check passes it through as clean digital text. Two text-based signals catch it:

  - symbol ratio: non-alphanumeric fraction over a threshold (language-agnostic, any
    length) — symbol-junk.
  - common-word hit rate: letter-shaped substitution-cipher noise has near-zero hits
    against a per-language common-word list; checked only past a min length so short
    readable pages are not misjudged.

The word check is fuzzy (a readable page with no function words trips it), so a hit is
treated as suspicion: the caller routes flagged pages to vision, which reads any
language and reports a page it truly cannot read.
"""

from __future__ import annotations


def _is_arabic(ch: str) -> bool:
    return "؀" <= ch <= "ۿ"


def is_garbage(text: str, cfg: dict) -> bool:
    """True when a page's extracted text looks unrecoverable and should go to vision.

    Near-blank pages are handled upstream by the char-count scanned check; the symbol
    check here still runs at any length, the word check only past word_check_min_chars.
    """
    symbol_ratio = cfg.get("symbol_ratio", 0.45)
    word_min_chars = cfg.get("word_check_min_chars", 80)
    min_hits = cfg.get("min_word_hits", 3)
    stop = cfg.get("stopwords", {})

    visible = [c for c in text if not c.isspace()]
    if not visible:
        return False

    # (1) symbol ratio — language-agnostic, catches symbol-junk at any length.
    symbols = sum(1 for c in visible if not c.isalnum())
    if symbols / len(visible) > symbol_ratio:
        return True

    # (2) common-word check — per-language, only when there is enough text to judge.
    if len(visible) < word_min_chars:
        return False
    arabic = sum(1 for c in text if _is_arabic(c))
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    if arabic > latin:
        return not any(w in text for w in stop.get("ar", []))
    latin_stop = {w for lang, words in stop.items() if lang != "ar" for w in words}
    words = "".join(c.lower() if c.isascii() else " " for c in text).split()
    return sum(1 for w in words if w in latin_stop) < min_hits
