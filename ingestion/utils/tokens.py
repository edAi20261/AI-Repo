"""Approximate token counting for chunk sizing.

The ~500-800 token budget is soft, so an exact model tokenizer is not worth
coupling to. Counts word-ish units and punctuation, scaled by a subword factor.
"""

from __future__ import annotations

import re

# A "word" run (Unicode letters/digits, so Arabic counts) or a lone symbol.
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# Subword tokenizers split longer words, so raw word counts undercount slightly.
_SUBWORD_FACTOR = 1.3


def count_tokens(text: str) -> int:
    """Estimated token count for ``text``."""
    if not text:
        return 0
    units = len(_TOKEN_RE.findall(text))
    return int(round(units * _SUBWORD_FACTOR))
