"""Approximate token counting.

Two budgets are set in tokens and neither is exact: the ~500-800 tokens a chunk
is sized to, and the slice of a conversation a prompt carries. Coupling either
to a model tokenizer buys precision nothing spends, and would make the size of a
chunk depend on which model happened to be configured.

Counts word-ish units and punctuation, scaled by a subword factor. Measured
against the 93 stored conversation turns — Arabic, English and French — this
reads 3.47 characters to the token.
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


_ELLIPSIS = " …"


def clip_to_tokens(text: str, budget: int) -> str:
    """``text`` cut down until it fits ``budget`` tokens.

    Binary search rather than a guessed character ratio: the ratio is 3.5 for
    English and nearer 2 for Arabic, and a budget that silently means half as
    much in one language as another is the kind of thing nobody notices until a
    student's own question has been cut out of their history.

    The bound is the estimator's, so the mark left where the text was cut can
    carry it a token past ``budget``. Every caller here is sizing a prompt, and
    none of them is within one token of anything.
    """
    if budget <= 0:
        return ""
    if count_tokens(text) <= budget:
        return text
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if count_tokens(text[:middle]) <= budget:
            low = middle
        else:
            high = middle - 1
    return text[:low].rstrip() + _ELLIPSIS
