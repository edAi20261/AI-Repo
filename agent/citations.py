"""Reading the citation markers a model actually writes.

The answer prompts ask for ``[n]``; models write ``[1, 2]`` when a sentence rests
on two passages. Our marker pattern only matched a single number, so grounding
detection failed on almost every real answer: citations were dropped, the sources
block never rendered, and the student was left with bracketed numbers pointing at
nothing.

Parsing is deliberately liberal and emission is strict — accept every form, then
rewrite the answer to one marker per source.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# [1] | [1, 2] | [1,2] — a group of numbers inside one pair of brackets.
MARKER = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")


def cited_numbers(text: str) -> list[int]:
    """Every source number referenced in ``text``, in order of first appearance."""
    seen: list[int] = []
    for group in MARKER.findall(text or ""):
        for part in group.split(","):
            number = int(part.strip())
            if number not in seen:
                seen.append(number)
    return seen


def has_citation(text: str) -> bool:
    """Whether the answer references any source at all."""
    return bool(cited_numbers(text))


def renumber(text: str, mapping: dict[int, int]) -> str:
    """Rewrite markers through ``mapping``, one bracket per source.

    ``[1, 2]`` becomes ``[3][4]`` rather than ``[3, 4]``: a renderer can then turn
    each marker into its own badge without parsing lists, and a marker whose source
    was dropped disappears instead of pointing at the wrong lesson.
    """
    dropped = False

    def replace(match: re.Match[str]) -> str:
        nonlocal dropped
        numbers = [int(part.strip()) for part in match.group(1).split(",")]
        kept = [mapping[n] for n in numbers if n in mapping]
        dropped = dropped or len(kept) < len(numbers)
        return "".join(f"[{n}]" for n in kept)

    rewritten = MARKER.sub(replace, text or "")
    return _tidy(rewritten) if dropped else rewritten


def strip_markers(text: str) -> str:
    """The answer with every citation marker removed."""
    return _tidy(MARKER.sub("", text or ""))


def _tidy(text: str) -> str:
    """Close the gap a removed marker leaves before punctuation."""
    return re.sub(r"[ \t]+([.,;:!?])", r"\1", text).strip()


def keep_cited(citations: Iterable, numbers: Iterable[int]) -> list:
    """Only the citations the answer actually referenced."""
    wanted = set(numbers)
    return [c for c in citations if getattr(c, "n", None) in wanted]
