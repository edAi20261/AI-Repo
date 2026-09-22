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

# [1] | [1, 2] | [1,2] | [1، 2] — a group of numbers inside one pair of brackets,
# separated the way the answer's own script separates a list.
SEPARATOR = "[,،、，]"
MARKER = re.compile(rf"\[\s*(\d+(?:\s*{SEPARATOR}\s*\d+)*)\s*\]")

# A run of markers with nothing between them: "[1][2][1]".
RUN = re.compile(r"(?:\[\d+\]){2,}")


def cited_numbers(text: str) -> list[int]:
    """Every source number referenced in ``text``, in order of first appearance."""
    seen: list[int] = []
    for group in MARKER.findall(text or ""):
        for part in re.split(SEPARATOR, group):
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
        numbers = [int(part.strip()) for part in re.split(SEPARATOR, match.group(1))]
        kept = [mapping[n] for n in numbers if n in mapping]
        dropped = dropped or len(kept) < len(numbers)
        return "".join(f"[{n}]" for n in kept)

    rewritten = _collapse_runs(MARKER.sub(replace, text or ""))
    return _tidy(rewritten) if dropped else rewritten


def _collapse_runs(text: str) -> str:
    """Drop repeats inside a run of adjacent markers.

    A model that leans on the same two passages all the way through an answer
    writes them after every sentence, and once grouped markers are split one per
    source that reads as "[1][2][1]". The same source twice in one breath tells
    the student nothing; keep the first of each, in order.
    """
    def collapse(match: re.Match[str]) -> str:
        seen: list[str] = []
        for number in re.findall(r"\[(\d+)\]", match.group(0)):
            if number not in seen:
                seen.append(number)
        return "".join(f"[{n}]" for n in seen)

    return RUN.sub(collapse, text)


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
