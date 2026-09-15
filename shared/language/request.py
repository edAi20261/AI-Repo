"""Reading an explicit language request out of the message itself.

"اشرح بالعربي" and "answer in French" are instructions, not evidence to weigh:
they outrank every other signal. Matching is deterministic so it costs nothing
and cannot drift, and every pattern requires a directive verb next to the
language name — otherwise "the French Revolution" would switch the whole turn.
"""

from __future__ import annotations

import re
from typing import Optional

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ar", re.compile(
        r"(?:بال|با|ب)\s?(?:عربي|العربيه|العربية)"
        r"|(?:اكتب|اشرح|جاوب|رد|تكلم|احك)\s+(?:لي\s+)?(?:بال)?(?:عربي|العربية)",
    )),
    ("en", re.compile(
        r"(?:بال|با|ب)\s?(?:انجليزي|إنجليزي|الانجليزية|الإنجليزية)"
        r"|(?:اكتب|اشرح|جاوب|رد)\s+(?:لي\s+)?(?:بال)?(?:انجليزي|إنجليزي)",
    )),
    ("fr", re.compile(r"(?:بال|با|ب)\s?(?:فرنسي|الفرنسية)")),
    ("de", re.compile(r"(?:بال|با|ب)\s?(?:الماني|ألماني|الألمانية)")),
]

_LATIN_NAMES = {
    "ar": r"arabic|arabe|arabisch|arabi",
    "en": r"english|anglais|englisch",
    "fr": r"french|français|francais|französisch|franzosisch",
    "de": r"german|allemand|deutsch",
}

# A language name only redirects the answer when a directive verb sits beside it;
# "English literature" and "the French Revolution" are subjects, not requests.
_LATIN_ANCHOR = (
    r"(?:in|into|en|auf|reply|respond|answer|write|explain|translate|speak|say|use"
    r"|réponds|repond|reponds|écris|ecris|antworte|schreibe|erkläre|erklare)"
)

for _code, _names in _LATIN_NAMES.items():
    _PATTERNS.append((
        _code,
        re.compile(rf"\b{_LATIN_ANCHOR}\b[^.!?\n]{{0,24}}?\b(?:{_names})\b", re.IGNORECASE),
    ))


# A language name followed by one of these is naming a subject, not asking for a
# reply: "my English literature homework", "the French Revolution".
_SUBJECT_NOUNS = frozenset({
    "literature", "homework", "class", "classes", "lesson", "lessons", "course",
    "exam", "exams", "test", "teacher", "book", "books", "grammar", "vocabulary",
    "revolution", "history", "culture", "empire", "club", "translation", "dictionary",
})

_DETERMINERS = re.compile(r"\b(?:the|la|le|les|der|die|das)\s+\S+$", re.IGNORECASE)
_NEXT_WORD = re.compile(r"[^\w]*([A-Za-zÀ-ɏ']+)")


def _is_request(text: str, match: re.Match[str]) -> bool:
    """Whether a matched language name is an instruction rather than a topic."""
    following = _NEXT_WORD.match(text[match.end():])
    if following is None:
        return True

    word = following.group(1)
    if word.lower() in _SUBJECT_NOUNS:
        return False
    # "in the French Revolution" — a determiner inside the span plus a proper noun
    # after it is a title, while "in Arabic Newton's law" is still a request.
    return not (word[:1].isupper() and _DETERMINERS.search(match.group(0)))


def explicit_request(text: str) -> Optional[tuple[str, str]]:
    """The language the message asks to be answered in, with the matched span."""
    if not text:
        return None

    best: Optional[tuple[int, str, str]] = None
    for code, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            if not _is_request(text, match):
                continue
            if best is None or match.start() > best[0]:
                best = (match.start(), code, match.group(0).strip())
    if best is None:
        return None
    return best[1], best[2]
