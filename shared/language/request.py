"""Reading an explicit language request out of the message itself.

"اشرح بالعربي" and "answer in French" are instructions, not evidence to weigh:
they outrank every other signal. Matching is deterministic so it costs nothing
and cannot drift, and every pattern requires a directive verb next to the
language name — otherwise "the French Revolution" would switch the whole turn.
"""

from __future__ import annotations

import re
import unicodedata
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
    r"|switch|change|continue|réponds|repond|reponds|écris|ecris|antworte|schreibe"
    r"|erkläre|erklare)"
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

# "stop replying in Arabic" asks for the opposite of what it names. The anchors
# include reply/respond/answer, so without this the escape hatch reads as a
# request for the very language the student is trying to get away from.
_NEGATION_WINDOW = 30

_NEGATION = re.compile(
    r"(?:\b(?:stop|don'?t|do not|doesn'?t|never|no longer|quit|cease|instead of"
    r"|rather than|arr[êe]te|cesse|au lieu de|nicht mehr|h[öo]r auf|statt)\b"
    r"|بلاش|توقف عن|مش عايز|مش عاوز|كفاية|بطل)",
    re.IGNORECASE,
)

# A negation only reaches the language name inside the same clause. Without the
# cut, "I don't like maths, explain in Arabic" would read as a refusal of Arabic.
_CLAUSE_BREAK = re.compile(r"[.,;:!?\n،؛؟]")

# "answer in the language I write in" — names no language, so the tables above
# cannot reach it. It revokes a standing instruction rather than giving one.
_FOLLOW_ME = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    (r"\b(?:answer|reply|respond|write|talk|speak)\b[^.!?\n]{0,20}?\bin\b\s+"
     r"(?:the\s+|my\s+|same\s+){0,3}language\b"
     r"(?:\s+(?:that\s+)?(?:i|I)\s+(?:write|use|type|speak|ask)\w*)?(?:\s+in\b)?"),
    (r"\b(?:match|mirror|follow|use)\b[^.!?\n]{0,16}?"
     r"\b(?:my|the same)\s+language\b"),
    r"\bsame\s+language\s+as\s+(?:me|my\s+\w+|i\s+\w+)",
    r"\bin\s+my\s+(?:own\s+)?language\b",
    r"(?:رد|جاوب|اكتب|تكلم)\s*(?:لي\s*)?(?:بنفس\s*(?:اللغة|لغة\s*\S+)|بلغتي)",
    r"بنفس\s*(?:اللغة|لغتي)",
    r"(?:r[ée]ponds?|[ée]cris)\s+(?:dans\s+)?(?:ma\s+langue|la\s+m[êe]me\s+langue)",
    r"(?:antworte|schreibe)\s+(?:in\s+)?(?:meiner\s+Sprache|derselben\s+Sprache)",
))

# Stripped before deciding a message is nothing but a language instruction.
_COURTESY = frozenset({
    "please", "pls", "plz", "kindly", "thanks", "thank", "you", "ok", "okay",
    "and", "also", "just", "now", "from", "on", "so", "hey", "hi",
    "من", "فضلك", "لو", "سمحت", "شكرا", "شكرًا", "من_الان", "الان", "بقى", "يا", "ريت",
    "merci", "svp", "stp", "s'il", "te", "plaît", "plait", "bitte", "danke",
})

_WORDS = re.compile(r"[^\W\d_]+", re.UNICODE)
_AR_DIACRITICS = re.compile(r"[\u064b-\u0652\u0670\u0640]")

_DETERMINERS = re.compile(r"\b(?:the|la|le|les|der|die|das)\s+\S+$", re.IGNORECASE)

# "how do you say X in Arabic" asks *about* Arabic; the answer belongs in the
# language the student asked the question in.
_ABOUT_THE_LANGUAGE = re.compile(
    r"\b(?:how (?:do|would) (?:you|i) say|how to say|what(?:'s| is) .{0,24} called)\b",
    re.IGNORECASE,
)
_NEXT_WORD = re.compile(r"[^\w]*([A-Za-zÀ-ɏ']+)")


def _negation_before(text: str, match: re.Match[str]) -> Optional[re.Match[str]]:
    """The negation governing a language name, if there is one.

    Looks back a short way, and only within the same clause: "stop replying in
    Arabic" negates it, "I don't like maths, explain in Arabic" does not.
    """
    start = max(0, match.start() - _NEGATION_WINDOW)
    window = text[start:match.start()]
    breaks = list(_CLAUSE_BREAK.finditer(window))
    if breaks:
        window = window[breaks[-1].end():]
    return _NEGATION.search(window)


def _is_request(text: str, match: re.Match[str]) -> bool:
    """Whether a matched language name is an instruction rather than a topic."""
    if _ABOUT_THE_LANGUAGE.search(text):
        return False
    if _negation_before(text, match):
        return False
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


def clear_request(text: str) -> Optional[str]:
    """The matched span when the message asks to stop using a fixed language.

    Two shapes, because students write both: refusing a language by name ("stop
    replying in Arabic", "بلاش ترد بالعربي"), and asking to be followed ("answer
    in the language I write in"). The second names no language at all, so the
    per-language tables above cannot reach it.
    """
    if not text:
        return None

    for pattern in _FOLLOW_ME:
        found = pattern.search(text)
        if found:
            return found.group(0).strip()

    for _code, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            negation = _negation_before(text, match)
            if negation is None:
                continue
            # Report the refusal, not just the language, so a log line reads as
            # the student wrote it.
            start = max(0, match.start() - _NEGATION_WINDOW)
            return text[start + negation.start():match.end()].strip()
    return None


def _fold(text: str) -> str:
    return _AR_DIACRITICS.sub("", unicodedata.normalize("NFKC", text)).lower()


def language_instruction_only(text: str) -> Optional[tuple[str, Optional[str]]]:
    """``(span, code)`` when the message is nothing but a language instruction.

    ``code`` is the language asked for, or ``None`` when the instruction is to
    stop using one. Returns ``None`` when anything else remains: "explain
    photosynthesis, reply in Arabic" is a question carrying an instruction, and
    still needs an answer.
    """
    if not text or not text.strip():
        return None

    cleared = clear_request(text)
    if cleared:
        span, code = cleared, None
    else:
        requested = explicit_request(text)
        if not requested:
            return None
        code, span = requested

    # A span reaching across a clause break has swallowed the question with it:
    # in "explain subtraction, reply in Arabic" the directive anchor is the word
    # "explain", so the match runs the whole length of the message.
    if _CLAUSE_BREAK.search(span):
        return None

    remainder = text.replace(span, " ", 1)
    words = [w for w in _WORDS.findall(_fold(remainder)) if w not in _COURTESY]
    return (span, code) if not words else None
