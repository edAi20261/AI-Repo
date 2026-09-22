"""Whether a turn asks something new, or carries on the one before it.

A tutor is asked "what is evaporation?" once and "explain that more simply"
five times. The second kind names nothing: embed those words and the syllabus
returns whatever sits near them. Measured on a real textbook, "I didn't
understand, explain more simply" after a good answer about evaporation
retrieved five unrelated pages, and "can you give me an example?" retrieved
five more.

So the decision is made before retrieval, not inside it. A turn that continues
reuses the evidence the previous answer was built from; only a turn that opens
something new is searched.

The decision is deterministic. A classifier here would be the expensive way to
be wrong: measured on tutoring dialogue, a policy layer that diagnoses badly
scores *below* having no policy layer at all (Wang et al., NAACL 2024 — 0.26
with random decisions against 0.51 with none). Cheap signals that can be read
and argued with are worth more than a model that cannot say why.
"""

from __future__ import annotations

import enum
import re
from typing import Optional

from shared.matching import fold_for_match

# Study modes are actions taken on a lesson — set a quiz, mark this answer —
# rather than remarks about the last thing said. Explain is absent on purpose:
# it is the mode the page sits in by default, so choosing it says nothing about
# whether this message started something new.
OPENING_MODES = frozenset({"quiz", "check", "homework"})

# Words that stand in for something already said rather than naming a subject.
# A turn built only from these has nothing of its own to search for.
#
# This is a list, and a list is the kind of thing that rots. It is kept small
# and behavioural on purpose — pointing words, asking words, politeness — and
# its accuracy is measured rather than assumed (tests/unit/test_continuation.py).
# The safety of the rule does not rest on the list being complete: see
# relation_of for why being wrong in one direction costs nothing.
# Pointing at what was said.
_REFERENTIAL = (
    "it", "its", "this", "that", "these", "those", "them", "they", "there",
    "the", "a", "an", "previous", "last", "one", "same", "other", "another",
)
# Asking for the same thing again, differently.
_RE_PRESENTATION = (
    "more", "again", "simpler", "simply", "differently", "further", "better",
    "easier", "bit", "little", "explain", "understand", "understood", "know",
    "get", "got", "mean", "meant", "example", "examples", "tell", "say",
    "show", "give", "repeat", "rephrase", "elaborate",
)
# The shape of a question, carrying none of its subject.
_INTERROGATIVE = (
    "why", "how", "what", "when", "where", "who", "which",
    "do", "does", "did", "doesnt", "dont", "didnt",
    "can", "cant", "could", "couldnt", "would", "wouldnt",
    "is", "isnt", "are", "arent", "was", "wasnt",
    "will", "wont", "should", "shouldnt",
)
# Everything a sentence needs and no subject lives in.
_FUNCTION = (
    "please", "sorry", "ok", "okay", "thanks", "yes", "no", "not", "but",
    "and", "so", "i", "you", "me", "my", "your", "we", "us", "our",
)
_ARABIC = (
    "ده", "دي", "دا", "هذا", "هذه", "ذلك", "تلك", "هو", "هي", "هم", "هن",
    "اكثر", "أكثر", "اكتر", "تاني", "ثاني", "مرة", "كمان", "زياده", "زيادة",
    "افهم", "أفهم", "فهمت", "مفهوم", "لم", "لا", "مش", "يعني",
    "ازاي", "إزاي", "ليه", "لماذا", "كيف", "ماذا", "ما", "من", "متى",
    "اين", "أين", "مثال", "امثلة", "أمثلة", "اشرح", "إشرح", "وضح", "بسط",
    "ابسط", "أبسط", "لي", "لك", "له", "لها", "على", "في", "عن",
)

# A turn built only from these has nothing of its own to search for.
#
# This is a list, and a list is the kind of thing that rots. It is kept small
# and behavioural on purpose, and its accuracy is measured rather than assumed
# (tests/unit/test_continuation.py). The safety of the rule does not rest on
# the list being complete: see relation_of for why being wrong in one
# direction costs nothing.
_POINTERS = frozenset(
    _REFERENTIAL + _RE_PRESENTATION + _INTERROGATIVE + _FUNCTION + _ARABIC)

# Words that can only be pointing at something already said. One of these with
# no subject beside it is the clearest signal a turn has of being a follow-up.
_ANAPHORS = frozenset(
    ("it", "its", "this", "that", "these", "those", "them", "they",
     "ده", "دي", "دا", "هذا", "هذه", "ذلك", "تلك", "هو", "هي"))

# A turn carrying an anaphor may still name a subject — "what about tornadoes?"
# does. More than this many subject words and it is opening something new.
MAX_WORDS_BESIDE_AN_ANAPHOR = 1

_CONTRACTION = re.compile(r"['\u2019]")

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

# Below this a turn is too short to be naming a new subject on its own.
MAX_CONTINUATION_WORDS = 12


class Relation(enum.Enum):
    """What this turn does to the conversation."""

    OPENS = "opens"          # asks something new; search for it
    CONTINUES = "continues"  # carries on the last answer; reuse its evidence


def _tokens(message: str) -> list[str]:
    """The message as comparable words.

    Folded the way the index is folded, so a vocalised Arabic word and a bare
    one are the same word here as they are everywhere else, lowercased so that
    a sentence-initial word is not mistaken for a subject, and with
    apostrophes closed up so "didn't" is one token rather than two fragments.
    """
    return _WORD.findall(_CONTRACTION.sub("", fold_for_match(message).lower()))


def content_words(message: str) -> list[str]:
    """The words that could name a subject worth searching for."""
    return [w for w in _tokens(message)
            if w not in _POINTERS and len(w) > 2]


def _has_anaphor(tokens: list[str]) -> bool:
    return any(t in _ANAPHORS for t in tokens)


def relation_of(
    message: str,
    *,
    answered_before: bool,
    mode: Optional[str] = None,
) -> Relation:
    """What to do with this turn, decided before anything is searched.

    Read in order, cheapest and most certain first: there is nothing to
    continue on the first turn; an explicit study mode is an action on a
    lesson; a long turn is saying too much to be a follow-up; and otherwise a
    turn carries on the last one when it names no subject, or points at one it
    does not name.

    The two ways of being wrong do not cost the same, and the rule leans on
    that. Calling a follow-up OPENS is what the service does today — it
    searches, which is the behaviour being replaced, so nothing regresses.
    Calling a new question CONTINUES answers it from the previous question's
    evidence, which is a wrong answer delivered confidently. So every
    uncertain turn resolves to OPENS.
    """
    if not answered_before:
        return Relation.OPENS
    if mode in OPENING_MODES:
        return Relation.OPENS

    tokens = _tokens(message)
    if len(tokens) > MAX_CONTINUATION_WORDS:
        return Relation.OPENS

    subjects = content_words(message)
    if not subjects:
        return Relation.CONTINUES
    if _has_anaphor(tokens) and len(subjects) <= MAX_WORDS_BESIDE_AN_ANAPHOR:
        return Relation.CONTINUES
    return Relation.OPENS


def carried_query(previous: str, message: str) -> str:
    """What to ask the answering model, for a turn that continues another.

    A follow-up names its subject nowhere. The planner turns "can you give me
    an example?" into the standalone query "example", and a model asked about
    "example" over a page about evaporation answers that the context is
    insufficient to explain the concept of example — which is true, and
    useless.

    So the subject is carried from the turn before and the student's own words
    are kept beside it, rather than replaced by a paraphrase of them. Frameworks
    that hand the generator only the rewritten query lose the thing that says
    what kind of answer is wanted: simpler, shorter, an example, a reason.
    """
    previous = (previous or "").strip()
    message = (message or "").strip()
    if not previous:
        return message
    if not message:
        return previous
    return f"{previous}\n\nThe student has now said: {message}"
