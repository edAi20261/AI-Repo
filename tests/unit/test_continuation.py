"""A follow-up is not a new question, and must not be searched like one.

A student who says "I didn't understand" is not asking about understanding.
Embedding those words and searching the syllabus with them retrieves whatever
happens to be near them — measured on a real textbook, a question about
evaporation followed by "explain more simply" returned five unrelated pages.

The turn that continues one already answered reuses what that answer was built
from. Nothing is searched, because nothing new was asked.
"""

from __future__ import annotations

import pytest

from agent.continuation import Relation, relation_of


def _opens(message: str, **kw) -> bool:
    return relation_of(message, **kw) is Relation.OPENS


# --- a turn that asks something new -----------------------------------------

@pytest.mark.parametrize("message", [
    "What is evaporation?",
    "How do hurricanes form?",
    "ما هي دورة الماء؟",
    "What about tornadoes?",          # a new subject, however casually asked
    "Tell me about watersheds instead",
])
def test_a_question_with_a_subject_of_its_own_opens_a_new_turn(message):
    assert _opens(message, answered_before=True)


def test_the_first_turn_always_opens():
    """There is nothing to continue. This is the one case no signal can get
    wrong, so it is decided before any other is consulted."""
    assert _opens("explain more", answered_before=False)
    assert _opens("", answered_before=False)


# --- a turn that continues the one before -----------------------------------

@pytest.mark.parametrize("message", [
    "I didn't understand. Can you explain more simply?",
    "explain more",
    "tell me more",
    "Can you give me an example?",
    "But why does that happen?",
    "why?",
    "لم أفهم، اشرح لي أكثر",
    "simpler please",
    "what do you mean?",
])
def test_a_turn_that_names_nothing_continues_the_one_before(message):
    assert not _opens(message, answered_before=True)


def test_a_continuation_needs_something_to_continue():
    """The same words, with no answer behind them, are a question like any
    other — and get searched like one rather than reusing nothing."""
    assert _opens("explain more", answered_before=False)
    assert not _opens("explain more", answered_before=True)


# --- the student said which mode they are in --------------------------------

def test_pressing_a_study_mode_button_opens_a_turn():
    """The modes are actions on a lesson, not remarks about the last answer.
    An explicit choice outranks anything read off the words."""
    assert _opens("give me a quiz", answered_before=True, mode="quiz")
    assert _opens("check this", answered_before=True, mode="check")


def test_an_explicit_mode_does_not_rescue_a_bare_follow_up():
    """Explain mode is the default the page sits in, so it says nothing about
    whether this particular message started something new."""
    assert not _opens("I didn't understand", answered_before=True, mode="explain")


# --- measured, not assumed --------------------------------------------------

# Turns a student actually types, labelled by what the tutor should do. The
# continuations are drawn from the failing conversations that prompted this
# work; the openers include the ones most likely to be mistaken for them.
LABELLED: list[tuple[str, Relation]] = [
    # continues — nothing named, or pointing at what was just said
    ("I didn't understand. Can you explain more simply?", Relation.CONTINUES),
    ("explain more", Relation.CONTINUES),
    ("tell me more", Relation.CONTINUES),
    ("more please", Relation.CONTINUES),
    ("Can you give me an example?", Relation.CONTINUES),
    ("give me another example", Relation.CONTINUES),
    ("But why does that happen?", Relation.CONTINUES),
    ("why?", Relation.CONTINUES),
    ("what do you mean?", Relation.CONTINUES),
    ("I still don't get it", Relation.CONTINUES),
    ("can you say that differently", Relation.CONTINUES),
    ("simpler please", Relation.CONTINUES),
    ("لم أفهم، اشرح لي أكثر", Relation.CONTINUES),
    ("مثال تاني", Relation.CONTINUES),
    ("ليه؟", Relation.CONTINUES),
    # opens — names a subject, however casually
    ("What is evaporation?", Relation.OPENS),
    ("How do hurricanes form?", Relation.OPENS),
    ("What about tornadoes?", Relation.OPENS),
    ("what is a watershed", Relation.OPENS),
    ("Now explain condensation", Relation.OPENS),
    ("ok but what causes erosion", Relation.OPENS),
    ("Who was Napoleon?", Relation.OPENS),
    ("ما هي دورة الماء؟", Relation.OPENS),
    ("اشرح لي قصيدة الشباب", Relation.OPENS),
    ("Tell me about the atmosphere instead", Relation.OPENS),
    ("I didn't understand the water cycle, explain it", Relation.OPENS),
]


def test_the_rule_is_right_often_enough_to_ship():
    """Measured over turns a student actually types. The bar is not perfection
    — it is that mistakes fall on the safe side."""
    wrong = [(m, want, relation_of(m, answered_before=True))
             for m, want in LABELLED
             if relation_of(m, answered_before=True) is not want]
    unsafe = [w for w in wrong if w[1] is Relation.OPENS]
    assert not unsafe, f"a new question treated as a follow-up: {unsafe}"
    assert len(wrong) <= 2, f"too many misread: {wrong}"
