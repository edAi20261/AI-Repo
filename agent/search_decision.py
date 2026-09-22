"""Whether a turn needs the syllabus searched, and what to search it for.

A tutor is asked "what is evaporation?" once and "explain that more simply"
five times. The second kind names nothing: embed those words and the syllabus
returns whatever sits near them. Measured on a real textbook, "I didn't
understand, explain more simply" after a good answer about evaporation
retrieved five unrelated pages.

So the decision is made before retrieval. A turn that carries on reuses the
passages the previous answer was built from; only a turn that opens something
new is searched.

The decision was first written as word lists — pointing words, asking words,
politeness — and every failure it had was a phrasing the lists did not carry.
Fixing one added a word and left the next one broken, in every language at
once: "مثال من فضلك" failed because the Arabic list had no entry for asking
politely. The phrasings are unbounded, so the list was the wrong shape of
answer, not an incomplete one.

What is bounded is the *question*: does this message name something to look
for, or does it point at the last answer? A model reading the two messages can
say, in any language, without anything here enumerating that language's words.
This is the pattern behind the retrieval tool in dialogue systems that handle
follow-ups well (HKUDS/DeepTutor exposes search as a tool and lets the model
decide when to reach for it) — the model chooses *when* to search, and never
chooses what it is allowed to see. Scope, score floor and lecture filter are
applied to the result no matter which way this decision goes.

It decides; it does not rewrite. Asked also to restate a carrying turn as a
standalone question, the same model answered "give me an example of the
previous concept" — a question with its subject taken out, which the tutor
could not answer while holding the very passages that contained it. So the
carrying question is composed here, from the question that was really asked
and the student's own words, and only an opening turn is the model's to
write.

Being wrong in the two directions does not cost the same, and the fallbacks
lean on that. Searching when the turn was a follow-up is what the service did
before any of this existed, so it regresses nothing. Reusing evidence when the
turn asked something new answers a new question from stale passages, which is
a wrong answer delivered with citations. So every failure here resolves to
searching.
"""

from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, Field

from shared.config import Config
from shared.structured import StructuredOutputError, call_structured

logger = logging.getLogger(__name__)

TASK = "search_decision"

# Study modes are actions taken on a lesson — set a quiz, mark this answer —
# rather than remarks about the last thing said, and they arrive with the
# lesson already named. Explain is absent on purpose: it is the mode the page
# sits in by default, so choosing it says nothing about whether this message
# started something new.
OPENING_MODES = frozenset({"quiz", "check", "homework"})

SYSTEM = """You prepare a student's message for a search over their textbook.

You are given the question the tutor answered immediately before, and what the
student has just said. Decide one thing: does the new message name a subject to
look up, or does it point back at the answer already on the table?

It opens something new when it names a subject of its own, even briefly, and
even while referring to the last answer.

It carries on the previous answer when it asks for that same answer again in
another form — shorter, simpler, with an example, with a reason, in steps — or
when it reacts to it without naming anything else. A student who is lost often
writes at length and still names nothing; length does not make a message new.

If it opens something new, also write it out as a complete question that names
its own subject, in the language the student used, adding nothing the student
did not raise. If it carries on, leave that empty: the question is built from
the previous one and is not yours to write."""


class SearchDecision(BaseModel):
    """What to do with a turn, decided before anything is searched."""

    opens_new_subject: bool = Field(
        description="true when the message names a subject to look up; "
                    "false when it asks for the previous answer in another form")
    question: str = Field(
        default="",
        description="only when opening something new: that message written as "
                    "a complete question, in the student's own language")


def _names_its_own_subject(decision: SearchDecision) -> None:
    """A turn that opens something new must say what it is about.

    Left to itself the model answers this field with a pointer — "give me an
    example of the previous concept" — which is a question only a reader who
    already knows the answer can use. The tutor, asked that over the right
    passages, replied that the context contained no previous concept.
    """
    if decision.opens_new_subject and not decision.question.strip():
        raise ValueError(
            "opens_new_subject is true, so question must be that message "
            "written as a complete question naming its own subject")


def carried_question(previous: str, message: str) -> str:
    """What to ask about a turn that carries the previous one on.

    Not written by a model. Asked to restate a question it cannot see the
    answer to, a model reaches for a pointer — "the previous concept", "the
    last thing we talked about" — and hands the tutor a question with its
    subject removed, which is the failure this whole module exists to stop.

    Composing it is exact instead: the subject comes from the question that
    was actually asked, and the student's own words stay beside it rather than
    being replaced by a paraphrase. Frameworks that hand the generator only a
    rewritten query lose the part that says what kind of answer is wanted —
    simpler, shorter, an example, a reason.
    """
    previous, message = (previous or "").strip(), (message or "").strip()
    if not previous:
        return message
    if not message:
        return previous
    return f"{previous}\n\nThe student has now said: {message}"


def decide(
    message: str,
    *,
    previous_question: str,
    config: Config,
    mode: Optional[str] = None,
    llm=None,
) -> SearchDecision:
    """Whether to search for ``message``, and what to search for.

    ``previous_question`` empty means nothing has been answered in this session
    yet, so there is no previous answer to carry on and nothing to ask a model
    about. An explicit study mode is an action on a lesson, which always opens.
    """
    message = (message or "").strip()
    previous_question = (previous_question or "").strip()

    if not previous_question or not message:
        return SearchDecision(opens_new_subject=True, question=message)
    if mode in OPENING_MODES:
        return SearchDecision(opens_new_subject=True, question=message)

    try:
        decision = call_structured(
            SearchDecision,
            [("system", SYSTEM),
             ("human", f"The tutor has just answered this question:\n"
                       f"{previous_question}\n\n"
                       f"The student now says:\n{message}")],
            task=TASK,
            config=config,
            llm=llm,
            validate=_names_its_own_subject,
        )
    except StructuredOutputError as exc:
        logger.warning("could not decide whether this turn follows the last; "
                       "searching", extra={"error": str(exc)[:200]})
        return SearchDecision(opens_new_subject=True, question=message)

    if not decision.opens_new_subject:
        decision.question = carried_question(previous_question, message)
    logger.info("turn %s", "opens a new subject" if decision.opens_new_subject
                else "carries on the last answer",
                extra={"question": decision.question[:120]})
    return decision
