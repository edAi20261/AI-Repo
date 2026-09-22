"""Whether a turn is searched, and what happens when that cannot be decided.

Whether the decision is *right* on real phrasings is not something a fake model
can answer; that is measured over whole conversations against a live model in
scripts/eval_conversations.py. What is tested here is everything around it: the
turns that never reach a model at all, and — the reason this module can be
trusted at all — that every way of failing resolves to searching.
"""

from __future__ import annotations

import pytest

from agent import search_decision
from agent.search_decision import SearchDecision, decide
from shared.config import get_config
from shared.structured import StructuredOutputError


def _never_called(*a, **k):
    raise AssertionError("no model call should have been made")


def test_the_first_turn_of_a_session_opens_without_asking_anyone(monkeypatch):
    """Nothing has been answered yet, so there is no previous answer to carry
    on — and no call worth paying for on the first turn of every session."""
    monkeypatch.setattr(search_decision, "call_structured", _never_called)

    out = decide("What is evaporation?", previous_question="", config=get_config())

    assert out.opens_new_subject
    assert out.question == "What is evaporation?"


@pytest.mark.parametrize("mode", ["quiz", "check", "homework"])
def test_choosing_a_study_mode_opens_whatever_the_words_are(monkeypatch, mode):
    """Set me a quiz, mark this answer — an action on a lesson, arriving with
    the lesson already named. Explain is absent from that set on purpose: it is
    the mode the page sits in by default."""
    monkeypatch.setattr(search_decision, "call_structured", _never_called)

    out = decide("more", previous_question="What is evaporation?",
                 config=get_config(), mode=mode)

    assert out.opens_new_subject


def test_a_model_that_cannot_answer_leaves_the_turn_searching(monkeypatch):
    """The two ways of being wrong do not cost the same. Searching a follow-up
    is what the service did before any of this existed, so it regresses
    nothing. Reusing evidence for a new question answers it from stale
    passages, with citations, confidently. So failure resolves to searching."""
    def _fails(*a, **k):
        raise StructuredOutputError("the provider refused",
                                    task=search_decision.TASK, label="SearchDecision")

    monkeypatch.setattr(search_decision, "call_structured", _fails)

    out = decide("explain more simply", previous_question="What is evaporation?",
                 config=get_config())

    assert out.opens_new_subject, "an undecided turn must search"
    assert out.question == "explain more simply"


def test_an_opening_turn_with_no_subject_in_it_is_sent_back():
    """A blank question searches for nothing. call_structured hands the reason
    back to the model and retries; what matters here is that this validator is
    what makes it do so."""
    with pytest.raises(ValueError, match="complete question"):
        search_decision._names_its_own_subject(
            SearchDecision(opens_new_subject=True, question="   "))


def test_a_carrying_turn_is_not_asked_to_write_its_own_question():
    """The model, asked to restate a question it cannot see the answer to,
    reaches for a pointer. "Give me an example." came back as "Give me an
    example of the previous concept.", and the tutor — holding exactly the
    right passages — replied that the context contained no previous concept."""
    search_decision._names_its_own_subject(
        SearchDecision(opens_new_subject=False, question=""))


def test_a_carried_question_keeps_the_subject_and_what_was_asked_for():
    """Both halves matter. Without the subject the tutor answers about
    nothing; without the student's words it answers the old question again
    and ignores that simpler, shorter or an example was what was wanted."""
    out = search_decision.carried_question("What is evaporation?",
                                           "Give me an example.")

    assert "evaporation" in out.lower()
    assert "Give me an example." in out


def test_the_prompt_names_no_language_and_no_language_s_words(monkeypatch):
    """The word lists this replaced had an Arabic section, which is why
    "مثال من فضلك" — an example, please — was read as a new subject and
    searched for. A rule written in one language's words is a rule that works
    in one language."""
    letters = [c for c in search_decision.SYSTEM if c.isalpha()]

    assert all(c.isascii() for c in letters), (
        "a word from one language in this prompt is a rule that works in one "
        "language")
    for named in ("Arabic", "English", "French", "German"):
        assert named not in search_decision.SYSTEM
