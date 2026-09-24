"""Every prompt declares who reads it and what language it comes back in.

Four times the same bug shipped, and each time it was caught by a person reading
an answer in a language they happened to understand: the topic extractor
defaulting to English on Arabic lessons, the maths figure builder pinned to
English, the planner declaring nothing, and the slides builder wrapping a
directive inside a directive.

They were one hole, not four bugs. A prompt was a bare string, so nothing
separated "a student reads this" from "this is internal JSON", and no test
inspected a prompt at all.

This is the sweep that closes it — the same shape as
``test_every_string_exists_in_every_supported_language``, which has kept the
translation catalogue complete since the day it was written. Adding a
student-facing prompt with no language rule now fails the build instead of
reaching a student.
"""

from __future__ import annotations

import pkgutil
from importlib import import_module

import pytest

from shared.prompting import DIRECTED_RULES, Audience, LanguageRule, Prompt

SUPPORTED = ("ar", "en", "fr", "de")


def _every_prompt() -> list[tuple[str, Prompt]]:
    """Each declared prompt in the repo, wherever it lives."""
    modules = ["agent.prompts"]
    package = import_module("ingestion.prompts")
    modules += [f"ingestion.prompts.{m.name}"
                for m in pkgutil.iter_modules(package.__path__)]

    found: dict[str, Prompt] = {}
    for name in modules:
        module = import_module(name)
        for attribute in vars(module).values():
            if isinstance(attribute, Prompt):
                found[attribute.name] = attribute
    return sorted(found.items())


def test_the_repo_has_prompts_to_check():
    """A sweep that silently finds nothing protects nothing."""
    assert len(_every_prompt()) >= 15


@pytest.mark.parametrize("name,prompt", _every_prompt())
def test_a_prompt_a_student_reads_says_what_language_it_is_in(name, prompt):
    """The recurring bug, as a build failure."""
    if prompt.audience is not Audience.STUDENT:
        return
    assert prompt.language_rule is not LanguageRule.NONE, (
        f"{name} produces text a student reads but nothing decides its language")


@pytest.mark.parametrize("name,prompt", _every_prompt())
def test_a_language_rule_has_somewhere_to_put_the_directive(name, prompt):
    if prompt.language_rule in DIRECTED_RULES:
        assert prompt.slot in prompt.placeholders(), (
            f"{name} declares a language rule with no {{{prompt.slot}}} slot")


@pytest.mark.parametrize("name,prompt", _every_prompt())
def test_every_placeholder_can_be_filled(name, prompt):
    """A `{placeholder}` with nothing to fill it is a KeyError at answer time,
    and the degrade path would turn it into "something went wrong" forever."""
    directed = prompt.language_rule in DIRECTED_RULES
    # A directed prompt has its slot filled by the rule; for the others the slot
    # is an ordinary placeholder the caller supplies.
    values = {slot: "x" for slot in prompt.placeholders()
              if not (directed and slot == prompt.slot)}
    rendered = prompt.render(language="ar" if directed else None, **values)
    assert rendered.strip()


@pytest.mark.parametrize("name,prompt", [
    (n, p) for n, p in _every_prompt()
    if p.language_rule in DIRECTED_RULES
])
@pytest.mark.parametrize("language", SUPPORTED)
def test_the_directive_names_the_language_in_every_supported_one(name, prompt, language):
    """Rendering must work for all four, not just the one someone tested."""
    values = {slot: "x" for slot in prompt.placeholders() if slot != prompt.slot}
    assert prompt.render(language=language, **values)


@pytest.mark.parametrize("name,prompt", _every_prompt())
def test_a_caller_cannot_hand_over_a_directive_instead_of_a_language(name, prompt):
    """How "Write the content in Write the content in Arabic" happened: the
    parameter was a code, someone passed the rendered directive, and nothing
    objected because both are strings."""
    if prompt.language_rule not in DIRECTED_RULES:
        return
    values = {slot: "x" for slot in prompt.placeholders() if slot != prompt.slot}
    with pytest.raises((ValueError, KeyError)):
        prompt.render(language="Write the content in Arabic (العربية).", **values)


@pytest.mark.parametrize("name,prompt", _every_prompt())
def test_transcription_never_carries_a_reply_directive(name, prompt):
    """A page is copied, not answered. Telling the model to reply in the
    student's language would ask it to translate the textbook."""
    if prompt.language_rule is not LanguageRule.TRANSCRIBE:
        return
    with pytest.raises(ValueError):
        prompt.render(language="ar",
                      **{s: "x" for s in prompt.placeholders() if s != prompt.slot})


def test_an_answer_does_not_open_with_a_greeting():
    """Every answer opened with "أهلاً بك!" or "Bonjour! C'est une excellente
    question" — the model's own habit, because nothing told it otherwise. A
    student mid-lesson is greeted once, by the tutor's first reply, not on
    every explanation they ask for."""
    from agent.prompts import FORMATTING_CONTRACT

    contract = FORMATTING_CONTRACT.lower()
    assert "greeting" in contract
    assert "start with the answer" in contract


def test_an_answer_does_not_open_by_greeting_a_student_it_is_already_talking_to():
    """16 of 40 production replies opened with a greeting, one of them 'أهلاً بك
    مرة أخرى!' on the fifth reply of a conversation. The model had the whole
    conversation in its prompt and greeted anyway: nothing told it not to, and
    its Arabic and French registers open that way by default."""
    from agent.prompts import FORMATTING_CONTRACT, NO_MATERIAL_PROMPT

    for text in (FORMATTING_CONTRACT, NO_MATERIAL_PROMPT.template):
        assert "greeting" in text.lower(), text[:60]


def test_the_social_turn_may_still_greet():
    """A student who says hello is answered by the converse prompt, and that one
    must stay free to say hello back."""
    from agent.prompts import CONVERSE_PROMPT

    assert "greeting" not in CONVERSE_PROMPT.template.lower()


def test_a_taught_answer_carries_the_no_greeting_rule_and_small_talk_does_not():
    """Which prompts may greet is a property of the prompt, not of a line
    somebody remembered to add: an answer to a question may not, an answer to
    "hello" must."""
    from shared.prompting import LanguageRule
    from agent.prompts import (
        COMPOSE_PROMPT,
        CONVERSE_PROMPT,
        EXPLAIN_PROMPT,
        NO_MATERIAL_PROMPT,
        SUMMARIZE_PROMPT,
    )

    for prompt in (EXPLAIN_PROMPT, SUMMARIZE_PROMPT, COMPOSE_PROMPT, NO_MATERIAL_PROMPT):
        assert prompt.language_rule is LanguageRule.ANSWER, prompt.name
        assert "أهلا" in prompt.render_language("ar"), prompt.name

    assert CONVERSE_PROMPT.language_rule is LanguageRule.REPLY
    assert "أهلا" not in CONVERSE_PROMPT.render_language("ar")
