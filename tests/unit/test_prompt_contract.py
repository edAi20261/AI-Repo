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

from shared.prompting import Audience, LanguageRule, Prompt

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
    if prompt.language_rule in (LanguageRule.REPLY, LanguageRule.CONTENT):
        assert prompt.slot in prompt.placeholders(), (
            f"{name} declares a language rule with no {{{prompt.slot}}} slot")


@pytest.mark.parametrize("name,prompt", _every_prompt())
def test_every_placeholder_can_be_filled(name, prompt):
    """A `{placeholder}` with nothing to fill it is a KeyError at answer time,
    and the degrade path would turn it into "something went wrong" forever."""
    directed = prompt.language_rule in (LanguageRule.REPLY, LanguageRule.CONTENT)
    # A directed prompt has its slot filled by the rule; for the others the slot
    # is an ordinary placeholder the caller supplies.
    values = {slot: "x" for slot in prompt.placeholders()
              if not (directed and slot == prompt.slot)}
    rendered = prompt.render(language="ar" if directed else None, **values)
    assert rendered.strip()


@pytest.mark.parametrize("name,prompt", [
    (n, p) for n, p in _every_prompt()
    if p.language_rule in (LanguageRule.REPLY, LanguageRule.CONTENT)
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
    if prompt.language_rule not in (LanguageRule.REPLY, LanguageRule.CONTENT):
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
