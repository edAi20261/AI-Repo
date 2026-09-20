"""A prompt says who reads its output and what language that output is in.

Four times now the same bug shipped: a prompt that generates text a student
reads, with nothing saying what language it must come back in. The topic
extractor defaulted to English on Arabic lessons, the maths figure builder was
pinned to English outright, the planner said nothing at all, and the slides
builder wrapped a directive inside a directive and asked the model to obey
"Write the content in Write the content in Arabic".

They were four samples from one hole. A prompt was a bare ``str``, so nothing
distinguished "a student reads this" from "this is internal JSON" — a page
transcriber and a maths figure builder look identical to a reviewer, but one must
never carry a reply-language directive and the other must always carry one. The
judgement lived in whoever wrote the call site.

Here it lives in the prompt. Declaring the audience and the language rule is what
constructing one costs, the rule renders the directive itself, and a call site
passes a language *code* — so pinning the wrong language, passing a name, passing
a directive twice, and forgetting entirely all stop being possible.

``tests/unit/test_prompt_contract.py`` walks every prompt in both packages and
fails the build on a fifth instance.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .language.directive import content_directive, language_directive
from .language.types import Lang


class Audience(Enum):
    """Who reads what comes back."""

    STUDENT = "student"    # prose, labels or captions a learner will read
    INTERNAL = "internal"  # JSON the service consumes and never displays


class LanguageRule(Enum):
    """What decides the language of the output."""

    REPLY = "reply"
    """Addressed to the student: their language, whatever the sources are in."""

    CONTENT = "content"
    """Generated material — quiz items, labels, captions — in a named language."""

    TRANSCRIBE = "transcribe"
    """Copied from a page: the language already printed there. Never translated,
    so a reply directive here would be actively wrong."""

    NONE = "none"
    """No natural-language output to govern — codes, ids, numbers."""


_RENDERERS = {
    LanguageRule.REPLY: language_directive,
    LanguageRule.CONTENT: content_directive,
}


@dataclass(frozen=True)
class Prompt:
    """A template that knows who reads it and how its language is decided."""

    name: str
    template: str
    audience: Audience
    language_rule: LanguageRule
    slot: str = "language"

    def __post_init__(self) -> None:
        if self.audience is Audience.STUDENT and self.language_rule is LanguageRule.NONE:
            raise ValueError(
                f"{self.name}: a student reads this, so its language has to be "
                "decided by something. Pick REPLY, CONTENT or TRANSCRIBE.")
        if self.language_rule in _RENDERERS and self.slot not in self.placeholders():
            raise ValueError(
                f"{self.name}: declares a {self.language_rule.value} language rule "
                f"but has no {{{self.slot}}} for the directive to go in.")

    @property
    def fingerprint(self) -> str:
        """A short digest of this prompt's text.

        What a cache of model output keys on. A hand-maintained version string
        is a step someone forgets, and the cache then serves pages the previous
        prompt produced — which is the failure a cache exists to be trusted not
        to have. Changing a word here invalidates the cache on its own.
        """
        import hashlib

        return hashlib.sha256(self.template.encode()).hexdigest()[:12]

    def placeholders(self) -> set[str]:
        """Every ``{name}`` the template expects to be filled."""
        return {field for _, field, _, _ in string.Formatter().parse(self.template)
                if field}

    def render(self, *, language: Lang | None = None, **values: Any) -> str:
        """The prompt text, with the language directive supplied by the rule.

        The caller passes a language *code*. It never builds the directive, which
        is what stops one being pinned, named, or wrapped twice.
        """
        renderer = _RENDERERS.get(self.language_rule)
        if renderer is not None:
            if language is None:
                raise ValueError(
                    f"{self.name}: needs the language its output is written in.")
            values[self.slot] = renderer(language)
        elif language is not None:
            raise ValueError(
                f"{self.name}: its language is not ours to choose "
                f"({self.language_rule.value}).")

        missing = self.placeholders() - set(values)
        if missing:
            # A KeyError here would surface as a degraded answer and nothing else,
            # so name the gap while there is still something to name it to.
            raise ValueError(f"{self.name}: nothing supplied for {sorted(missing)}")
        return self.template.format(**values)

    def __str__(self) -> str:  # pragma: no cover - debugging convenience
        return f"<Prompt {self.name} {self.audience.value}/{self.language_rule.value}>"
