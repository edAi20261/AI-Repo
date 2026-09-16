"""LLM prompt: choose a topic's prerequisites from the subject catalog."""

from shared.prompting import Audience, LanguageRule, Prompt

DERIVE_DEPS_PROMPT = Prompt(
    name="DERIVE_DEPS_PROMPT",
    audience=Audience.INTERNAL,
    language_rule=LanguageRule.NONE,
    template="""\
You are given a target skill and a list of candidate skills from the same
subject. Choose the candidates that are genuine PREREQUISITES — skills a student
must already have to learn the target. Pick from anywhere in the list, not just
nearby ones. Choose none if there are no real prerequisites.

Use only labels from the candidate list, verbatim. Return JSON only, no prose. Fields:
{schema}

Target skill: {topic}

Candidate skills:
{candidates}
""",
)
