"""LLM prompt: write a lesson's existing topic labels in the lesson's language.

A repair, not an extraction. The skills named in ``ai.topics`` are the right
skills; they were named in the wrong language, because the extraction prompt did
not say which one to use. Re-extracting would name new skills and mint new
topic_ids, and a topic_id is what a student's mastery history hangs from.
"""

from shared.prompting import Audience, LanguageRule, Prompt

TOPIC_RELABEL_PROMPT = Prompt(
    name="TOPIC_RELABEL_PROMPT",
    audience=Audience.STUDENT,
    language_rule=LanguageRule.CONTENT,
    template="""\
Below are skills already identified in one lesson, each with an id. They were
named in the wrong language.

Write each one again, naming the same skill — do not add skills, drop skills,
merge them, or change which skill an id refers to.

{language}

Return every id exactly once. JSON only, no prose. Fields:
{schema}

Skills:
{topics}
""",
)
