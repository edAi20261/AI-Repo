"""LLM prompt: extract 3-8 topics (skills) from a lesson's chunks."""

TOPIC_EXTRACTION_PROMPT = """\
Below are the chunks of one lesson. Identify the {min_topics}-{max_topics}
distinct skills a student is meant to learn from it.

A topic is a skill, transferable across lessons — "solving linear equations",
"balancing chemical equations", "the causes of the Nile flood". It is NOT a
section heading, a page label, or the lesson title.

{language}

Most central skill first. Return JSON only, no prose. Fields:
{schema}

Lesson chunks:
{chunks}
"""
