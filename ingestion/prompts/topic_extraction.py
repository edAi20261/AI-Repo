"""LLM prompt: extract 3-8 topics (skills) from a lesson's chunks."""

TOPIC_EXTRACTION_PROMPT = """\
Below are the chunks of one lesson. Identify the {min_topics}-{max_topics}
distinct skills a student is meant to learn from it.

A topic is a skill, transferable across lessons — "solving linear equations",
"balancing chemical equations", "the causes of the Nile flood". It is NOT a
section heading, a page label, or the lesson title.

For each topic return:
- label       — the skill, 2-6 words, in the lesson's main language.
- description — one sentence on what mastering it means.
- bloom_level — one of: remember, understand, apply, analyze, evaluate, create.

Return a JSON array of objects with those three keys, most central skill first.

Lesson chunks:
{chunks}
"""
