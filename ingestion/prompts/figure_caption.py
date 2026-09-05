"""Vision prompt: caption + alt_text for a cropped figure."""

FIGURE_CAPTION_PROMPT = """\
The image is a single figure cropped from a textbook page. Produce two things
about what it shows:

1. caption  — one sentence naming what the figure depicts (a labelled diagram of
   the heart, a right-triangle with sides 3-4-5, a bar chart of rainfall).
2. alt_text — a fuller description a student who cannot see the image could learn
   from: the parts, labels, relationships, and any numbers shown.

Write BOTH the caption and the alt_text in the language given by the hint below
(e.g. ar = Arabic, en = English, fr = French) — never translate to another
language; if the hint is empty, use the language of the text visible in the image.
For Arabic, write right-to-left and keep any embedded numbers or Latin terms in
place. Describe content, not style. Do not guess beyond what is visible.
Return JSON: {{"caption": "...", "alt_text": "..."}}

Surrounding heading (may be empty): {heading_path}
Language (write your answer in this language): {language_hint}
"""
