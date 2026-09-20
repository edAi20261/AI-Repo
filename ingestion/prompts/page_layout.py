"""LLM prompt: read a page image as blocks rather than as flat text.

The pipeline already sends every scanned page to a vision model once. Asking
that same call for the page's structure instead of a wall of text costs output
tokens, not another request — and it is the only way the Arabic corpus becomes
layout-aware, because none of it has a text layer for the cheap tier to read.
"""

from shared.prompting import Audience, LanguageRule, Prompt

PAGE_LAYOUT_PROMPT = Prompt(
    name="PAGE_LAYOUT_PROMPT",
    audience=Audience.STUDENT,
    # Copied off the page, never translated. A textbook rendered into the
    # student's language is not the textbook.
    language_rule=LanguageRule.TRANSCRIBE,
    template="""\
You are a document layout parser. Look at this page image.

Return every piece of text on the page as a block, in the order a human would
read it. Transcribe exactly what you see, including text that is rotated,
curved, or printed over an image. Do not invent text. Do not summarise. Do not
translate.

Rules:
- bbox is [x0, y0, x1, y1] as fractions of the page width and height, origin at
  the top-left. A block spanning the left half of the page is [0, y0, 0.5, y1].
- reading_order numbers the blocks from 1, in the order they should be read.
- Text printed upside down is almost always a self-check answer key. Transcribe
  it correctly, set type to "answer_key" and rotation to 180.
- A short label or caption that only means something next to another block sets
  belongs_to to that block's reading_order.
- Repeated legal text, copyright lines and image credits are type
  "boilerplate", wherever they sit on the page.
- page_layout is "image_dominant" when one large image and a title dominate,
  "scattered" when the text boxes have no column structure and only make sense
  together, "diagram" when the boxes annotate parts of a single illustration,
  and "standard" otherwise.

If the page is too degraded to read with confidence, return no blocks and set
"unreadable": true. A page invented is worse than a page flagged.

Language hint (may be empty): {language_hint}

Return JSON only, no prose:
{schema}
""",
)
