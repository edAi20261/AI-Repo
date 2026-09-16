"""Vision prompt: transcribe a rendered page and locate its figures, in one call.

Two variants share one JSON contract (this general/scanned prompt and the maths
variant in ``math_transcription``); figure captioning stays a separate call.
"""

from shared.prompting import Audience, LanguageRule, Prompt

# The shared output contract. Both variants must return exactly this shape.
_JSON_CONTRACT = """\
Return ONLY a JSON object, no prose and no code fences:
{{"unreadable": false,
  "text": "<the full page text>",
  "figures": [{{"bbox": [x0, y0, x1, y1]}}, ...]}}

- "unreadable": set to true ONLY if the page is too blurred, low-resolution or
  degraded to read reliably. When true, leave "text" empty and do NOT guess —
  never invent numbers, words or a translation to fill an unreadable page. A
  wrong-but-plausible transcription is far worse than admitting the page cannot
  be read. When the page is legible, set false and transcribe fully.
- "text": the page's text in natural reading order, in the EXACT language printed
  on the page — never translate to another language (a French page stays French,
  an Arabic page stays Arabic). For Arabic, read right-to-left and keep embedded
  left-to-right runs (numbers, Latin terms, equations) in their correct place
  within the line. Keep headings on their own line. Transcribe mathematics
  faithfully — a fraction is (numerator)/(denominator), e.g. 8/12, never two
  separate lines "8" then "12"; keep roots, exponents, and any boxed final
  answer. Do NOT describe figures inside "text".
- "figures": one entry per diagram, chart, illustration or photo — NOT for text
  blocks, tables of text, or equations. "bbox" is [x0, y0, x1, y1] as FRACTIONS of
  the page width and height in the range 0..1 (x is width, y is height), origin at
  the top-left, so x1 > x0 and y1 > y0. Return [] if there are none.
"""

PAGE_TRANSCRIPTION_PROMPT = Prompt(
    name="PAGE_TRANSCRIPTION_PROMPT",
    audience=Audience.STUDENT,
    # Copied off the page, never translated — so a reply-language directive here
    # would be actively wrong. The hint only helps the model read the script.
    language_rule=LanguageRule.TRANSCRIBE,
    template=(
        "You are transcribing one page of a scanned textbook and locating its figures.\n\n"
        + _JSON_CONTRACT
        + "\nLanguage hint (may be empty): {language_hint}\n"
    ),
)
