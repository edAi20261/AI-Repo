"""LLM prompt: split a book into the lessons it teaches, by page."""

from shared.prompting import Audience, LanguageRule, Prompt

LESSON_BREAKDOWN_PROMPT = Prompt(
    name="LESSON_BREAKDOWN_PROMPT",
    audience=Audience.STUDENT,
    language_rule=LanguageRule.TRANSCRIBE,
    template="""\
Below is an outline of a school book, one line per page: the page number, then
the headings and opening words on that page. The first pages are given in full,
because a table of contents is usually there.

List the lessons the book teaches, in book order. A lesson is the unit a
teacher would teach in one or a few sessions and a student would open on its
own: "Unit 1 - Lesson 2: My Daily Routine". It is NOT a single exercise, a
section inside a lesson (Reading, Writing, Grammar, Listening), a revision
sheet for one lesson, or the book's cover.

For each lesson give:
- title: its title as printed in the book, in the book's own words and
  language, including its unit and lesson numbers when the book shows them
- unit_number: the unit it belongs to, when the book numbers units
- start_page: the page number (from the outline) where the lesson starts

Use the page numbers of the outline, not the numbers printed in the table of
contents: the two can differ by the pages before the first lesson. Find each
lesson's first page in the outline. Pages before the first lesson belong to
it; revision and exam pages at the end belong to the last lesson unless the
book presents them as a lesson of their own.

The book has {page_count} pages. Return JSON only, no prose. Fields:
{schema}

The first pages in full:
{opening_pages}

Outline:
{outline}
""",
)
