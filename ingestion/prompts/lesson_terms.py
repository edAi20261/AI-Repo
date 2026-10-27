"""LLM prompt: the term of the school year each lesson of an already split book is in."""

from shared.prompting import Audience, LanguageRule, Prompt

LESSON_TERMS_PROMPT = Prompt(
    name="LESSON_TERMS_PROMPT",
    audience=Audience.INTERNAL,
    language_rule=LanguageRule.NONE,
    template="""\
Below are the first pages of a school book, then the lessons it was split into,
in book order: each lesson's number in the list, its title, and how its first
page opens.

A school year is often split into terms (also called semesters). A book may be
for one term, or for the whole year with each term in a part of its own. Give
every lesson the term it belongs to, as a number, where the cover, the title
page, the table of contents, a part's title or the lesson's own first page says
which term it is: 1 for the first term, 2 for the second. In a book for one
term, give that term as the book's term; every lesson in it is in that term. A
book for the whole year has no book term: give each lesson its own term. Leave
a term empty when the book does not say; never guess it from the subject matter.

List every lesson, once, by its number. Return JSON only, no prose. Fields:
{schema}

The first pages:
{opening_pages}

The lessons:
{lessons}
""",
)
