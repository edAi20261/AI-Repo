"""LLM prompt: split a book into its units and the lessons inside each, by page."""

from shared.prompting import Audience, LanguageRule, Prompt

LESSON_BREAKDOWN_PROMPT = Prompt(
    name="LESSON_BREAKDOWN_PROMPT",
    audience=Audience.STUDENT,
    language_rule=LanguageRule.TRANSCRIBE,
    template="""\
Below is an outline of a school book, one line per page: the page number, then
the headings and opening words on that page. The first pages are given in full,
because a table of contents is usually there.

Return the book's structure: its units, and inside each unit every lesson it
contains, in book order.

A unit (also called a module, chapter or theme) is a group of lessons under one
title, such as "Unit 2: Our Environment". A unit is NEVER a lesson. A
lesson is the part of a unit taught in one or a few sessions, usually numbered
inside its unit, such as "Lesson 3: Saving Water". Look inside every
unit for its lessons: a unit's opening page often lists them, and each lesson's
first page usually carries its number or title. A unit typically has several
lessons; a unit with a single lesson is rare and only correct when the book
really teaches it that way.

These are NOT lessons: a section inside a lesson (Reading, Writing, Grammar,
Listening, Vocabulary), a single exercise, the book's cover or introduction.
Review pages, tests and question banks that close a unit belong to that unit's
last lesson.

Some parts of a book belong to no single lesson: revision of the whole term,
model exams, weekly or monthly assessments, exam papers, answer keys, a grammar
or vocabulary reference for the whole book. They are often at the end of the
book, and sometimes at its start. List each of them under sections, with its
title as printed and the outline page where it starts, so that its pages are not
filed under the lesson next to it. A book that is only questions or exams can
have sections and no lessons.

For each unit give its title as printed and its number. For each lesson give
its own title as printed, in the book's own words and language, with its number
when the book shows one, and the outline page where its own content starts. A
lesson whose pages show only its number is titled with that number as printed.
When the book has no units, return one unit with an empty title holding every
lesson.

A school year is often split into terms (also called semesters). A book may be
for one term, or for the whole year with each term in a part of its own. Give
every unit and section the term it belongs to, as a number, where the cover,
the title page, the table of contents or the title of a part says which term
it is: 1 for the first term, 2 for the second. In a book for one term every
unit and section has that term. Leave the term empty when the book does not
say; never guess it from the subject matter.

Use the page numbers of the outline, not the numbers printed in the table of
contents: the two can differ by the pages before the first lesson.

Some books teach lessons in groups: one part or section introduces two or more
lessons at once, and its pages, exercises and tests are headed with all of them
together ("Lessons 1 and 2"). Lessons taught together like that share their
pages: give every lesson of the group the same start page, the page where the
group starts. Only give a lesson a start page of its own when the book really
teaches it on pages of its own. The cover, contents and introduction before
the first lesson or section belong to it.

The book has {page_count} pages. Return JSON only, no prose. Fields:
{schema}

The first pages in full:
{opening_pages}

Outline:
{outline}
""",
)
