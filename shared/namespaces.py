"""What a chunk is to the platform: the syllabus a student studies, or
background a lesson is taught from.

A ministry book is the syllabus: its lessons are what the app shows. An
external book (a commercial guide, a question bank, a revision booklet) is
background: it is searched alongside the syllabus of its subject, never shown
as a lesson of its own, and never named to a student as a source.
"""

SYLLABUS = "syllabus"
BACKGROUND = "external"

# A topic is background when the lesson it was named from is: its chunks carry
# the namespace, and ai.topics has no column of its own for it.
NOT_BACKGROUND_TOPIC = (
    "lecture_id NOT IN (SELECT lecture_id FROM ai.vector_index "
    f"WHERE namespace = '{BACKGROUND}')")
