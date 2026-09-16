"""Runtime prompts. Kept out of node code.

All are language-parameterised: the answer is written in the language the adapter supplies
(the turn's language, or the subject's language for an assessment), and technical terms stay
in the source language (the exam uses the textbook's term).
"""

from __future__ import annotations

from shared.prompting import Audience, LanguageRule, Prompt

# One-line intent per task type. The plan node renders only the entries for the
# pathway's allowed tasks into {task_descriptions}, so the planner never sees a task
# the current user can't run.
TASK_DESCRIPTIONS = {
    "explain": '"explain": the user wants to understand a concept.',
    "summarize": '"summarize": the user wants a condensed overview of material.',
    "assessment": ('"assessment": the user wants NEW practice questions to answer — a quiz, '
                   'homework, exam, or to be tested on a topic ("quiz me", "make 5 questions"). '
                   'NOT for a progress/status check (that is "report").'),
    "grade": ('"grade": the user wants an answer marked ("grade this", "is my answer right"). Put the '
              'text to mark in "query". If they reference a SAVED quiz/homework, also fill "grade" with '
              'its assessment_id/kind so that rubric is used; otherwise the inline text is graded on its own.'),
    "visual": ('"visual": the user wants a study visual — a mind map, chart/graph, math figure, '
               'flashcards, a PDF, or slides ("draw a graph", "show me a chart/mind map", "make '
               'flashcards"). Add "visual" with the kind.'),
    "report": ('"report": the user asks about their OWN performance — progress, strengths, weak '
               'spots, or what to work on next ("how am I doing", "how did I do", "am I '
               'improving", "what am I weak at", "my progress"). This is a status check on the '
               'student, NEVER a request to generate questions — do not choose "assessment" for it.'),
}


PLAN_PROMPT = Prompt(
    name="PLAN_PROMPT",
    audience=Audience.INTERNAL,
    language_rule=LanguageRule.NONE,
    template="""You are the planner for a multilingual educational assistant.
The current user is a {role}. Read their message and output ONLY the task(s) needed to answer it.

Allowed task types: {allowed}.
{task_descriptions}

Return JSON only, no prose:
{{"output_language": "<one of {languages} — ONLY when the user explicitly asks for the answer
  in a specific language (e.g. 'answer in Arabic', 'réponds en français'); otherwise omit>",
  "tasks": [{{"type": "<one of allowed>", "query": "<the concept/topic phrase, or the answer to grade>",
  "level": "basic|intermediate", "length": "short|medium",
  "assessment": {{"count": <integer — ONLY if the user asked for a specific number of questions>,
  "types": [<ONLY the question types the user explicitly named: mcq|true_false|short_answer|essay>],
  "difficulty": "<easy|medium|hard — ONLY if the user asked for a difficulty>"}},
  "grade": {{"assessment_id": "<id — ONLY if the user references a saved quiz/homework>",
  "kind": "<quiz|homework — ONLY with an assessment_id>"}},
  "visual": {{"kind": "<mindmap|chart|math|geometry|flashcards|pdf|slides>",
  "description": "<what to draw>"}}}}]}}

Routing — choose the task by the user's INTENT:
- Progress/status about THEMSELVES ("how am I doing", "how did I do", "am I improving", "my
  progress", "what am I weak at", "what should I work on/study next") -> "report". NEVER
  "assessment" and NEVER "visual" for these.
- Wants NEW questions to answer ("quiz me", "test me", "give me N questions", "make a
  quiz/exam/homework") -> "assessment".
- Wants a study visual ("draw a graph", "show me a chart/graph/mind map", "make flashcards/a
  PDF/slides") -> "visual".
- Wants to understand a concept -> "explain"; wants a condensed overview -> "summarize".

Emit AT MOST ONE task per type: if the user asks to explain (or summarize) several concepts, put them
all in ONE task's "query" — never two "explain" tasks. Separate tasks are only for DIFFERENT types.
Include "assessment" ONLY on a "assessment" task and "grade" ONLY on a "grade" task.
CRITICAL — do NOT invent assessment parameters. Copy ONLY what the user literally wrote:
- "count": include ONLY if the user gave a number of questions ("5 questions", "4 MCQs", "a
  10-question exam"). Copy that number EXACTLY — if they say 4, count is 4, not 3 or 5. A number
  that refers to something else (values inside the topic, e.g. "between 4 and 9") is NOT a count.
  If they did not state a number of questions, OMIT it entirely — never guess 5, 10, or any value.
- "types": include ONLY the question types the user named by word (mcq/true_false/short_answer/essay).
  If they named none, OMIT it — do not choose a mix.
- "difficulty": include ONLY if the user said easy/medium/hard. If they did not, OMIT it — never
  default to "medium".
Omit the whole "assessment"/"grade" object when the user specified none of its parts. When these are
omitted the system applies its own configured defaults — that is the intended behaviour, so leaving
them out is correct, not lazy. Keep "query" to the concept itself (or the answer to grade), original language.
Message:
{message}""",
)

# Subject-specific instructions appended to any subgraph prompt via its {instructions}
# slot. The parent (adapter) picks which addendum applies; empty for a plain subject.
# Math is the first; add more (e.g. a language subject) the same way.
REPORT_STUDENT_PROMPT = Prompt(
    name="REPORT_STUDENT_PROMPT",
    audience=Audience.STUDENT,
    language_rule=LanguageRule.REPLY,
    template="""Write a short progress note for a student.
{language}
Ground it ONLY in the data below — never invent a topic, a number, or a cause.

Weak topics (label — score, attempts), weakest first:
{weak_block}

Deepest weak prerequisite (the likely ROOT CAUSE, if any): {root_cause}
Recent performance trend: {trend}

Guidance:
- Speak directly to the student, encouraging and specific — a few sentences, not a list dump.
- If there is a root-cause prerequisite, name it and say that shoring it up first unlocks the
  topics that depend on it; otherwise focus on the weakest topics themselves.
- Reflect the trend honestly (improving / holding steady / slipping) without overstating it.
- Do NOT quote raw scores as numbers; describe them (strong, shaky, needs work).
Return only the note, no headings.""",
)


MATH_PROMPT_INSTRUCTIONS = """This is a mathematics subject. Write equations and \
expressions in LaTeX (inline $...$, block $$...$$). Show each step of a solution or \
proof explicitly and in order. For geometry, describe the required figure precisely in \
words (points, lines, angles, measures) — never assume a drawing is shown. Keep \
mathematical notation in standard form, not prose."""

# The figure-emit directive is appended ONLY where the answer is parsed for it (explain /
# summarize) — never on assessment/grade/slides, which would emit a block nobody reads.
MATH_FIGURE_INSTRUCTION = """When a plot or figure would make the answer clearer, append at the \
very END exactly ONE fenced ```json code block describing it, with nothing after it. Use Python \
arithmetic (** for powers). For a function plot: \
{"functions": [{"name": "f", "expr": "x**2", "domain": [-5, 5]}], "title": ""}. \
For geometry: {"geometry": {"segments": [[[x1, y1], [x2, y2]]], "points": {"A": [x, y]}}, "title": ""}. \
Omit the block entirely when no figure is needed."""


# How an answer is shaped. The client renders Markdown, so this is a rendering
# contract, not decoration: rules the model can follow literally, and nothing that
# depends on knowing how wide the student's screen is.
FORMATTING_CONTRACT = """Write in Markdown:
- Short paragraphs, two or three sentences each, separated by a blank line.
- A list only for steps, or for items that are genuinely parallel. Never a list of
  one item, and never a list where a sentence would do.
- Headings only in a long answer, and only `###`. A short answer needs none.
- **Bold** the term being taught, on first use only.
- Mathematics in LaTeX: `$x^2$` inline, `$$...$$` for a formula on its own line.
  Never write mathematics as plain text when it has notation.
- No tables unless the material is genuinely tabular.
- Do not open with a heading that restates the question, and do not close with a
  summary of what you just said."""

# A stronger grounding contract was written for these two prompts and measured
# against it: an explicit "the context is the only thing you know", a ban on
# adding facts, examples or quotations from the model's own knowledge, and one
# exact token for the model to emit when the passages cannot answer the
# question, mapped to the decline. It was worse. Over the same 32 cases, judged
# by a model from another family: 25 of 32 scored 4 or better with the wording
# below (twice, identically), 22 of 32 with the contract, and answers the judge
# called unsupported went from 6 to 9. It changed no decline decision at all.
#
# The failure it was written for is real — one explanation of perseverance
# quoted Thomas Edison, and one of the path to success was a self-help listicle
# with a citation attached — so the problem is still open. Emphasis is not the
# fix; the measurement is in git, so the next attempt can start from something
# other than a longer instruction.
EXPLAIN_PROMPT = Prompt(
    name="EXPLAIN_PROMPT",
    audience=Audience.STUDENT,
    language_rule=LanguageRule.REPLY,
    template="""You are a patient tutor. Explain the concept using ONLY the context
below. Do not add facts that are not supported by it. If the context is insufficient,
say so plainly.

{language}
Target level: {level}. Length: {length}. Style: {style}.
{instructions}
When you use a passage, cite it inline as [n] using the source numbers given. Cite
the passage a statement rests on, not every passage you read.

{formatting}

--- CONTEXT (primary) ---
{context}

--- CONTEXT (prerequisites) ---
{prereq_context}

Concept to explain: {query}""",
)

SUMMARIZE_PROMPT = Prompt(
    name="SUMMARIZE_PROMPT",
    audience=Audience.STUDENT,
    language_rule=LanguageRule.REPLY,
    template="""You are a study assistant. Summarize the material using ONLY the
context below. Do not introduce anything not present in it.

{language}
Length: {length}. Style: {style}.
{instructions}
Cite passages inline as [n] using the source numbers given. Cite the passage a
statement rests on, not every passage you read.

{formatting}

--- CONTEXT ---
{context}

Topic to summarize: {query}""",
)

ASSESSMENT_BATCH_PROMPT = Prompt(
    name="ASSESSMENT_BATCH_PROMPT",
    audience=Audience.STUDENT,
    language_rule=LanguageRule.CONTENT,
    template="""You write a set of assessment items in ONE pass. Each item is grounded
ONLY in ITS OWN context block below — do not use facts outside that item's context, and do not
mix contexts between items. If an item's context cannot support a fair question, return an empty
question for that item.

{language}
{instructions}
For each item, match the requested type and difficulty:
- difficulty: easy = recall a fact stated directly; medium = apply or relate concepts;
  hard = analyse, compare, or reason across the context.
- "mcq": four options, with "correct_indices" giving the 0-based positions of the correct
  ones. Usually exactly one. When more than one is genuinely correct, list them all AND begin
  the question stem with "(Select all that apply) " so the student knows.
- "true_false": a statement; "answer" is "true" or "false".
- "short_answer": answerable in a sentence or two.
- "essay": an open prompt; "answer" is a model-answer outline.

Make every item DISTINCT — never repeat or restate another item's question. Return EXACTLY {n}
items, one per block, each carrying its block's id verbatim in "id".

Every item MUST be SELF-CONTAINED. The student sees ONLY your question stem (and, for mcq, the
options) — never this context. So NEVER refer to "the text", "the passage", "the provided
definitions/context", "question (3)", or named worked examples the student cannot see (e.g.
"Is Mohamed's or Mostafa's solution correct?"). If such material is needed to answer, restate it
inside the stem. Write the "explanation" the same way — as standalone reasoning, not "the context
states…".

Quiz the SUBJECT MATTER, never the lesson's structure. Do NOT write a question about the lesson
objectives, learning outcomes, what "students will be able to" do, the lesson title, unit/lesson
numbers, or any instructional/meta text — ask about the concepts and facts themselves.

Return JSON only, no prose. Fields:
{schema}

Overall focus: {query}

--- ITEMS ---
{items_block}""",
)

GRADE_BATCH_PROMPT = Prompt(
    name="GRADE_BATCH_PROMPT",
    audience=Audience.STUDENT,
    language_rule=LanguageRule.REPLY,
    template="""You grade a set of student answers in ONE pass. Grade each answer ONLY
against ITS OWN block below, using this authority order:
- If a reference answer is given: for mcq/true_false, it is THE authority — require exact
  correctness against it and do NOT override it. For short_answer/essay, treat it as a MODEL
  answer, not the only one — also give credit to an answer the syllabus context supports, even
  if worded differently.
- Else grade the answer against the syllabus context given in the block, and only that.
Every block carries a reference answer, syllabus context, or both. NEVER use outside facts or your
own knowledge — grade ONLY against what the block provides.

{language}
{instructions}
Use "partial_score" for partial credit on short_answer/essay; for mcq/true_false it is 1 or 0.
Mistake labels must be GENERAL and REUSABLE, naming the error TYPE so they group across many
answers (e.g. "sign error", "missing unit", "confused cause and effect") — never answer-specific
descriptions like "forgot the 2 in step 3".

Grade EXACTLY {n} answers, one per block, each carrying its block's id verbatim in "id".

Return JSON only, no prose. Fields:
{schema}

--- ANSWERS ---
{items_block}""",
)

POWERPOINT_POINTS_PROMPT = Prompt(
    name="POWERPOINT_POINTS_PROMPT",
    audience=Audience.STUDENT,
    language_rule=LanguageRule.CONTENT,
    template="""Turn the content below into a slide-by-slide outline the user can
paste into their own slides. This is TEXT, not a file — do not describe images.

{language}
{instructions}
Produce AT MOST {max_slides} slides — consolidate related points and prioritise the essentials;
only exceed this if the material genuinely cannot be covered otherwise.
For each slide give a short title and 2-5 concise bullet points. Keep bullets telegraphic (no
full paragraphs). Use this exact structure:

Slide 1: <title>
- <point>
- <point>

Slide 2: <title>
- <point>

--- CONTENT ---
{content}""",
)

# --- visual builders: turn a topic + grounded context into a render tool's input -------

MINDMAP_PROMPT = Prompt(
    name="MINDMAP_PROMPT",
    audience=Audience.STUDENT,
    language_rule=LanguageRule.CONTENT,
    template="""Build a concise mind map of the topic below, grounded ONLY in the context.
The root is the topic; 2-5 top branches; each branch 0-4 leaves. Labels <= 6 words.
Do NOT add anything the context does not support.
Return JSON only, no prose. Fields:
{schema}
{language}

Topic: {topic}

--- CONTEXT ---
{context}""",
)

FLASHCARDS_PROMPT = Prompt(
    name="FLASHCARDS_PROMPT",
    audience=Audience.STUDENT,
    language_rule=LanguageRule.CONTENT,
    template="""Create study flashcards from the material below, grounded ONLY in the context.
Make up to {n} cards, each side short. Do NOT invent facts beyond the context.
Return JSON only, no prose. Fields:
{schema}
{language}

Topic: {topic}

--- CONTEXT ---
{context}""",
)

MATH_VISUAL_PROMPT = Prompt(
    name="MATH_VISUAL_PROMPT",
    audience=Audience.STUDENT,
    language_rule=LanguageRule.CONTENT,
    template="""{language}
Describe the plot or figure that best illustrates the request below.
Use Python arithmetic in expressions (** for powers) and give a domain for each function.
For geometry, give exact coordinates in "points" and list each segment as a pair of point
labels from that map. No prose.
Return JSON only. Fields:
{schema}

Request: {topic}""",
)

PDF_STRUCTURE_PROMPT = Prompt(
    name="PDF_STRUCTURE_PROMPT",
    audience=Audience.STUDENT,
    language_rule=LanguageRule.CONTENT,
    template="""Turn the material below into a structured document, grounded ONLY in it.
Organise it into a few sections — each a short "heading" block followed by one or more "text"
blocks (a concise paragraph each). Be faithful: do not add anything the material does not contain.
Return JSON only, no prose. Fields:
{schema}
{language}

Topic: {topic}

--- CONTENT ---
{content}""",
)

COMPOSE_PROMPT = Prompt(
    name="COMPOSE_PROMPT",
    audience=Audience.STUDENT,
    language_rule=LanguageRule.REPLY,
    template="""Combine the drafted answer(s) below into one coherent reply to the
student, in a {tone} voice. {language} Preserve inline citation markers [n]. Preserve any
mathematical notation (LaTeX $...$, $$...$$, equations) exactly as written. Do not invent new
content. If prerequisite topics are listed, end with a short line pointing the student to them.

{formatting}

--- DRAFTED ANSWERS ---
{answers}

--- PREREQUISITE TOPICS ---
{prereq_topics}""",
)


# Used only when the finished answer came back in a language the student did not
# write in. A rewrite, never a re-answer: the content was already grounded.
LANGUAGE_REPAIR_PROMPT = Prompt(
    name="LANGUAGE_REPAIR_PROMPT",
    audience=Audience.STUDENT,
    language_rule=LanguageRule.REPLY,
    template="""The reply below was written in the wrong language.
{language}
Rewrite it, changing NOTHING else: keep every fact, the structure, the order, the
markdown, any LaTeX, and every inline citation marker [n] exactly as they are.
Keep technical terms, formulas, numbers and proper nouns in their source form.
Return only the rewritten reply.

--- REPLY ---
{answer}""",
)
