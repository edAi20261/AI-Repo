# Rendering the tutor's reply in the web app

For `WebApp-Frontend`. All of this is merged and deployed — the running AI
service behind the backend already behaves this way, so you can check your work
against it as you go. `docs/openapi.json` is the generated spec.

The tutor used to answer in one long paragraph, so printing it as plain text was
correct. That is no longer true. Replies now come back as **Markdown**, in the
**student's own language** including Arabic, and with the **sources as
structured data** instead of only as text.

Rendered as plain text, a student sees `**bold**` and `###` instead of the
formatting, and an Arabic reply lays out left-to-right with the punctuation on
the wrong side.

**Files:** `src/components/student-pages/ai-tutor/chat-message.tsx` is where all
of this lives.

**Nothing changes in the request.** Every field the response had before is still
there with the same meaning; `citations` is added, not swapped in. Replies with
no formatting in them render as an ordinary paragraph, so nothing regresses.

---

## Requirements at a glance

| # | Requirement | Priority | Roughly |
|---|---|---|---|
| 1 | Render the reply as Markdown | **Required** | 30 min |
| 2 | Lay Arabic out right-to-left | **Required** | 15 min |
| 3 | Fix the typing animation | **Required** | 20 min |
| 4 | Show sources as badges | **Blocked on me** | 1–2 h |
| 5 | Render maths with KaTeX | Later | — |

1–3 are what stop the current output from looking broken, and they are
independent of each other. 4 is waiting on a fix on my side — please do not start
it. 5 only matters once maths lessons are indexed, and none are yet.

**Prerequisite:** the tutor page still reads from `use-demo-chat.ts`, so it is
showing scripted demo replies. None of the below is visible until that hook calls
the real service. That was the earlier handoff — noted here so the order is
clear, not as a new request.

---

## 1. Render the reply as Markdown — required

The reply is Markdown by contract: short paragraphs, `-` and `1.` lists for
steps, `**bold**` on the term being taught, `###` headings in long answers only,
and LaTeX for maths.

### Install

```bash
npm i react-markdown remark-gfm rehype-sanitize
```

None of these are currently in `package.json`. All three support React 19.

### Change

Currently around line 90 of `chat-message.tsx`:

```tsx
<Text textStyle="body" whiteSpace="pre-wrap">
  {text}
</Text>
```

Replace with:

```tsx
<Box textStyle="body" className="tutor-md" dir="auto">
  <ReactMarkdown
    remarkPlugins={[remarkGfm]}
    rehypePlugins={[rehypeSanitize]}
  >
    {text}
  </ReactMarkdown>
</Box>
```

### Notes

- **`rehype-sanitize` is not optional.** The text is written by a language model
  reading lesson files, so it is untrusted input. Sanitizing is what stops any
  HTML inside it from being rendered.
- **Drop `whiteSpace="pre-wrap"`.** Markdown handles line breaks now. Leaving it
  on doubles the spacing between paragraphs.
- **Style the output.** Add rules on `.tutor-md` for paragraph spacing, list
  indent, `h3` size and inline `code` so it matches the bubble instead of using
  browser defaults.

### Done when

A reply containing a bold term, a numbered list and a `###` heading renders as
formatted HTML, with no raw `*` or `#` visible, and paragraph spacing matches the
rest of the chat.

---

## 2. Lay Arabic out right-to-left — required

Students ask in Arabic and the tutor answers in Arabic. The `language` field on
the response tells you which language was used (`ar`, `en`, `fr`, `de`), but you
do not need to read it for layout — the two rules below handle it.

### `dir="auto"` on the bubble

Included in the snippet above. The browser reads the first strong character and
picks the direction per message, so one thread holds both languages without you
tracking which is which.

### `<bdi>` around Latin text inside Arabic

A citation badge, a term like *photosynthesis*, a number or a formula sitting
inside an Arabic sentence will jump to the wrong side without it:

```tsx
<bdi>{children}</bdi>
```

### Logical CSS properties

Any hardcoded `marginLeft` / `paddingRight` in the component should become
`marginInlineStart` / `paddingInlineEnd` so the bubble mirrors correctly. The
`isUser` alignment and the avatar side are already handled.

### Done when

An Arabic reply reads right-to-left with the full stop on the left-hand end, and
an English term inside it stays in the correct position.

---

## 3. Fix the typing animation — required

`useTypewriter` (line 12) advances a character counter and returns
`text.slice(0, visible)` (line 34). With Markdown this hands the renderer
`**Column subtr` mid-tick — an unclosed bold marker — so the text visibly flips
between raw and formatted while it types. It also cuts Arabic words apart.

Either fix works:

- **Advance by word.** Split on whitespace keeping the separators
  (`text.split(/(\s+)/)`), and slice the token list instead of the string. Same
  effect on screen, no broken symbols.
- **Or drop the slicing** and fade the finished message in. Simpler, and it stops
  the Markdown being re-parsed on every tick.

Please keep the existing `prefers-reduced-motion` branch either way.

### Done when

No raw Markdown symbols appear at any point during the animation, and Arabic
text is not cut mid-word.

---

## 4. Show sources as badges — **not ready yet, do not start**

Replies carry `[1]`, `[2]` markers pointing at the lessons they came from. Until
now the only way to resolve them was a plain-text list at the end of the reply.
The response now carries the same information as data.

> **Wait for me before building this.** The field ships and the shape below is
> final, but the lesson names are not populated in the current index — every
> `lesson_title`, `subject_name` and `heading_path` comes back `null` today, so a
> badge would render an empty chip or a raw id. That is a fix on my side. I will
> tell you when real names are coming back; the rest of this section is here so
> you know what is coming, not as something to start now.

### The new field

The shape, with the lesson names filled in as they will be once I have
fixed the index:

```json
{
  "response": "Column subtraction lines the digits up [1][2].",
  "language": "ar",
  "citations": [
    {
      "n": 1,
      "lecture_id": "lec-9f3",
      "lesson_title": "Subtraction",
      "lesson_order": 4,
      "subject_name": "Mathematics",
      "heading_path": "Unit 2 > Column subtraction",
      "page_from": 12,
      "page_to": 13
    }
  ]
}
```

What the same call returns **today** — the ids and pages are real, the names are
not there yet:

```json
{
  "n": 1,
  "lecture_id": "061d4017-9ed6-475f-8bfd-8dc44c0f5d09",
  "lesson_title": null,
  "lesson_order": null,
  "subject_name": null,
  "heading_path": null,
  "page_from": 1,
  "page_to": 1
}
```

| Field | Type | Notes | Populated today |
|---|---|---|---|
| `n` | `int` | Matches the `[n]` in the text. Starts at 1, no gaps | yes |
| `lecture_id` | `string \| null` | What to link to, if the badge opens the lesson | yes |
| `page_from`, `page_to` | `int \| null` | Page range in the source PDF; null for lessons with no PDF | yes |
| `lesson_title` | `string \| null` | What to show on the badge or tooltip | **no — null** |
| `lesson_order` | `int \| null` | Lesson number within the subject | **no — null** |
| `subject_name` | `string \| null` | Subject the lesson belongs to | **no — null** |
| `heading_path` | `string \| null` | Where in the lesson, e.g. unit and section | **no — null** |

Guarantees from the service side: every `[n]` in the text has a matching entry,
every entry is referenced by at least one marker, and the list is capped at 6 and
deduplicated by lesson. An empty list means the reply cited nothing — a greeting,
or a question the syllabus does not cover.

### Rendering

A `components` override on `ReactMarkdown`: match `[n]` in the text node and
replace it with a chip that looks up the matching entry by `n`. Wrap the chip in
`<bdi>` per requirement 2.

### Decision needed from me

The reply text **still ends with its own `Sources:` list**, translated into the
student's language. If you render badges as well, the same information appears
twice. Tell me which you want and I will make the service send only that one —
it is a one-line change on my side and cleaner than stripping it in the
component.

### Done when

Each `[n]` renders as a badge showing the lesson it refers to, and the sources
appear once rather than twice. Blocked until the lesson names are populated.

---

## 5. Render maths with KaTeX — later

Maths comes back as LaTeX — `$x^2$` inline, `$$...$$` on its own line — because
a fraction or a root cannot be written correctly in plain text.

```bash
npm i remark-math rehype-katex katex
```

`remark-math` in `remarkPlugins`, `rehype-katex` in `rehypePlugins` after it, and
the KaTeX stylesheet imported.

**Watch out:** the default `rehype-sanitize` schema strips the class names KaTeX
puts on its output, so formulas render unstyled unless you extend the schema to
allow them. Worth knowing before spending an hour on it.

No hurry — the lessons indexed so far are Arabic language and science, so nothing
is sending LaTeX yet.

---

## Also worth doing

Not part of the rendering work, but visible to students:

- A reply can take a few seconds. The thinking state should stay up until the
  response lands.
- If the call fails, show a retry rather than an empty bubble.

---

## Reference

- Full response schema: [`docs/openapi.json`](openapi.json) → `ChatResponse`,
  `SourceRef`
- Backend-side contract: [`docs/BACKEND_INTEGRATION.md`](BACKEND_INTEGRATION.md)
- Running the service gives Swagger at `/docs`
