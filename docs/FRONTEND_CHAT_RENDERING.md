# What the tutor chat needs on the platform side

The AI service changed. This is everything that has to happen in
`WebAppBackEnd-` and `WebApp-Frontend` for a student to see the difference.

All of it is merged and deployed — the running service already behaves this way,
so you can check your work against it as you go. `docs/openapi.json` is the
generated spec.

## What changed in the service

1. **Replies are Markdown now.** Short paragraphs, `-` and `1.` lists for steps,
   `**bold**` on the term being taught, `###` headings in long answers, LaTeX for
   maths.
2. **Replies come back in the student's language.** Arabic in, Arabic out.
3. **Greetings and thanks are answered as conversation**, not looked up in the
   syllabus.
4. **The sources are structured data**, not only a list at the end of the text.

Rendered as plain text, a student sees `**bold**` and `###` instead of the
formatting, and an Arabic reply lays out left-to-right with the punctuation on
the wrong side.

---

## Who does what

| # | Change | Where | Priority |
|---|---|---|---|
| 1 | Stop forcing the answer language | Frontend | **Bug — do first** |
| 2 | Render the reply as Markdown | Frontend | **Required** |
| 3 | Right-to-left layout | Frontend | **Required** |
| 4 | Fix the typing animation | Frontend | **Required** |
| 5 | Send the selected subject and lesson | Frontend | Recommended |
| 6 | Pass `citations` through | **Backend** | Needed for 7 |
| 7 | Show sources as badges | Frontend | Recommended |
| 8 | Maths with KaTeX | Frontend | Later |

**Prerequisite for all of it:** the tutor page still reads from
`use-demo-chat.ts`, so it shows scripted demo replies. Nothing below is visible
until that hook calls the real service — that was the earlier handoff.

**Nothing changes in the request shape.** Every response field that existed still
exists with the same meaning; `citations` is added, not swapped in.

---

## 1. Stop forcing the answer language — bug, do first

**Symptom:** a student writes in Arabic, the tutor answers in English. Same for
French. Every reply comes back English.

**Cause:** the request carries a `language` field, and it is being filled with the
**interface** language — the one from the header dropdown. The service treats
`language` as an explicit instruction from the platform and honours it over
everything else, including the language the student actually typed in:

```
Arabic question + language: "en"   ->  answered in English   (decided by: api)
Arabic question, no language sent  ->  answered in Arabic    (decided by: script)
```

The service is doing as it was told. The header dropdown is the language of the
app's buttons and menus — it is not the student saying "answer me in English".

**Fix:** do not send `language` at all, unless you have a real per-student
preference for the tutor. With the field absent, the service reads the message:
Arabic script means Arabic, every time.

```ts
// wrong — the header dropdown is not a statement about the answer
language: i18n.language

// right — omit it, and let the message decide
```

**Done when:** a question typed in Arabic comes back in Arabic while the
interface is still set to English.

---

## 2. Render the reply as Markdown — required

### Install

```bash
npm i react-markdown remark-gfm rehype-sanitize
```

None of these are in `package.json` today. All three support React 19.

### Change

In `src/components/student-pages/ai-tutor/chat-message.tsx`, around line 90:

```tsx
<Text textStyle="body" whiteSpace="pre-wrap">
  {text}
</Text>
```

becomes:

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
- **Drop `whiteSpace="pre-wrap"`.** Markdown handles line breaks now; leaving it
  on doubles the spacing between paragraphs.
- **Style the output.** A few rules on `.tutor-md` for paragraph spacing, list
  indent, `h3` size and inline `code`, so it matches the bubble instead of using
  browser defaults.

**Done when:** a reply with a bold term, a numbered list and a `###` heading
renders as formatted text, with no raw `*` or `#` visible.

---

## 3. Right-to-left layout — required

### `dir="auto"` on the bubble

Included in the snippet above. The browser reads the first strong character and
picks the direction per message, so one thread holds both languages without you
tracking which is which.

### `<bdi>` around Latin text inside Arabic

A citation badge, a term like *photosynthesis*, a number or a formula inside an
Arabic sentence jumps to the wrong side without it:

```tsx
<bdi>{children}</bdi>
```

### Logical CSS properties

Any hardcoded `marginLeft` / `paddingRight` in the component becomes
`marginInlineStart` / `paddingInlineEnd` so the bubble mirrors correctly.

**Done when:** an Arabic reply reads right-to-left with the full stop on the
left-hand end, and an English term inside it stays in the right place.

---

## 4. Fix the typing animation — required

`useTypewriter` (line 12 of `chat-message.tsx`) advances a character counter and
returns `text.slice(0, visible)` (line 34). With Markdown, the renderer gets
`**Column subtr` mid-tick — an unclosed bold marker — so the text visibly flips
between raw and formatted while it types. It also cuts Arabic words apart.

Either fix works:

- **Advance by word.** Split on whitespace keeping the separators
  (`text.split(/(\s+)/)`) and slice the token list instead of the string. Same
  effect on screen, no broken symbols.
- **Or drop the slicing** and fade the finished message in. Simpler, and it stops
  the Markdown being re-parsed on every tick.

Please keep the existing `prefers-reduced-motion` branch either way.

**Done when:** no raw Markdown symbols appear at any point while it types, and
Arabic is not cut mid-word.

---

## 5. Send the selected subject and lesson — recommended

The Learning Setup panel lets a student pick Grade, Subject, Curriculum and
Lesson, but none of it reaches the service — the hook is called with no options,
so every question searches the whole syllabus.

It usually still finds the right lesson, but a student who has opened
Photosynthesis and asks "explain this" can be answered from another subject.

Pass the selected subject, and the lesson when there is one:

```ts
useAiChat({ subjectId, lectureIds: lessonId ? [lessonId] : undefined })
```

Both are already fields on the request — nothing is needed from me.

> **Study Mode** (Explain Lesson / Homework Help / Check My Answer / Quick Quiz)
> also does not reach the service today. The service works out what to do from
> the message itself, so nothing is broken. If you want those buttons to actually
> steer the answer, say so and I will add a field for it.

**Done when:** asking "explain this lesson" with a lesson selected cites that
lesson.

---

## 6. Pass `citations` through — backend

`citations` is dropped before it reaches the browser. The AI service sends it,
but `ai-client.service.ts` lists the response fields explicitly and
`ai-chat.service.ts` rebuilds the object field by field, so anything not named is
lost.

**`src/ai-integration/ai-client.service.ts`** — add to the response interface:

```ts
citations: unknown[];
```

**`src/ai-chat/ai-chat.service.ts`** — add to the stored `metadata` (around line
77) and to the returned object (around line 90):

```ts
citations: response.citations,
```

Without this, item 7 cannot work at all.

**Done when:** the browser's network tab shows `citations` on the message
response.

---

## 7. Show sources as badges — recommended

Replies carry `[1]`, `[2]` markers pointing at the lessons they came from. Until
now the only way to resolve them was the plain-text list at the end of the reply.

### The shape

```json
{
  "response": "Column subtraction lines the digits up [1][2].",
  "language": "ar",
  "citations": [
    {
      "n": 1,
      "lecture_id": "061d4017-9ed6-475f-8bfd-8dc44c0f5d09",
      "lesson_title": "Subtraction Part 2",
      "lesson_order": 0,
      "subject_name": "Mathematics",
      "heading_path": null,
      "page_from": 1,
      "page_to": 1
    }
  ]
}
```

| Field | Type | Notes |
|---|---|---|
| `n` | `int` | Matches the `[n]` in the text. Starts at 1, no gaps |
| `lesson_title` | `string \| null` | What to show on the badge or tooltip |
| `lesson_order` | `int \| null` | Lesson number within the subject |
| `subject_name` | `string \| null` | Subject the lesson belongs to |
| `lecture_id` | `string \| null` | What to link to, if the badge opens the lesson |
| `page_from`, `page_to` | `int \| null` | Page range in the source PDF |
| `heading_path` | `string \| null` | Where in the lesson. **Empty on everything indexed so far** — do not rely on it yet |

Guarantees from my side: every `[n]` in the text has a matching entry, every
entry is referenced by at least one marker, the list is capped at 6 and
deduplicated by lesson, and the same source is never cited twice in a row. An
empty list means the reply cited nothing — a greeting, or a question the syllabus
does not cover.

Add it to `SendMessageResponse` in `src/api/ai-chat/ai-chat-api.ts` — it is not
in the type today.

### Rendering

A `components` override on `ReactMarkdown`: match `[n]` in the text node and
replace it with a chip that looks up the matching entry by `n`. Wrap the chip in
`<bdi>` per item 3.

### One decision for me

The reply text **still ends with its own `Sources:` list**, translated into the
student's language. If you render badges too, the same information shows twice.
Tell me which you want and I will make the service send only that one — a
one-line change on my side, cleaner than stripping it in the component.

**Done when:** each `[n]` renders as a badge naming its lesson, and the sources
appear once rather than twice.

---

## 8. Maths with KaTeX — later

Maths comes back as LaTeX — `$x^2$` inline, `$$...$$` on its own line — because a
fraction or a root cannot be written correctly in plain text.

```bash
npm i remark-math rehype-katex katex
```

`remark-math` in `remarkPlugins`, `rehype-katex` in `rehypePlugins` after it, and
the KaTeX stylesheet imported.

**Watch out:** the default `rehype-sanitize` schema strips the class names KaTeX
puts on its output, so formulas render unstyled unless you extend the schema to
allow them. Worth knowing before spending an hour on it.

No hurry — nothing indexed so far sends LaTeX.

---

## Small things, same area

- A reply can take several seconds. The thinking state should stay up until the
  response lands.
- If the call fails, show a retry rather than an empty bubble.
- The tutor panel still shows a **Demo** chip and a "This is a limited demo"
  footer. Once it answers for real, both should go.

---

## Reference

- Full response schema: [`docs/openapi.json`](openapi.json) → `ChatResponse`,
  `SourceRef`
- Backend-side contract: [`docs/BACKEND_INTEGRATION.md`](BACKEND_INTEGRATION.md)
- Running the service gives Swagger at `/docs`
