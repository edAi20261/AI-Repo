# Study Mode — what has to happen for the four buttons to work

**Written 16 September 2026, against these commits:**

| Repo | Branch | Commit |
|---|---|---|
| `AI-Repo` | `fix/config-single-source` | `427f453` |
| `WebAppBackEnd-` | `main` | `51fa362` |
| `WebApp-Frontend` | `main` | `b40695d` |

Every statement below was checked against that code. Where something does not
exist, it says so and gives the file it belongs in. Nothing here is assumed.

The AI-Repo commit above is a branch that is not merged yet. **Nothing in this
document depends on it.** The response contract this describes — `citations`,
`assessment`, `visuals`, `prereq_topics`, and the six task types — is identical
on `staging` and in the deployed service, so you can build and test against what
is running today.

---

## The short version

The four study modes — **Explain Lesson, Homework Help, Check My Answer, Quick
Quiz** — are all supported by the AI service **today**. No new AI endpoint is
needed and no database migration is needed.

They do not work because of three gaps:

1. **The frontend tutor is not connected to anything.** It returns scripted
   replies from a local array (`use-demo-chat.ts`). It never calls the backend.
2. **The backend drops `citations`** on the way through, so sources can never be
   shown even once the frontend is connected.
3. **The study mode is never sent anywhere.** The button changes a local
   variable that is passed to the chat hook and ignored.

Everything else — the AI service, the chat session endpoints, persistence — is
built and working.

---

## Priority order

Do them in this order. **P0 items are the minimum for all four buttons to
function.** P1 items are needed for the result to look right.

| # | Task | Repo | Priority | Rough size |
|---|---|---|---|---|
| B1 | Pass `citations` through | backend | **P0** | 10 min |
| F1 | Replace the demo chat with real API calls | frontend | **P0** | 2–3 h |
| F2 | Send the study mode as an instruction | frontend | **P0** | 30 min |
| F3 | Send real subject and lesson UUIDs | frontend | **P0** | 1 h |
| F4 | Render the reply as Markdown | frontend | **P1** | 30 min |
| F5 | Render the quiz (`assessment`) | frontend | **P1** | 1–2 h |
| F6 | Render the sources (`citations`) | frontend | **P1** | 45 min |
| E1 | Confirm the two environment variables | backend | **P0** | 5 min |

---

# PART 1 — BACKEND (`WebAppBackEnd-`)

## B1 — Pass `citations` through *(P0, blocks F6)*

The AI service returns a `citations` array on every answer. The backend's
interface does not declare it, so it is silently discarded and the frontend can
never show sources.

**Verified:** `src/ai-integration/ai-client.service.ts:22-32` — the
`ChatResponsePayload` interface has no `citations` field.
`src/ai-chat/ai-chat.service.ts:88-97` — the returned object has no
`citations` key.

### Change 1 of 3 — `src/ai-integration/ai-client.service.ts`

Add the type and the field to `ChatResponsePayload`:

```ts
export interface AiCitation {
  n: number;
  lecture_id?: string | null;
  lesson_title?: string | null;
  lesson_order?: number | null;
  subject_name?: string | null;
  heading_path?: string | null;
  page_from?: number | null;
  page_to?: number | null;
}

export interface ChatResponsePayload {
  request_id: string;
  response: string;
  language: string;
  pathway?: string | null;
  tasks: string[];
  visuals: unknown[];
  assessment: Record<string, unknown>;
  grade_result: Record<string, unknown>;
  prereq_topics: unknown[];
  citations: AiCitation[];        // <-- ADD THIS
}
```

### Change 2 of 3 — `src/ai-chat/ai-chat.service.ts`, inside `sendMessage`

Add `citations` to the metadata that is persisted:

```ts
        metadata: {
          requestId: response.request_id,
          pathway: response.pathway ?? null,
          tasks: response.tasks,
          visuals: response.visuals,
          assessment: response.assessment,
          gradeResult: response.grade_result,
          prereqTopics: response.prereq_topics,
          citations: response.citations ?? [],     // <-- ADD THIS
        } as Prisma.InputJsonValue,
```

### Change 3 of 3 — same file, the returned object

```ts
    return {
      requestId: response.request_id,
      response: response.response,
      language: response.language,
      pathway: response.pathway ?? null,
      tasks: response.tasks,
      visuals: response.visuals,
      assessment: response.assessment,
      gradeResult: response.grade_result,
      prereqTopics: response.prereq_topics,
      citations: response.citations ?? [],         // <-- ADD THIS
    };
```

**No Prisma migration is needed.** `AiChatMessage.metadata` is already a `Json?`
column (`prisma/schema.prisma:1001`), so the citations fit inside it.

**Why it matters:** without this, an answer shows `[1]` and `[2]` in the text
with nothing to click, which looks broken rather than unfinished.

---

## E1 — Confirm the two environment variables *(P0)*

The backend refuses to start without these two
(`src/ai-integration/ai-client.service.ts:50,52` — both use `getOrThrow`):

```
AI_SERVICE_BASE_URL          e.g. http://localhost:8000   (local)
AI_WEBHOOK_SHARED_SECRET     must match the AI service's .env exactly
```

Please confirm they are set in **both** your local `.env` **and** in whatever
environment the demo runs against. If the secret does not match on both sides
every call returns 401.

---

## Not needed — read this before you build it

**Do not build a `/ai/grade` proxy.** The AI service has a separate
`POST /ai/grade` endpoint, and the backend does not call it
(verified: `aiClient` is used in exactly two places — `ai-chat.service.ts:62`
and `lessons.service.ts:259`). **Check My Answer does not need it.** The chat
endpoint routes a grading request to the same grading engine internally. Adding
a proxy is a bigger piece of work than the whole rest of this list and is not
required for any of the four buttons.

---

# PART 2 — FRONTEND (`WebApp-Frontend`)

## F1 — Replace the demo chat with real API calls *(P0, blocks everything)*

**Verified:** `src/components/student-pages/ai-tutor/use-demo-chat.ts` returns
canned strings from an array after a fake 900 ms delay. There is no call to the
backend anywhere in the tutor — `grep -rn "ai-chat" src` returns nothing.

### Step 1 — add the API module

New file `src/api/ai-chat/ai-chat-api.ts`, following the same pattern as
`src/api/homework/homework-api.ts`:

```ts
import { api } from "@/api/client";
import type { Paginated } from "@/api/types";

// snake_case on purpose — see the warning under this block.
export interface AiCitation {
  n: number;
  lecture_id?: string | null;
  lesson_title?: string | null;
  lesson_order?: number | null;
  subject_name?: string | null;
  heading_path?: string | null;
  page_from?: number | null;
  page_to?: number | null;
}

export interface AiChatReply {
  requestId: string;
  response: string;               // Markdown
  language: "ar" | "en" | "fr" | "de";
  tasks: string[];                // e.g. ["explain"] | ["assessment"] | ["grade"]
  visuals: unknown[];
  assessment: Record<string, unknown>;
  gradeResult: Record<string, unknown>;
  prereqTopics: { label?: string }[];
  citations: AiCitation[];
}

export const aiChatApi = {
  createSession(body: { subjectId?: string; relatedHomeworkId?: string }) {
    return api.post<{ id: string }>("/ai-chat/sessions", { body });
  },

  sendMessage(
    sessionId: string,
    body: {
      message: string;
      subjectId?: string;
      lectureIds?: string[];
      language?: "ar" | "en" | "fr" | "de";
    },
  ) {
    return api.post<AiChatReply>(`/ai-chat/sessions/${sessionId}/messages`, {
      body,
    });
  },

  listMessages(sessionId: string, page = 1, pageSize = 50) {
    return api.get<Paginated<unknown>>(
      `/ai-chat/sessions/${sessionId}/messages?page=${page}&pageSize=${pageSize}`,
    );
  },
};
```

Three details to get right, all verified in the existing code:

- **`api.post` takes an options object, not a raw body** — it is
  `api.post<T>(path, { body })`, matching `classes-api.ts:137`.
- **Pagination is `page` + `pageSize`**, not `limit`
  (`src/common/dto/pagination.dto.ts:11,19`; default `pageSize` is 20, max 100).
- **Paths are relative to `/api`** — the client's `BASE_URL` already ends in
  `/api` and the backend sets `app.setGlobalPrefix("api")` (`src/main.ts:11`),
  so write `"/ai-chat/sessions"`, never `"/api/ai-chat/sessions"`.

> **Warning — inconsistent casing.** The backend camelCases the *top level* of
> the reply but passes the nested blobs (`citations`, `assessment`, `visuals`,
> `prereqTopics`) through **untouched from the AI service, in snake_case**. So
> it is `reply.citations[0].lesson_title`, not `lessonTitle`. This is deliberate
> and documented in `ai-chat.service.ts:43-49`. Do not "fix" it — just read the
> right key.

### Step 2 — replace the hook

Replace `useDemoChat` with a hook of the **same shape** —
`{ messages, isThinking, send, reset }` — so `ai-tutor-page.tsx` barely changes.
It must:

1. Call `aiChatApi.createSession({ subjectId })` **once**, lazily, on the first
   message, and keep the returned `id` in state.
2. Call `aiChatApi.sendMessage(sessionId, …)` for each turn.
3. Keep the full reply on the assistant message, not just the text — the UI
   needs `citations`, `assessment` and `tasks` later.

**Keep the same session id for the whole conversation.** The AI service uses it
to remember a standing instruction like *"reply in Arabic from now on"*. A new
session per message throws that away.

### Step 3 — timeouts

A real answer takes **8 to 16 seconds** (measured). The first request after the
AI service restarts takes longer because a model loads into memory once. Make
sure the client does not time out below **60 seconds**, and keep the thinking
indicator visible for the whole wait.

---

## F2 — Send the study mode as an instruction *(P0)*

**Verified:** `use-demo-chat.ts:64` — the context (which carries `studyMode`) is
received as `_context` and never used. `ai-tutor-page.tsx:102,119` — the mode is
put into the context object and the context goes nowhere.

The AI service has **no `mode` parameter**. It decides what to do by reading the
message. So the mode has to be expressed *in the message text*.

I tested this against the live planner. **Send the student's text wrapped in the
prefix for the selected mode**, in the interface language:

| Mode | English wrapper | Arabic wrapper |
|---|---|---|
| `explain` | `Explain this: {text}` | `اشرح لي: {text}` |
| `homework` | `Help me understand my homework. My question is: {text}` | `محتاج مساعدة في فهم واجبي. سؤالي: {text}` |
| `check` | `Check my answer: {text}` | `صحح لي إجابتي: {text}` |
| `quiz` | `Give me a quick quiz on this lesson. {text}` | `اعطني اختبار سريع على هذا الدرس. {text}` |

**Verified routing** (each of these was run against the live planner and
returned the task shown):

```
"اشرح لي درس جابر بن حيان"                          -> ["explain"]
"explain column subtraction"                         -> ["explain"]
"Give me a quick quiz on this lesson"                -> ["assessment"]
"اعطني اختبار سريع على هذا الدرس"                    -> ["assessment"]
"Check my answer: Question: what is 81-52? ..."      -> ["grade"]
"صحح لي إجابتي: السؤال: ... إجابتي: ..."             -> ["grade"]
"Help me understand my homework. My question is: ..." -> ["explain"]
"محتاج مساعدة في فهم واجبي. سؤالي: ..."              -> ["explain"]
```

**The wrapper is not optional for Homework Help.** Without it, "I need help with
my homework on subtraction" routed to `explain` in English but to `assessment`
in Arabic — it generated a quiz instead of helping. With the wrapper both
languages route to `explain` every time.

**For Check My Answer**, the student must supply both the question and their
answer, or there is nothing to grade. Please make the input make that obvious —
either two fields, or placeholder text like
`Question: … / My answer: …`. If only an answer arrives, the reply will be poor
and it will not be the AI service's fault.

You can confirm the routing worked by reading `reply.tasks` — it will contain
`"explain"`, `"assessment"` or `"grade"`.

---

## F3 — Send real subject and lesson UUIDs *(P0)*

**Verified:** `ai-tutor-page.tsx:85-88` — the learning setup state is hardcoded
mock data:

```ts
grade: 8, subject: "science", curriculum: "national", lesson: "photosynthesis"
```

Those are display strings, not UUIDs, and "science"/"photosynthesis" do not
exist in the database at all.

Replace the mock selects with real data using the API modules you already have:

- `subjectsApi.list({ gradeLevel, curriculumId })` — `src/api/subjects/subjects-api.ts:32`
- `lessonsApi.listBySubject(subjectId)` — `src/api/lessons/lessons-api.ts:90`

Then send the **UUIDs** on every message:

```ts
aiChatApi.sendMessage(sessionId, {
  message: wrappedText,
  subjectId: selectedSubject.id,        // UUID
  lectureIds: selectedLesson ? [selectedLesson.id] : undefined,   // UUID[]
});
```

**Why this matters more than it looks.** `lectureIds` restricts the search to
the lesson the student is actually studying. Without it, "explain this lesson"
searches all eleven indexed lessons and can answer from the wrong one. With it,
the answer is scoped correctly. This is the single biggest quality lever the
frontend controls.

**Do not send `language`.** Leave it undefined. The AI service detects the
language from the message and remembers a standing instruction. Sending the
interface language overrides the student's actual language — an Arabic question
would be answered in English because the UI happens to be in English.

---

## F4 — Render the reply as Markdown *(P1)*

**Verified:** `chat-message.tsx:90-91` renders with
`<Text whiteSpace="pre-wrap">{text}</Text>` — plain text.

Every answer is Markdown: `**bold**`, `###` headings, `-` and `1.` lists, and
LaTeX for maths. Rendered as plain text the student literally sees the asterisks
and hashes.

This is already specified in detail — including the exact packages, the
sanitiser, the RTL handling and the typewriter fix — in
**`docs/FRONTEND_CHAT_RENDERING.md` sections 2, 3 and 4**. Follow that document;
do not re-derive it.

---

## F5 — Render the quiz *(P1 — without this, Quick Quiz shows nothing useful)*

When the mode is Quick Quiz, the text reply is only one line
(*"Here is a quiz of 5 questions."*). **The questions are in `reply.assessment`**,
as structured JSON. If you do not render it, the button appears to do nothing.

Real example returned by the running service:

```json
{
  "subject_id": "fcb7398d-…",
  "count": 1,
  "complete": true,
  "degraded": false,
  "language": "ar",
  "items": [
    {
      "type": "mcq",
      "question": "ما هو الغرض الرئيسي من الرسالة …؟",
      "topic_id": "3b9802b4-…",
      "lecture_id": "e74be3e6-…",
      "answer": ["استفسار"],
      "options": [
        { "text": "تهنئة",   "is_correct": false },
        { "text": "طلب",     "is_correct": false },
        { "text": "استفسار", "is_correct": true  },
        { "text": "شكر",     "is_correct": false }
      ],
      "explanation": "الفقرة الأولى تذكر أن الغرض من الكتابة …",
      "difficulty": "hard",
      "multiple_correct": false
    }
  ]
}
```

Minimum for the demo: render each `item.question`, and `item.options` as
radio buttons. Reveal `is_correct` and `explanation` after the student answers.

`type` can be `mcq`, `true_false`, `short_answer` or `essay`. For the demo,
handling `mcq` and `true_false` is enough — show `short_answer`/`essay` as a
text box.

**If `assessment.degraded` is `true`**, the model failed; show the text reply
and no quiz.

---

## F6 — Render the sources *(P1, needs B1 first)*

Answers contain `[1]`, `[2]` markers. `reply.citations` says what each number
is:

```json
[
  { "n": 1,
    "lesson_title": "Unit 2 · Lesson 2: Jabir ibn Hayyan",
    "subject_name": "Arabic",
    "lesson_order": 4,
    "page_from": 2, "page_to": 2,
    "lecture_id": "e74be3e6-…" }
]
```

Render the markers as small clickable badges and list the sources under the
answer. `docs/FRONTEND_CHAT_RENDERING.md` section 7 has the exact shape and a
worked rendering example.

**If `citations` is empty, the answer was a refusal** — the AI service found
nothing in the syllabus. Do not show an empty "Sources" heading in that case.

---

# PART 3 — DATA (the database, which you own)

## D1 — Lesson titles are stored transliterated *(not blocking, but visible)*

The indexed Arabic lessons have English titles: `Unit 2 · Lesson 4: AlShabab
Poem`. So an Arabic answer ends with a sources list in English:

> `المصادر: [1] Arabic · الدرس 6: Unit 2 · Lesson 4: AlShabab Poem, p.8`

If the Arabic title were stored in `content.lessons.title`, the sources line
would be fully Arabic — and lesson-title searches would work better, because the
title is matched against the student's question.

**Not required today.** Flagging it because it is yours and it is visible in
every Arabic answer.

## D2 — Only published + ingested lessons are searchable

A lesson is only reachable by the AI if it is `published` **and** its ingestion
finished. Publishing triggers ingestion automatically
(`lessons.service.ts:259`), but ingestion is fire-and-forget — if the AI service
was down at that moment the lesson stays pending and is never retried
automatically.

If a lesson you expect is missing from answers, that is the first thing to
check.

---

# PART 4 — ACCEPTANCE TEST

Run these four in the UI when you are done. This is exactly what I will check.

| # | Mode | Type this | Must happen |
|---|---|---|---|
| 1 | Explain Lesson | `اشرح لي درس جابر بن حيان` | Arabic answer, right-to-left, Markdown rendered, at least one `[1]` badge, sources list naming the Jabir lesson |
| 2 | Explain Lesson | `explain column subtraction` | English answer, sources naming a Subtraction lesson |
| 3 | Quick Quiz | `(lesson selected)` press the button | `reply.tasks` contains `assessment`, and the questions render as options — not just one line of text |
| 4 | Check My Answer | `Question: what is 81 - 52? My answer: 29` | `reply.tasks` contains `grade`, and the reply says the answer is wrong and why |

And one negative test that must **not** be "fixed":

| 5 | Explain Lesson | `explain photosynthesis` | *"There isn't enough syllabus material to answer this yet."* No sources, no invented answer. **This is correct behaviour.** |

---

# What I am doing on my side

Nothing in this document depends on an AI-service change. The running service
already behaves as described, so you can build and test against it immediately.

If any of it does not match what you observe, tell me before you work around it
— it means I have documented something wrong, and I would rather fix the
document than have you build against a guess.
