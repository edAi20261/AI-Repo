# How we write code here

Every rule below is here because breaking it cost us something real. The failure
is named next to each one, so none of this is style preference — it is a list of
ways this system has already gone wrong.

Read this before changing code. If a rule blocks something genuinely necessary,
change the rule in a PR and say why; don't route around it.

---

## 1. Fix the producer, never filter at display

If data is wrong, regenerate it. Hiding it downstream leaves the wrong rows in
the database, leaves every other consumer exposed, and removes the evidence that
anything is wrong.

> Arabic lessons were tagged with English topic labels, so a filter dropped
> labels whose script did not match the reply. The rows stayed. The API kept
> serving them. And the filter only worked for Arabic — an English label under a
> French answer went straight through.

**Instead:** validate where the data is created. `call_structured` takes a
`validate=` hook whose failure is handed back to the model as feedback.

## 2. A contract that is described must be enforced

A docstring, a `Field(description=...)` or a comment is documentation, not a
constraint. If it matters, something must fail when it is violated.

> `TopicItem.label` said "in the lesson's main language" from the first day.
> Nothing checked. Pydantic validates shape, never content.

## 3. One decision, one place

A question answered in two places will be answered differently.

> The turn's language was decided in three, and one reply could hold two
> languages. There is one resolver now (`shared/language/resolver.py`), and every
> node reads its result.

## 4. Put an invariant where it cannot be bypassed

If correctness depends on every call site remembering something, it will break at
the site nobody thought about.

> Document and query embeddings must be folded identically — folding one and not
> the other is worse than folding neither. The fold lives inside the `Embedder`
> base class, not at the six call sites.

## 5. Defaults live in one place

`config.get(key, default)` puts a second value in the source. The two drift, and
which one applies becomes an accident of whether an overlay carried the block.

> Six keys had drifted to the opposite of what ships. `agent.min_score` was 0.50
> in config and 0.0 at five call sites — the value that answers questions the
> syllabus cannot answer.

Enforced by `tests/unit/test_config_contract.py`.

## 6. Never return a neutral value a caller cannot distinguish from success

Empty dict, empty list, `None`, zero — if that is also what success looks like,
the failure is invisible.

> `parse_structured` returned `{}` on any failure. Grading's degrade path was
> correct and never fired, so every answer fell through to string matching,
> `degraded` stayed `False`, and mastery was written from it.

## 7. Degrade loudly

Graceful degradation is fine. Silent degradation is not. Log the reason, count
it, and make it visible to someone who could fix it. `except Exception: pass` is
never acceptable.

> A vision transcription failure returned `None` with no log, so an expired API
> key and a blurred page were indistinguishable. Operators saw "needs review" on
> a whole textbook.

## 8. Parse liberally, emit strictly

Accept every form a real producer emits; emit exactly one.

> Citation markers were matched with `\[\d+\]` while models write `[1, 2]`. Every
> answer that cited two passages at once was treated as ungrounded, so citations
> were discarded and students read bracketed numbers pointing at nothing.

## 9. A threshold comes from a sweep, and the numbers go in the comment

Never guess a number that changes behaviour. Measure it, and leave the evidence
where the next person will find it.

> `agent.min_score` — the comment beside it records what 0.35 and 0.50 each did
> to the same eval set.

## 10. A regression test must fail before the fix

If you did not watch it fail, you have not proved it tests the bug.

> The first grading regression test passed against the broken code, because the
> fake returned a dict where the real call raised.

## 11. Expensive resources load lazily, and never on a health-check path

> Resolving the embedder while building the graph moved a 2 GB model load into
> startup. The load balancer's health check timed out, ECS killed the task, and
> the deploy looped.

## 12. Measure before and after a behaviour change

If you cannot show the number moved, you are guessing.

> `scripts/eval_retrieval.py` exists because there was no way to tell whether a
> retrieval change helped. It reports precision@1 and how well answerable
> questions separate from unanswerable ones.

## 13. Build for the general case, not the example in front of you

A branch keyed to one subject, one language or one textbook is a bug waiting for
the second customer.

> Arabic is a first-class branch in several places and every other language is
> "else". It works because the pilot corpus is Arabic.

---

## Working agreements

- **Scope.** This repo only. The backend (`WebAppBackEnd-`) and frontend
  (`WebApp-Frontend`) belong to another engineer; read them freely, change
  nothing without being asked.
- **Comments** explain *why*, never *what*. If the code needs a comment to say
  what it does, the code is wrong.
- **Tests** are named for the behaviour they protect, and say in the docstring
  what broke. `test_a_prompt_a_student_reads_says_what_language_it_is_in`, not
  `test_prompt_2`.
- **Lint** must not regress: `ruff check` before and after, no new findings.
  Never run `ruff --fix` across the repo — it reformats unrelated files and has
  already deleted a deliberate re-export.
- **PRs** are one concern each, and describe the failure they fix, not the diff.
  No attribution lines.
- **Shared resources.** `DATABASE_URL` and Qdrant point at production. Reads are
  free; anything that writes gets confirmed first.
- **Corpus licensing.** A book used to measure the pipeline is not a book we may
  index. *Science Fusion* (Houghton Mifflin Harcourt, © 2017) is the fixture the
  layout parser was measured against; its copyright page forbids reproduction
  "by any information storage and retrieval system" without written permission,
  which is what this pipeline is. It stays a local fixture: it is not in
  `ai.chunks` and not in the index, and it must not be ingested or demoed from.
  Permission, if it is ever wanted, goes through
  customercare.hmhco.com/permission. Check the copyright page of anything new
  before it reaches `ingest`.

## Layering

```
shared/      ← agent/, ingestion/, system/     (leaf: depends on nothing above)
agent/       ← system/
ingestion/   ← system/
agent/  ⟷  ingestion/                          never import each other
```

`shared/` is where a thing goes when both halves need it. Two implementations of
one job is the bug, not the duplication of a few lines.
