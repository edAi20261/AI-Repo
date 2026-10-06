---
description: Build an approved plan with the developer and QA agents, test-first
argument-hint: <change id>
---

Change folder: `sdlc/changes/$ARGUMENTS/`. You are the orchestrator: you
delegate, check each handoff, and stop at the gates. You do not write code or
tests yourself.

## Gate: the plan is approved

`plan.md` must say `Status: approved` and name someone under `Approved by:`. If
not, stop and tell me. Nothing is built without an accepted plan.

## 1. QA proves it first

Delegate to `qa-tester`: "Prove sdlc/changes/$ARGUMENTS/." When it returns,
check the handoff yourself. `qa.md` must have a `Proof commit:`, and
`.venv/bin/python scripts/verify.py <the Proof test ids>` must fail. If the tests
pass, or fail with an import error, collection error or typo instead of an
assertion, send QA back with what you saw.

## 2. Developer builds it

Delegate to `developer`: "Build sdlc/changes/$ARGUMENTS/." If its final message
starts with `BLOCKED:`, stop and bring me the reason. That decision is mine.

## 3. QA verifies it, in a fresh context

Delegate to a **new** `qa-tester`, not the one from step 1: "Verify
sdlc/changes/$ARGUMENTS/." A verdict from the context that wrote the tests is
coloured by it.

## 4. Loop or stop

- `Verdict: FAIL`: send the developer back with "Fix the findings in
  sdlc/changes/$ARGUMENTS/qa.md." Then return to step 3 and increment `Round:`.
  After round 3, stop and bring me the open findings. A change that will not
  converge needs a human, not a fourth round.
- `Verdict: PASS`: stop and report the verdict, the `scripts/verify.py` tail,
  the deviations from the plan, and the commits on the branch.

Do not push, open a PR or merge. Those are my gates.
