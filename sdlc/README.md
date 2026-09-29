# The squad: a developer agent and a QA agent

This is the Build and Test half of the
[AI-native SDLC](https://claude.com/blog/the-ai-native-sdlc-playbook). Two Claude
Code subagents do the work between two human decisions. Every handoff is a file
committed to git, so the chain of commits is also the audit trail: who asked for
what, what the agents produced, and who approved it.

```
 human           developer          qa-tester          developer        qa-tester (fresh)      human
   │                 │                  │                  │                  │                  │
intent.md ──▶ plan.md (draft) ──▶ ✋ approve ──▶ failing proof tests ──▶ code until green ──▶ qa.md verdict ──▶ ✋ PR
 /sdlc-intent     /sdlc-plan                        └──────────── /sdlc-build ─────────────┘
                                                                      ▲                 │
                                                                      └── FAIL (≤ 3) ───┘
```

| Stage | Who | Artifact | Gate that ends it |
|---|---|---|---|
| Plan | you, with Claude | `intent.md` | you set `Status: accepted` |
| Design and plan | `developer` | `plan.md` | a human sets `Status: approved` and signs `Approved by:` |
| Prove | `qa-tester` | failing tests and a `Proof commit:` in `qa.md` | the orchestrator watches them fail |
| Build | `developer` | commits | `scripts/verify.py` passes; the hook enforces this |
| Verify | a new `qa-tester` | `qa.md` with `Verdict:` | PASS, or back to the developer, at most 3 rounds |
| Deploy | you | the PR | code-owner review, then CI |

## Why QA writes the tests first

A test the developer writes after the code proves only that the code does what
the developer thought. QA writes each proof test from the plan and watches it
fail for the right reason before any code exists, which is `CLAUDE.md` rule 10.
The developer then cannot edit that test. A green suite at the end is therefore
evidence, because the thing being judged had no way to change the judge.

The verifying QA agent is a fresh subagent that did not write the tests. It
takes its evidence from the toolchain rather than from the developer's report.

## What is enforced and what is only asked

The agent definitions are instructions, and a model can drift from an
instruction. Anything that must hold without exception is also in
`.claude/hooks/sdlc_guard.py`, which runs from `.claude/settings.json` in every
session, and is tested in `tests/unit/test_sdlc_squad.py`.

| Rule | How it is enforced |
|---|---|
| The developer does not edit existing tests, `conftest.py`, `pyproject.toml`, `.github/`, `.claude/` or `scripts/verify.py` | The hook denies the edit, through the file tools and the common shell writers |
| The developer does not report done while lint or the suite is red | The `SubagentStop` hook runs `scripts/verify.py` and sends the failure back. The only exit is a final message starting with `BLOCKED:` |
| QA does not fix the code under test | The hook denies writes outside `tests/` and `sdlc/` |
| Production writes are confirmed by a human | The hook returns `ask` for ingestion, `bootstrap.py`, `reembed_chunks.py --apply`, `aws ecs`/`s3` writes, `psql` writes, and pushes to `main` |
| *Science Fusion* is never ingested | The hook denies it, including when the book is named only inside a manifest |
| Unit tests never reach production | `scripts/verify.py` forces the test `DATABASE_URL` and Qdrant, whatever `.env` says |
| A `PASS` verdict names proof tests that exist | `tests/unit/test_sdlc_squad.py` checks every `sdlc/changes/*/` |
| Plan-first, test-first, fresh-context verification, three-round cap | Only asked, in the agents and `/sdlc-build`. The artifacts show whether it happened |

The shell checks read the command, so they are a tripwire rather than a sandbox:
a determined `python -c` can still write a file. QA's first verification step is
the backstop, because `git diff <proof commit> HEAD -- <proof tests>` must be
empty.

## Running it

In Claude Code, from the repository root, with the `.venv` from the README's
quickstart:

```
/sdlc-intent Students who ask in French get cited passages in Arabic
# review intent.md, set Status: accepted
/sdlc-plan french-citations
# review plan.md, set Status: approved and Approved by: <you>
/sdlc-build french-citations
# read qa.md; if PASS, push the branch and open the PR yourself
```

Each step is also a plain request to a subagent, for example "use the developer
subagent to write plan.md for sdlc/changes/french-citations/". The commands only
fix the order and the checks between steps.

## Measuring whether it works

Everything below can be read from git, so none of it needs a dashboard to begin
with.

- Time from `intent.md` to an approved `plan.md`, and from approval to QA's PASS.
- Rounds per change, from `Round:` in `qa.md`. Rising rounds mean the plans are
  too loose.
- Changes that reach PASS in round 1.
- Findings QA's probing catches that the plan's proof did not list. If this
  stays at zero, QA is not probing.
- Defects found after merge in code that QA passed. This is the number that
  says whether PASS means anything.

## Not yet

These are the next plays in the playbook, left for separate changes:
`REVIEW.md` and agent review on PRs, evals that run in CI when `CLAUDE.md`,
`.claude/` or a prompt changes, and a production monitor that writes its
findings as `intent.md`.
