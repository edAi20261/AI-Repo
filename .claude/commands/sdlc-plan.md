---
description: Have the developer agent write plan.md for an accepted intent
argument-hint: <change id>
---

Change folder: `sdlc/changes/$ARGUMENTS/`.

1. Read `intent.md`. If it is not `Status: accepted`, stop and tell me so.
2. Delegate to the `developer` subagent: "Write plan.md for
   sdlc/changes/$ARGUMENTS/." Pass nothing else. Everything it needs is in the
   folder, and a plan that depends on this conversation cannot be reviewed.
3. Interrogate the plan before showing it to me. Ask what it could break, which
   step is riskiest, and whether every Proof entry is a behaviour rather than an
   implementation detail. Have the developer revise the plan until an engineer
   who never saw this conversation could build from it alone.
4. Show me the plan and the developer's open question. Commit it:
   `plan: $ARGUMENTS (draft)`.

Stop there. I approve by setting `Status: approved` and `Approved by:`. Then
`/sdlc-build $ARGUMENTS` starts the build.
