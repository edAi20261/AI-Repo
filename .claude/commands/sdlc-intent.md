---
description: Capture an idea, bug or incident as sdlc/changes/<id>/intent.md
argument-hint: <what you want, in your own words>
---

Turn this into an intent for the squad: $ARGUMENTS

1. Brainstorm with me until the idea is concrete. Ask what an analyst would ask:
   who is affected, what they cannot do today, what better looks like, what is
   out of scope, and how we will know it worked. Ask a few questions at a time,
   not a questionnaire. Read the code only as far as it takes to ask good
   questions. Do not design the solution.
2. Choose a short kebab-case id and write `sdlc/changes/<id>/intent.md` from
   `sdlc/templates/intent.md`, in my words rather than yours. Leave
   `Status: draft`.
3. Show me the file and ask what you got wrong. Correct it until I say it is
   right, then commit it alone: `intent: <id>`.

Stop there. When I set `Status: accepted`, the next step is `/sdlc-plan <id>`.
