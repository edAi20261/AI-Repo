"""LLM prompt: choose a topic's prerequisites from the subject catalog."""

DERIVE_DEPS_PROMPT = """\
You are given a target skill and a list of candidate skills from the same
subject. Choose the candidates that are genuine PREREQUISITES — skills a student
must already have to learn the target. Pick from anywhere in the list, not just
nearby ones. Choose none if there are no real prerequisites.

Use only labels from the candidate list, verbatim. Return a JSON array of objects:
{{"prereq": "<exact candidate label>", "strength": "required" | "recommended",
  "rationale": "<short reason>"}}

Target skill: {topic}

Candidate skills:
{candidates}
"""
