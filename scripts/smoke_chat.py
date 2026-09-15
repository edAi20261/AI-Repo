"""Walk the chat endpoint through the behaviours that broke, and report.

Not a test suite — a hand-check against a running service, so the answers are
read by a person. Usage:

    .venv/bin/python scripts/smoke_chat.py                    # localhost
    .venv/bin/python scripts/smoke_chat.py https://host       # deployed

Conversational turns cost nothing. Each real question is one LLM call.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
USER = os.environ.get("SMOKE_USER_ID", "00000000-0000-0000-0000-000000000001")

# (label, message, what to look for)
TURNS = [
    ("greeting en", "Hi", "a short greeting, no sources, no 'not in the syllabus'"),
    ("greeting ar", "مرحبا", "a short greeting IN ARABIC"),
    ("thanks ar", "شكراً", "a short reply IN ARABIC"),
    ("meta", "what can you do?", "says what it can help with"),
    ("chitchat", "how are you", "a short reply, not a lesson"),
    ("trap: greeting + question", "hi, can you explain subtraction to me?",
     "MUST be answered as a question, not as a greeting"),
    ("question en", "explain column subtraction", "English, formatted, [n] markers"),
    ("question ar", "اشرح لي عملية الطرح", "ARABIC — this is the reported bug"),
    ("explicit language", "answer in French: what is subtraction", "French"),
    ("off-syllabus", "who won the world cup in 1998?", "declines, cites nothing"),
]


def ask(message: str) -> dict:
    body = json.dumps({
        "user_id": USER, "user_type": "student", "message": message,
    }).encode()
    request = urllib.request.Request(
        f"{BASE}/ai/chat", data=body,
        headers={"content-type": "application/json",
                 "x-service-token": os.environ.get("AI_WEBHOOK_SHARED_SECRET", "")},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read())


def main() -> int:
    print(f"→ {BASE}\n")
    failures = 0
    for label, message, expect in TURNS:
        try:
            result = ask(message)
        except urllib.error.HTTPError as error:
            print(f"✗ {label:28} HTTP {error.code}: {error.read()[:200].decode()}\n")
            failures += 1
            continue
        except Exception as error:  # noqa: BLE001 — a hand-check, report and carry on
            print(f"✗ {label:28} {type(error).__name__}: {error}\n")
            failures += 1
            continue

        reply = result.get("response", "")
        citations = result.get("citations") or []
        markers = reply.count("[")
        print(f"● {label}")
        print(f"  ask      {message}")
        print(f"  expect   {expect}")
        print(f"  lang     {result.get('language')}   tasks={result.get('tasks')}   "
              f"sources={len(citations)}")
        print(f"  reply    {reply[:300]}{'…' if len(reply) > 300 else ''}")
        if citations:
            for c in citations:
                where = c.get("lesson_title") or c.get("lecture_id")
                print(f"           [{c['n']}] {where}")
        if markers and not citations:
            print("  ⚠ the reply has brackets but sent no sources — check this")
        print()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
