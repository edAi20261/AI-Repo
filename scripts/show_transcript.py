"""Read back what students actually asked and what the tutor actually answered.

The service logs what it decided — the language, the chunks it matched, whether
the answer was grounded — but not the answer text, because a log is not the
place to keep a transcript. The platform already keeps one:
``engagement.ai_chat_messages``, written by the backend on every turn, with the
AI service's own metadata attached.

This is the view to open when someone says "it answered about the wrong thing":
the question, the reply, and what the service thought it was doing, in order.

    .venv/bin/python scripts/show_transcript.py                  # last 10 turns
    .venv/bin/python scripts/show_transcript.py --limit 30
    .venv/bin/python scripts/show_transcript.py --session <uuid>
    .venv/bin/python scripts/show_transcript.py --full           # whole answers

Read-only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from agent.config import Config
from shared.postgres import connect

WIDTH = 96


def _wrap(text: str, indent: str, full: bool) -> str:
    body = " ".join((text or "").split())
    if not full and len(body) > 400:
        body = body[:400] + " …"
    out, line = [], ""
    for word in body.split(" "):
        if len(line) + len(word) + 1 > WIDTH:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    out.append(line)
    return f"\n{indent}".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10, help="turns to show")
    parser.add_argument("--session", help="one conversation only")
    parser.add_argument("--full", action="store_true", help="do not truncate answers")
    args = parser.parse_args()

    where, params = "", []
    if args.session:
        where = "WHERE session_id = %s::uuid"
        params.append(args.session)

    config = Config.load()
    with connect(config, connect_timeout=15, statement_timeout_ms=0) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT session_id::text, sender, created_at, message_text, metadata "
            f"FROM engagement.ai_chat_messages {where} "
            "ORDER BY created_at DESC LIMIT %s",
            (*params, args.limit * 2))
        rows = list(reversed(cur.fetchall()))

    if not rows:
        print("no messages found")
        return 0

    session = None
    for session_id, sender, created, text, metadata in rows:
        if session_id != session:
            session = session_id
            print(f"\n{'=' * WIDTH}\nconversation {session_id}")
        if sender == "student":
            print(f"\n  {created:%H:%M:%S}  asked")
            print(f"    {_wrap(text, '    ', True)}")
        else:
            meta = metadata or {}
            print(f"  {created:%H:%M:%S}  answered")
            print(f"    {_wrap(text, '    ', args.full)}")
            # What the service thought it was doing, straight from the reply the
            # backend stored — the same fields the API returned.
            facts = [f"tasks={meta.get('tasks')}"]
            citations = meta.get("citations")
            if citations is not None:
                facts.append(f"sources={len(citations)}")
                named = [c.get("lesson_title") or c.get("lecture_id")
                         for c in citations[:3] if isinstance(c, dict)]
                if any(named):
                    facts.append("cited=" + ", ".join(str(n) for n in named))
            prereqs = meta.get("prereqTopics")
            if prereqs:
                facts.append(f"prereqs={len(prereqs)}")
            print(f"    [{'  '.join(facts)}]")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
