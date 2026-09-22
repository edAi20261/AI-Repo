"""How a tutor holds a conversation, measured turn by turn.

A single-turn eval cannot see the failure this exists to measure. "Explain
more" is only wrong relative to what came before it, and every metric that
scores turns independently reports a system like that as healthy.

Each turn is scored on what a student would notice:

    grounded     it answered from the syllabus, with citations
    on subject   the answer still concerns what the conversation is about
    right book   the sources come from the lesson in play, not another corpus
    refused      it declined what the syllabus cannot answer
    no search    it chatted back instead of searching for "hello"

Run against a live service so it measures the product, not a stand-in:

    .venv/bin/python scripts/eval_conversations.py
    .venv/bin/python scripts/eval_conversations.py --json before.json
    .venv/bin/python scripts/eval_conversations.py --compare before.json after.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.config import Config
from shared.postgres import connect

CASES = Path(__file__).with_suffix(".json")
DEFAULT_URL = "http://localhost:8000/ai/chat"
TIMEOUT = 180


def _session(config: Config) -> tuple[str, str]:
    """A real chat session, because evidence is keyed on one.

    Two foreign keys bind turn memory to an existing session and user, so an
    invented id degrades to remembering nothing — and a run against one would
    score the system as though every turn were the first.
    """
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute("SELECT id::text, student_id::text FROM engagement.ai_chat_sessions "
                    "WHERE student_id IS NOT NULL LIMIT 1")
        row = cur.fetchone()
    if not row:
        raise SystemExit("no chat session exists to run against; open the app once")
    return row


def _forget(config: Config, session_id: str) -> None:
    """Start each conversation with nothing remembered."""
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM ai.user_memory WHERE source_conversation_id = %s::uuid "
                    "AND kind = 'turn_evidence'", (session_id,))
        conn.commit()


def _ask(url: str, token: str, *, user_id: str, session_id: str, message: str,
         subject_id: str | None) -> dict:
    body = {"user_id": user_id, "user_type": "student", "session_id": session_id,
            "message": message}
    if subject_id:
        body["subject_id"] = subject_id
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"content-type": "application/json", "x-service-token": token})
    try:
        return json.loads(urllib.request.urlopen(request, timeout=TIMEOUT).read())
    except urllib.error.HTTPError as error:
        return {"error": f"HTTP {error.code}", "response": "", "citations": []}
    except Exception as error:  # noqa: BLE001 — a dead turn is a failed turn, not a dead run
        return {"error": type(error).__name__, "response": "", "citations": []}


def _score(turn: dict, reply: dict, lecture_id: str) -> dict:
    """What this turn got right, as a student would judge it."""
    citations = reply.get("citations") or []
    answer = (reply.get("response") or "").lower()
    expect = turn["expect"]
    wanted = (turn.get("about") or "").lower()

    grounded = bool(citations)
    on_subject = wanted in answer if wanted else True
    # A citation from another lecture is the failure that looks like nothing:
    # a fluent English answer with sources from an Arabic lesson.
    right_book = all(str(c.get("lecture_id") or lecture_id) == lecture_id
                     for c in citations) if citations else True

    if expect == "answers":
        passed = grounded and on_subject and right_book
    elif expect == "refuses":
        passed = not grounded
    else:                                   # chats
        passed = not grounded and bool(answer)

    return {"expect": expect, "passed": passed, "grounded": grounded,
            "on_subject": on_subject, "right_book": right_book,
            "citations": len(citations), "error": reply.get("error"),
            "answer": (reply.get("response") or "")[:110]}


def run(url: str) -> dict:
    config = Config.load()
    token = config.secret("AI_WEBHOOK_SHARED_SECRET")
    session_id, user_id = _session(config)
    data = json.loads(CASES.read_text())
    lecture_id = data["lecture_id"]

    conversations = []
    for conversation in data["conversations"]:
        _forget(config, session_id)
        print(f"\n  {conversation['name']}")
        turns = []
        for index, turn in enumerate(conversation["turns"], start=1):
            reply = _ask(url, token, user_id=user_id, session_id=session_id,
                         message=turn["say"], subject_id=data.get("subject_id"))
            scored = _score(turn, reply, lecture_id)
            scored.update(turn_index=index, say=turn["say"])
            turns.append(scored)
            mark = "OK " if scored["passed"] else "✗  "
            why = "" if scored["passed"] else "  <- " + ", ".join(
                name for name, ok in (("not grounded", scored["grounded"]),
                                      ("off subject", scored["on_subject"]),
                                      ("wrong book", scored["right_book"]))
                if not ok) or "  <- unexpected"
            print(f"  {mark}{index}. {turn['say'][:58]:58} {scored['expect']:8}{why}")
        conversations.append({"name": conversation["name"], "turns": turns})
    return {"conversations": conversations}


def report(result: dict) -> None:
    turns = [t for c in result["conversations"] for t in c["turns"]]
    total = len(turns)
    passed = sum(1 for t in turns if t["passed"])
    first = [t for t in turns if t["turn_index"] == 1]
    later = [t for t in turns if t["turn_index"] > 1]

    def rate(rows):
        return (sum(1 for r in rows if r["passed"]) / len(rows)) if rows else 0.0

    print(f"\n  turns {total}   passed {passed}   {passed / total:.3f}")
    print(f"    opening turns   {rate(first):.3f}  ({len(first)} turns)")
    print(f"    later turns     {rate(later):.3f}  ({len(later)} turns)   "
          f"<- the number this eval exists for")
    for expect in ("answers", "refuses", "chats"):
        rows = [t for t in turns if t["expect"] == expect]
        if rows:
            print(f"    {expect:8}        {rate(rows):.3f}  ({len(rows)} turns)")
    wrong_book = [t for t in turns if not t["right_book"]]
    if wrong_book:
        print(f"    cited another lecture on {len(wrong_book)} turn(s)")


def compare(before: Path, after: Path) -> None:
    for label, path in (("before", before), ("after", after)):
        print(f"\n=== {label} ===")
        report(json.loads(path.read_text()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--compare", nargs=2, type=Path)
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return 0
    result = run(args.url)
    report(result)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
