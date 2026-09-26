"""How a tutor holds a conversation, measured turn by turn.

A single-turn eval cannot see the failure this exists to measure. "Explain
more" is only wrong relative to what came before it, and every metric that
scores turns independently reports a system like that as healthy.

Each turn is scored on what a student would notice:

    remembers    it still knows something only an earlier turn established —
                 across the fold, or across a new chat thread
    greeted      it opened with hello, mid-conversation, to a student it
                 is already talking to
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
import re
import time
from pathlib import Path
import sys
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.conversation.store import KINDS
from shared.config import Config
from shared.language import LanguagePolicy
from shared.language.verify import prose_of
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


def _spare_session(config: Config, besides: str) -> str | None:
    """Another chat session belonging to the same student, or None.

    Two foreign keys bind a turn to an existing session, so the second thread
    has to be one the platform already has; an invented id would remember
    nothing and the case would pass by measuring the wrong thing.
    """
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute("SELECT id::text FROM engagement.ai_chat_sessions "
                    "WHERE student_id = (SELECT student_id FROM engagement.ai_chat_sessions "
                    "                    WHERE id = %s::uuid) "
                    "  AND id <> %s::uuid LIMIT 1", (besides, besides))
        row = cur.fetchone()
    return row[0] if row else None


def _forget(config: Config, *session_ids: str) -> None:
    """Start each conversation with nothing remembered."""
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM ai.user_memory WHERE source_conversation_id = ANY(%s::uuid[]) "
                    "AND kind = ANY(%s)", ([s for s in session_ids if s], list(KINDS)))
        conn.commit()


def _ask(url: str, token: str, *, user_id: str, session_id: str, message: str,
         subject_id: str | None) -> dict:
    started = time.perf_counter()
    reply = _post(url, token, user_id=user_id, session_id=session_id,
                  message=message, subject_id=subject_id)
    reply["ms"] = (time.perf_counter() - started) * 1000
    return reply


def _post(url: str, token: str, *, user_id: str, session_id: str, message: str,
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


# Every greeting seen opening a production reply, in the languages the corpus is
# taught in. A turn that answers "hello" is allowed one; a turn that answers a
# question is not, and it is that second kind this counts.
OPENING_GREETING = re.compile(
    r"^\s*\W*(أهلا|أهلاً|مرحبا|مرحباً|السلام\s+عليكم|يا\s+هلا|سلام"
    r"|bonjour|salut|bonsoir|coucou"
    r"|hello|hi\b|hey\b|welcome\b"
    r"|hallo|guten\s+tag)",
    re.IGNORECASE,
)


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

    written = reply.get("response") or ""
    language = LanguagePolicy().resolve_document(prose_of(written)).code if written else None
    right_language = language == turn["language"] if turn.get("language") else True

    # A greeting is not what makes a turn wrong, so it does not fail one: it is
    # counted, because a rate is the only way to tell a prompt that works from a
    # prompt that was merely written.
    greeted = bool(OPENING_GREETING.match(written)) and expect != "chats"

    # What a later turn knows that its own message never said. `about` is the
    # word only an earlier turn can account for, so a reply carrying it is a
    # reply written with the conversation in view — and one that does not is
    # the amnesia this measures, whether it lost it to the fold or to a new
    # chat thread.
    remembers = on_subject if expect == "remembers" else None

    if expect == "answers":
        passed = grounded and on_subject and right_book
    elif expect == "remembers":
        passed = bool(answer) and on_subject
    elif expect == "refuses":
        passed = not grounded
    elif expect == "chats":
        passed = not grounded and bool(answer)
    else:
        passed = bool(answer)
    passed = passed and right_language

    return {"expect": expect, "passed": passed, "greeted": greeted, "grounded": grounded,
            "remembers": remembers,
            "on_subject": on_subject, "right_book": right_book,
            "language": language, "right_language": right_language,
            "citations": len(citations), "error": reply.get("error"),
            "ms": round(reply.get("ms") or 0.0),
            "answer": (reply.get("response") or "")[:110]}


def run(url: str, *, quiet: bool = False) -> dict:
    config = Config.load()
    token = config.secret("AI_WEBHOOK_SHARED_SECRET")
    session_id, user_id = _session(config)
    data = json.loads(CASES.read_text())
    lecture_id = data["lecture_id"]

    # The student's other chat thread. A case that asks what survives a new
    # conversation needs a second one to move to, and the same student behind
    # both: memory that only ever saw one session id cannot be told apart from
    # memory that does not cross sessions at all.
    other_session = _spare_session(config, session_id)

    conversations = []
    for conversation in data["conversations"]:
        _forget(config, session_id, other_session)
        if not quiet:
            print(f"\n  {conversation['name']}")
        turns = []
        speaking_in = session_id
        for index, turn in enumerate(conversation["turns"], start=1):
            if turn.get("session") == "new":
                if not other_session:
                    print("    (skipped: this student has only one chat session)")
                    break
                speaking_in = other_session
            reply = _ask(url, token, user_id=user_id, session_id=speaking_in,
                         message=turn["say"], subject_id=data.get("subject_id"))
            scored = _score(turn, reply, lecture_id)
            scored.update(turn_index=index, say=turn["say"])
            turns.append(scored)
            mark = "OK " if scored["passed"] else "✗  "
            why = "" if scored["passed"] else "  <- " + ", ".join(
                name for name, ok in (("not grounded", scored["grounded"]),
                                      ("off subject", scored["on_subject"]),
                                      ("wrong book", scored["right_book"]),
                                      ("wrong language", scored["right_language"]))
                if not ok) or "  <- unexpected"
            if not quiet:
                print(f"  {mark}{index}. {turn['say'][:58]:58} "
                      f"{scored['expect']:8}{why}")
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
    for expect in ("answers", "remembers", "refuses", "chats", "any"):
        rows = [t for t in turns if t["expect"] == expect]
        if rows:
            print(f"    {expect:8}        {rate(rows):.3f}  ({len(rows)} turns)")
    wrong_book = [t for t in turns if not t["right_book"]]
    if wrong_book:
        print(f"    cited another lecture on {len(wrong_book)} turn(s)")
    taught = [t for t in turns if t["expect"] != "chats"]
    greeted = [t for t in taught if t.get("greeted")]
    if taught:
        print(f"    opened with a greeting   {len(greeted) / len(taught):.3f}  "
              f"({len(greeted)}/{len(taught)} taught turns)")


def stability(rounds: list[dict]) -> None:
    """How often each turn passes, which is what "it is unpredictable" means.

    One pass over the set says whether a turn *can* work. A student meets it
    every day, so what matters is whether it works *every time*. A turn that
    passes four rounds in five is not a turn that works — it is the kind of
    failure that cannot be reproduced on request, and it is the reason this
    service has been hard to trust.
    """
    seen: dict[tuple[str, int], dict] = {}
    for result in rounds:
        for conversation in result["conversations"]:
            for turn in conversation["turns"]:
                key = (conversation["name"], turn["turn_index"])
                row = seen.setdefault(key, {"say": turn["say"], "passed": 0,
                                            "runs": 0, "ms": [], "errors": 0})
                row["passed"] += int(turn["passed"])
                row["runs"] += 1
                row["ms"].append(turn["ms"])
                row["errors"] += int(bool(turn["error"]))

    n = len(rounds)
    always = [r for r in seen.values() if r["passed"] == r["runs"]]
    never = [r for r in seen.values() if r["passed"] == 0]
    flaky = [r for r in seen.values() if 0 < r["passed"] < r["runs"]]
    latencies = sorted(ms for r in seen.values() for ms in r["ms"])
    errors = sum(r["errors"] for r in seen.values())

    def at(fraction):
        return latencies[min(int(len(latencies) * fraction), len(latencies) - 1)]

    print(f"\n  {len(seen)} turns over {n} rounds")
    print(f"    passed every round   {len(always):3}/{len(seen)}")
    print(f"    flaky                {len(flaky):3}/{len(seen)}   "
          f"<- the number that makes it untestable")
    print(f"    failed every round   {len(never):3}/{len(seen)}")
    print(f"    latency   median {at(0.5) / 1000:.1f}s   p95 {at(0.95) / 1000:.1f}s   "
          f"worst {latencies[-1] / 1000:.1f}s")
    print(f"    errors    {errors}")

    for label, rows in (("flaky", flaky), ("failed every round", never)):
        if rows:
            print(f"\n  {label}:")
            for row in sorted(rows, key=lambda r: r["passed"]):
                print(f"    {row['passed']}/{row['runs']}  {row['say'][:64]}")


def compare(before: Path, after: Path) -> None:
    for label, path in (("before", before), ("after", after)):
        print(f"\n=== {label} ===")
        report(json.loads(path.read_text()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--compare", nargs=2, type=Path)
    parser.add_argument("--repeat", type=int, default=1,
                        help="run the whole set N times and report how often "
                             "each turn passes, not just whether it can")
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return 0
    rounds = []
    for round_number in range(1, args.repeat + 1):
        if args.repeat > 1:
            print(f"\n=== round {round_number}/{args.repeat} ===")
        rounds.append(run(args.url, quiet=args.repeat > 1))
        report(rounds[-1])

    if args.repeat > 1:
        stability(rounds)

    if args.json:
        payload = rounds[0] if args.repeat == 1 else {"rounds": rounds}
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
