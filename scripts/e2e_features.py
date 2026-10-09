"""Feature-level end-to-end checks against a deployed stack, once and under load.

Unit tests prove a function; they never proved that the platform reaches this
service, that the reply survives the backend, or that thirty students at once
are answered. This drives the real routes with real models and reports what came
back, so "is it wired" has an answer that is not a reading of the code.

Two vantage points, because the AI service is only reachable inside the VPC:

  backend   from anywhere: register students, chat through the platform the way
            the frontend does. Writes the students it made to --students-out.
  direct    from inside the VPC: every AI route the platform does not proxy yet.
  stress-backend / stress-direct   the same features, N students at once.

    python scripts/e2e_features.py backend --api https://.../api --students 30
    python scripts/e2e_features.py direct --ai http://internal-alb --users-file u.json
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

SCIENCE_EN = {
    "subject_id": "25e488c1-9582-40f8-9656-ccbd229a8656",
    "lecture_id": "9d3f1b20-0002-4a00-9000-000000000006",
    "topics": ["5e26e122-17aa-40c6-b8bc-3fdb311bd3c2", "49605bfb-cfd4-423a-9e33-0ebd0756c02a",
               "58a4091c-d74f-4e36-8b51-bd92067743fc", "86e0323a-60f6-4a60-860b-d3894c1a662c"],
}
SCIENCE_AR = {
    "subject_id": "7b1e5c40-0a6e-4d1a-9f3c-5d2a91c00007",
    "lecture_id": "9d3f1b20-0002-4a00-9000-000000000014",
    "topics": ["4e491ad3-0d7f-4c5e-8365-9f5f4a47fcd6", "e1af75dd-9de4-421f-94ec-b00c1119d069"],
}
CURRICULUM_NATIONAL = "72ab028e-b404-40dc-81e7-19cbc7b29579"
TURN_TIMEOUT = 180.0


@dataclass
class Check:
    feature: str
    endpoint: str
    passed: bool
    status: int | None
    seconds: float
    detail: str = ""
    evidence: Any = None


@dataclass
class Report:
    mode: str
    checks: list[Check] = field(default_factory=list)
    load: dict[str, Any] = field(default_factory=dict)

    def add(self, check: Check) -> Check:
        self.checks.append(check)
        mark = "PASS" if check.passed else "FAIL"
        print(f"[{mark}] {check.feature:<44} {check.status!s:>4} {check.seconds:6.1f}s  "
              f"{check.detail}", flush=True)
        return check


def _clip(value: Any, limit: int = 400) -> Any:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "…"


async def _timed(call: Awaitable[httpx.Response]) -> tuple[httpx.Response | None, float, str]:
    started = time.perf_counter()
    try:
        response = await call
        return response, time.perf_counter() - started, ""
    except httpx.HTTPError as exc:
        return None, time.perf_counter() - started, f"{type(exc).__name__}: {exc}"


def _body(response: httpx.Response | None) -> Any:
    if response is None:
        return None
    try:
        return response.json()
    except ValueError:
        return response.text[:400]


async def check(report: Report, feature: str, endpoint: str,
                call: Awaitable[httpx.Response],
                expect: Callable[[int, Any], tuple[bool, str]]) -> tuple[Check, Any]:
    response, seconds, error = await _timed(call)
    if response is None:
        return report.add(Check(feature, endpoint, False, None, seconds, error)), None
    body = _body(response)
    try:
        passed, detail = expect(response.status_code, body)
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        passed, detail = False, f"unexpected body: {type(exc).__name__}: {exc}"
    return report.add(Check(feature, endpoint, passed, response.status_code, seconds, detail,
                            _clip(body))), body


def ok(predicate: Callable[[Any], bool], why: Callable[[Any], str] = lambda b: ""):
    def judge(status: int, body: Any) -> tuple[bool, str]:
        if status >= 400:
            return False, f"HTTP {status}: {_clip(body, 200)}"
        return bool(predicate(body)), why(body)
    return judge


def status_is(expected: int, why: str = ""):
    def judge(status: int, body: Any) -> tuple[bool, str]:
        return status == expected, why or f"expected {expected}, got {status}"
    return judge


class Platform:
    """The backend, called the way the frontend calls it."""

    def __init__(self, client: httpx.AsyncClient, api: str) -> None:
        self.client, self.api = client, api.rstrip("/")

    async def register(self, tag: str, index: int) -> dict[str, Any]:
        email = f"e2e-{tag}-{index:02d}@test.eduai"
        password = f"E2e-{tag}-pass!"
        body = {"role": "student", "email": email, "password": password,
                "firstName": "E2E", "lastName": f"Student {index:02d}",
                "gradeLevel": "GRADE_7", "curriculumId": CURRICULUM_NATIONAL}
        response = await self.client.post(f"{self.api}/auth/register", json=body)
        if response.status_code == 409:
            response = await self.client.post(f"{self.api}/auth/login",
                                              json={"email": email, "password": password})
        response.raise_for_status()
        data = response.json()
        return {"email": email, "user_id": data["user"]["id"], "token": data["accessToken"]}

    def headers(self, student: dict[str, Any]) -> dict[str, str]:
        return {"authorization": f"Bearer {student['token']}"}

    def open_session(self, student, subject_id: str | None):
        return self.client.post(f"{self.api}/ai-chat/sessions", headers=self.headers(student),
                                json={"subjectId": subject_id} if subject_id else {})

    def send(self, student, session_id: str, message: str, **extra):
        return self.client.post(f"{self.api}/ai-chat/sessions/{session_id}/messages",
                                headers=self.headers(student), json={"message": message, **extra},
                                timeout=TURN_TIMEOUT)

    def history(self, student, session_id: str):
        return self.client.get(f"{self.api}/ai-chat/sessions/{session_id}/messages",
                               headers=self.headers(student))


def _replied(body: Any) -> bool:
    return isinstance(body, dict) and len((body.get("response") or "").strip()) > 20


def _declined(body: Any) -> bool:
    text = (body.get("response") or "").lower() if isinstance(body, dict) else ""
    return any(m in text for m in ("don't seem to cover", "do not cover", "can only help",
                                   "not covered", "isn't covered"))


def _arabic(text: str) -> bool:
    return sum("؀" <= ch <= "ۿ" for ch in text) > len(text) * 0.3


async def run_backend(args) -> Report:
    report = Report("backend")
    async with httpx.AsyncClient(timeout=60) as client:
        platform = Platform(client, args.api)
        await check(report, "Backend liveness", "GET /api/health",
                    client.get(f"{platform.api}/health"), ok(lambda b: b.get("status") == "ok"))

        students = []
        for index in range(args.students):
            students.append(await platform.register(args.tag, index))
        report.add(Check("Register test students", "POST /api/auth/register", True, 201, 0,
                         f"{len(students)} students"))
        _write_json(args.students_out, [{k: s[k] for k in ("email", "user_id")}
                                        for s in students])

        student, other = students[0], students[1]
        _, session = await check(
            report, "Open chat session (EN science)", "POST /api/ai-chat/sessions",
            platform.open_session(student, SCIENCE_EN["subject_id"]),
            ok(lambda b: bool(b.get("id"))))
        sid = session["id"]

        await check(
            report, "Tutor explains from the syllabus (EN)", "POST /api/ai-chat/sessions/:id/messages",
            platform.send(student, sid, "What are the main parts of an atom?"),
            ok(lambda b: _replied(b) and b["language"] == "en",
               lambda b: f"lang={b.get('language')} tasks={b.get('tasks')} "
                         f"prereqs={len(b.get('prereqTopics') or [])}"))
        await check(
            report, "Citations reach the frontend", "POST /api/ai-chat/sessions/:id/messages",
            platform.send(student, sid, "Where is the mass of an atom concentrated?"),
            ok(lambda b: bool(b.get("citations")),
               lambda b: f"citations={len(b.get('citations') or [])} (frontend reads `citations`)"))
        await check(
            report, "Follow-up uses conversation memory", "POST /api/ai-chat/sessions/:id/messages",
            platform.send(student, sid, "Explain that more simply, like I'm 10."),
            ok(_replied, lambda b: _clip(b.get("response", ""), 120)))
        await check(
            report, "Practice quiz generation", "POST /api/ai-chat/sessions/:id/messages",
            platform.send(student, sid, "Give me a 3 question quiz on atomic structure."),
            ok(lambda b: bool(b.get("assessment")),
               lambda b: f"tasks={b.get('tasks')} assessment_keys={list((b.get('assessment') or {}).keys())[:5]}"))
        await check(
            report, "Answering the open quiz is graded", "POST /api/ai-chat/sessions/:id/messages",
            platform.send(student, sid, "My answers: 1) protons and neutrons 2) electrons 3) the nucleus"),
            ok(lambda b: bool(b.get("gradeResult")),
               lambda b: f"tasks={b.get('tasks')} grade_keys={list((b.get('gradeResult') or {}).keys())[:5]}"))
        await check(
            report, "Visual (mind map) generation", "POST /api/ai-chat/sessions/:id/messages",
            platform.send(student, sid, "Draw me a mind map of the structure of the atom."),
            ok(lambda b: bool(b.get("visuals")),
               lambda b: f"tasks={b.get('tasks')} visuals={_clip(b.get('visuals'), 160)}"))
        await check(
            report, "Off-syllabus question is declined", "POST /api/ai-chat/sessions/:id/messages",
            platform.send(student, sid, "Who won the 2022 football World Cup final?"),
            ok(lambda b: "argentina" not in b.get("response", "").lower(),
               lambda b: _clip(b.get("response", ""), 140)))
        await check(
            report, "Lesson-scoped question (lectureIds)", "POST /api/ai-chat/sessions/:id/messages",
            platform.send(student, sid, "What is valency?", lectureIds=[SCIENCE_EN["lecture_id"]]),
            ok(_replied, lambda b: _clip(b.get("response", ""), 120)))

        _, ar_session = await check(
            report, "Open chat session (AR science)", "POST /api/ai-chat/sessions",
            platform.open_session(student, SCIENCE_AR["subject_id"]), ok(lambda b: bool(b.get("id"))))
        await check(
            report, "Tutor answers in Arabic", "POST /api/ai-chat/sessions/:id/messages",
            platform.send(student, ar_session["id"], "ما هي مكونات الذرة؟"),
            ok(lambda b: b.get("language") == "ar" and _arabic(b.get("response", "")),
               lambda b: f"lang={b.get('language')} {_clip(b.get('response', ''), 100)}"))
        await check(
            report, "Chat history is persisted", "GET /api/ai-chat/sessions/:id/messages",
            platform.history(student, sid),
            ok(lambda b: (b.get("meta") or {}).get("total", len(b.get("data") or [])) >= 10,
               lambda b: f"messages={(b.get('meta') or {}).get('total', len(b.get('data') or []))}"))
        await check(
            report, "Another student cannot read the session", "GET /api/ai-chat/sessions/:id/messages",
            platform.history(other, sid), status_is(403))
        await check(
            report, "Unauthenticated chat is refused", "POST /api/ai-chat/sessions",
            client.post(f"{platform.api}/ai-chat/sessions", json={}), status_is(401))
    return report


class AI:
    def __init__(self, client: httpx.AsyncClient, base: str, token: str) -> None:
        self.client, self.base = client, base.rstrip("/")
        self.auth = {"x-service-token": token}

    def get(self, path: str, **kw):
        return self.client.get(self.base + path, headers=self.auth, **kw)

    def post(self, path: str, **kw):
        return self.client.post(self.base + path, headers=self.auth, **kw)

    def delete(self, path: str, **kw):
        return self.client.delete(self.base + path, headers=self.auth, **kw)


def _submissions(lesson: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"question": "What particles are found in the nucleus of an atom?",
         "student_answer": "Protons and neutrons", "correct_answer": "Protons and neutrons",
         "topic_id": lesson["topics"][0], "lecture_id": lesson["lecture_id"],
         "question_id": str(uuid.uuid4()), "difficulty": "easy", "type": "short_answer"},
        {"question": "Is salt water an element, a compound or a mixture?",
         "student_answer": "An element", "correct_answer": "A mixture",
         "topic_id": lesson["topics"][1], "lecture_id": lesson["lecture_id"],
         "question_id": str(uuid.uuid4()), "difficulty": "medium", "type": "short_answer"},
        {"question": "Oxygen has valency 2. True or false?", "student_answer": "True",
         "correct_answer": "True", "known_score": 1.0, "topic_id": lesson["topics"][2],
         "lecture_id": lesson["lecture_id"], "question_id": str(uuid.uuid4()),
         "type": "true_false"},
    ]


def _homework_png(answers: tuple[str, ...] = ()) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6.5, 4), dpi=150)
    fig.text(0.05, 0.9, "Science Homework - Grade 7", fontsize=15, weight="bold")
    lines = ["1. What are the three particles that make up an atom?",
             "2. Define an isotope and give one example.",
             "3. Classify: water, oxygen, air (element / compound / mixture).",
             "4. What is the valency of sodium?"]
    for i, line in enumerate(lines):
        answer = f"   Answer: {answers[i]}" if i < len(answers) and answers[i] else ""
        fig.text(0.05, 0.75 - i * 0.15, line + answer, fontsize=10 if answer else 12)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return buffer.getvalue()


async def _stream(ai: AI, payload: dict[str, Any]) -> httpx.Response:
    events, result, status = [], None, None
    async with ai.client.stream("POST", ai.base + "/ai/chat/stream", headers=ai.auth,
                                json=payload, timeout=TURN_TIMEOUT) as response:
        status = response.status_code
        name = None
        async for line in response.aiter_lines():
            if line.startswith("event: "):
                name = line[7:]
                events.append(name)
            elif line.startswith("data: ") and name in ("result", "error"):
                result = {"event": name, "data": json.loads(line[6:])}
    body = {"events": events, "final": result}
    return httpx.Response(status, json=body)


async def run_direct(args) -> Report:
    report = Report("direct")
    users = _users(args)
    ids = [u["user_id"] for u in users]
    student, parent_view = ids[0], ids[0]
    token = os.environ["AI_WEBHOOK_SHARED_SECRET"]
    async with httpx.AsyncClient(timeout=TURN_TIMEOUT) as client:
        ai = AI(client, args.ai, token)
        await check(report, "AI liveness", "GET /health", client.get(ai.base + "/health"),
                    ok(lambda b: b.get("status") == "ok", lambda b: f"env={b.get('environment')}"))
        await check(report, "AI readiness", "GET /ready", client.get(ai.base + "/ready"),
                    ok(lambda b: b.get("graph_ready") is True, lambda b: f"in_flight={b.get('in_flight')}"))
        await check(report, "Service token is enforced", "POST /ai/chat",
                    client.post(ai.base + "/ai/chat", json={}), status_is(401))
        await check(report, "Unknown field rejected (422 envelope)", "POST /ai/chat",
                    ai.post("/ai/chat", json={"user_id": student, "user_type": "student",
                                              "message": "hi", "bogus": 1}),
                    lambda s, b: (s == 422 and "error" in b, f"status={s}"))

        chat = {"user_id": student, "user_type": "student", "session_id": str(uuid.uuid4()),
                "subject_id": SCIENCE_EN["subject_id"],
                "message": "What is the difference between an element and a compound?"}
        await check(report, "Chat returns citations + usage", "POST /ai/chat",
                    ai.post("/ai/chat", json=chat),
                    ok(lambda b: b.get("response") and b.get("citations") and "usage" in b,
                       lambda b: f"citations={len(b.get('citations') or [])} usage={b.get('usage')}"))
        await check(report, "Chat stream: progress then result", "POST /ai/chat/stream",
                    _stream(ai, {**chat, "message": "What is an isotope?"}),
                    ok(lambda b: "progress" in b["events"] and (b["final"] or {}).get("event") == "result",
                       lambda b: f"events={len(b['events'])} final={(b['final'] or {}).get('event')}"))
        await check(report, "Teacher pathway chat", "POST /ai/chat",
                    ai.post("/ai/chat", json={"user_id": str(uuid.uuid4()), "user_type": "teacher",
                                              "subject_id": SCIENCE_EN["subject_id"],
                                              "message": "Write 3 exam questions on isotopes for my class."}),
                    ok(lambda b: bool(b.get("response")),
                       lambda b: f"pathway={b.get('pathway')} tasks={b.get('tasks')}"))
        await check(report, "Parent pathway chat", "POST /ai/chat",
                    ai.post("/ai/chat", json={"user_id": str(uuid.uuid4()), "user_type": "parent",
                                              "subject_id": SCIENCE_EN["subject_id"],
                                              "message": "What is my child learning about atoms?"}),
                    ok(lambda b: bool(b.get("response")),
                       lambda b: f"pathway={b.get('pathway')} tasks={b.get('tasks')}"))

        await check(report, "Grading moves mastery", "POST /ai/grade",
                    ai.post("/ai/grade", json={"user_id": student, "billed_user_id": student,
                                               "subject_id": SCIENCE_EN["subject_id"], "kind": "quiz",
                                               "submissions": _submissions(SCIENCE_EN)}),
                    ok(lambda b: b.get("mastery_recorded") and b.get("mastery_events", 0) >= 1
                       and not b.get("degraded"),
                       lambda b: f"events={b.get('mastery_events')} topics={b.get('topics_updated')} "
                                 f"degraded={b.get('degraded')} scores="
                                 f"{[g.get('score') for g in b.get('graded', [])]}"))
        await check(report, "Learner summary (homepage)", "GET /ai/learners/{id}/summary",
                    ai.get(f"/ai/learners/{student}/summary",
                           params={"subject_ids": [SCIENCE_EN["subject_id"]]}),
                    ok(lambda b: b.get("available") and not b.get("degraded"),
                       lambda b: f"available={b.get('available')} weak={len(b.get('weaknesses') or [])} "
                                 f"recs={len(b.get('recommendations') or [])}"))
        await check(report, "Student progress report", "GET /ai/reports/{id}",
                    ai.get(f"/ai/reports/{student}",
                           params={"billed_user_id": student, "subject_id": SCIENCE_EN["subject_id"]}),
                    ok(lambda b: b.get("available") and b.get("narrative"),
                       lambda b: f"persisted={b.get('persisted')} visual={bool(b.get('visual'))} "
                                 f"{_clip(b.get('narrative', ''), 100)}"))
        await check(report, "Stored report read-back", "GET /ai/reports/{id}/latest",
                    ai.get(f"/ai/reports/{student}/latest"),
                    ok(lambda b: bool(b.get("report_id")), lambda b: f"stale={b.get('stale')}"))
        await check(report, "Parent-scope report", "GET /ai/reports/{id}?scope=parent",
                    ai.get(f"/ai/reports/{parent_view}",
                           params={"billed_user_id": parent_view, "scope": "parent"}),
                    ok(lambda b: bool(b.get("narrative")), lambda b: _clip(b.get("narrative", ""), 100)))

        class_id = str(uuid.uuid4())
        roster = {"user_ids": ids, "subject_ids": [SCIENCE_EN["subject_id"]], "class_id": class_id}
        await check(report, "Class analytics (teacher)", "POST /ai/classes/analytics",
                    ai.post("/ai/classes/analytics", json=roster),
                    ok(lambda b: not b.get("degraded"),
                       lambda b: f"available={b.get('available')} at_risk={len(b.get('at_risk') or [])} "
                                 f"not_started={len(b.get('not_started') or [])}"))
        await check(report, "Class report written", "POST /ai/classes/report",
                    ai.post("/ai/classes/report", json={**roster, "billed_user_id": student}),
                    ok(lambda b: b.get("narrative") and b.get("persisted"),
                       lambda b: f"persisted={b.get('persisted')} {_clip(b.get('narrative', ''), 100)}"))
        await check(report, "Class report read-back", "GET /ai/classes/{id}/report/latest",
                    ai.get(f"/ai/classes/{class_id}/report/latest"),
                    ok(lambda b: bool(b.get("report_id"))))

        await check(report, "Topic suggestions for questions", "POST /ai/topics/suggest",
                    ai.post("/ai/topics/suggest", json={
                        "subject_id": SCIENCE_EN["subject_id"],
                        "questions": [{"text": "What is the charge of a neutron?"},
                                      {"text": "Give two examples of mixtures."}]}),
                    ok(lambda b: b.get("suggestions") and not b.get("degraded"),
                       lambda b: _clip([[t.get("label") for t in (s.get("topics") or [])[:2]]
                                        for s in b.get("suggestions", [])], 160)))
        await check(report, "Homework photo capture", "POST /ai/homework/capture",
                    ai.post("/ai/homework/capture", params={"billed_user_id": student},
                            files={"image": ("hw.png", _homework_png(), "image/png")}),
                    ok(lambda b: len(b.get("questions") or []) >= 3,
                       lambda b: f"questions={len(b.get('questions') or [])} "
                                 f"unreadable={b.get('unreadable')}"))
        await check(report, "Homework photo marked", "POST /ai/homework/mark",
                    ai.post("/ai/homework/mark",
                            params={"user_id": student, "billed_user_id": student,
                                    "subject_id": SCIENCE_EN["subject_id"]},
                            files={"image": ("hw.png", _homework_png(
                                ("protons, neutrons and electrons", "", "", "2")), "image/png")}),
                    ok(lambda b: any(q.get("status") == "marked" for q in b.get("questions") or [])
                       and any(q.get("status") == "blank" for q in b.get("questions") or [])
                       and not b.get("degraded"),
                       lambda b: f"statuses={[q.get('status') for q in b.get('questions') or []]} "
                                 f"needs_review={(b.get('summary') or {}).get('needs_review')}"))
        await check(report, "Homework capture rejects non-image", "POST /ai/homework/capture",
                    ai.post("/ai/homework/capture", params={"billed_user_id": student},
                            files={"image": ("a.txt", b"hello", "text/plain")}), status_is(422))

        await check(report, "Memory read-back", "GET /ai/memory/{id}",
                    ai.get(f"/ai/memory/{student}"),
                    ok(lambda b: isinstance(b.get("conversations"), int),
                       lambda b: f"conversations={b.get('conversations')} (memory is keyed to "
                                 f"backend chat sessions; direct turns are not remembered)"))
        await check(report, "Credits plans", "GET /ai/credits/plans", ai.get("/ai/credits/plans"),
                    lambda s, b: (s in (200, 503), f"status={s} (503 = credits disabled by config)"))

        missing = str(uuid.uuid4())
        _, job = await check(report, "Ingestion trigger accepted", "POST /ai/ingest/lesson",
                             ai.post("/ai/ingest/lesson", json={"lesson_id": missing}),
                             lambda s, b: (s == 202 and bool(b.get("job_id")), f"status={s}"))
        await asyncio.sleep(8)
        if job and job.get("job_id"):
            await check(report, "Ingestion job status (unknown lesson fails)", "GET /ai/ingest/{job_id}",
                        ai.get(f"/ai/ingest/{job['job_id']}"),
                        ok(lambda b: b.get("status") in ("failed", "running", "queued"),
                           lambda b: f"status={b.get('status')} errors={_clip(b.get('errors'), 120)}"))
        await check(report, "Ingestion job list", "GET /ai/ingest", ai.get("/ai/ingest"),
                    ok(lambda b: isinstance(b, list), lambda b: f"jobs={len(b)}"))
        for path in ("/ai/personas/" + student, "/ai/voice/speak"):
            method = ai.get if "personas" in path else ai.post
            await check(report, f"Deployed? {path.split('/')[2]}", path,
                        method(path, **({} if "personas" in path else {"json": {}})),
                        lambda s, b: (s != 404, f"status={s} (404 = route not deployed)"))
    return report


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)


def _users(args) -> list[dict[str, Any]]:
    if args.users_file:
        with open(args.users_file) as handle:
            return json.load(handle)
    return json.loads(args.users)


def _ok(status: int, body: Any) -> bool:
    return status == 200


def _chat_replied(status: int, body: Any) -> bool:
    return status in (200, 201) and _replied(body)


def _report_written(status: int, body: Any) -> bool:
    return status == 200 and bool(body.get("narrative"))


def _latency(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {}
    ordered = sorted(samples)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    return {"n": len(samples), "p50": round(statistics.median(ordered), 1),
            "p95": round(p95, 1), "max": round(ordered[-1], 1)}


async def _burst(name: str, calls: list[Callable[[], Awaitable[httpx.Response]]],
                 good: Callable[[int, Any], bool]) -> dict[str, Any]:
    async def one(make):
        response, seconds, error = await _timed(make())
        status = response.status_code if response is not None else None
        body = _body(response)
        passed = response is not None and good(status, body)
        return passed, seconds, status, error or ("" if passed else _clip(body, 300)), body

    started = time.perf_counter()
    results = await asyncio.gather(*(one(c) for c in calls))
    wall = time.perf_counter() - started
    failures = [r for r in results if not r[0]]
    statuses: dict[str, int] = {}
    for r in results:
        statuses[str(r[2])] = statuses.get(str(r[2]), 0) + 1
    declined = sum(1 for r in results if r[0] is not None and r[4] and _declined(r[4]))
    summary = {"feature": name, "concurrent": len(calls), "ok": len(results) - len(failures),
               "declined_as_off_syllabus": declined,
               "failed": len(failures), "wall_s": round(wall, 1), "statuses": statuses,
               "latency_ok": _latency([r[1] for r in results if r[0]]),
               "sample_errors": list({r[3] for r in failures})[:3]}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def _burst_checks(report: Report, bursts: list[dict[str, Any]]) -> None:
    for burst in bursts:
        report.add(Check(burst["feature"], f"{burst['concurrent']} at once",
                         burst["failed"] == 0, None, burst["wall_s"],
                         f"ok={burst['ok']}/{burst['concurrent']} "
                         f"latency={burst['latency_ok']} statuses={burst['statuses']}"))


async def run_stress_backend(args) -> Report:
    report = Report("stress-backend")
    async with httpx.AsyncClient(timeout=TURN_TIMEOUT,
                                 limits=httpx.Limits(max_connections=200)) as client:
        platform = Platform(client, args.api)
        students = [await platform.register(args.tag, i) for i in range(args.students)]
        sessions = await asyncio.gather(*(platform.open_session(s, SCIENCE_EN["subject_id"])
                                          for s in students))
        sids = [r.json()["id"] for r in sessions]
        questions = ["What is an atom made of?", "What is an isotope?",
                     "What is the difference between a compound and a mixture?",
                     "How are elements arranged in the periodic table?", "What is valency?"]
        bursts = [
            await _burst("chat: first question, all students at once", [
                (lambda s=s, sid=sid, i=i: platform.send(s, sid, questions[i % len(questions)]))
                for i, (s, sid) in enumerate(zip(students, sids))], _chat_replied),
            await _burst("chat: follow-up, all students at once", [
                (lambda s=s, sid=sid: platform.send(s, sid, "Can you give me an example of that?"))
                for s, sid in zip(students, sids)], _chat_replied),
            await _burst("chat: quiz request, all students at once", [
                (lambda s=s, sid=sid: platform.send(s, sid, "Quiz me with 3 questions on this."))
                for s, sid in zip(students, sids)], _chat_replied),
            await _burst("chat history, all students at once", [
                (lambda s=s, sid=sid: platform.history(s, sid)) for s, sid in zip(students, sids)],
                lambda s, b: s == 200),
        ]
        report.load = {"students": len(students), "bursts": bursts}
        _burst_checks(report, bursts)
    return report


async def run_stress_direct(args) -> Report:
    report = Report("stress-direct")
    users = _users(args)
    ids = [u["user_id"] for u in users][: args.students]
    token = os.environ["AI_WEBHOOK_SHARED_SECRET"]
    png = _homework_png()
    async with httpx.AsyncClient(timeout=TURN_TIMEOUT,
                                 limits=httpx.Limits(max_connections=300)) as client:
        ai = AI(client, args.ai, token)
        health_samples: list[float] = []
        stop = asyncio.Event()

        async def probe():
            while not stop.is_set():
                _, seconds, _ = await _timed(client.get(ai.base + "/health", timeout=30))
                health_samples.append(seconds)
                await asyncio.sleep(1)

        prober = asyncio.create_task(probe())
        bursts = [
            await _burst("grade: every student submits at once", [
                (lambda u=u: ai.post("/ai/grade", json={
                    "user_id": u, "billed_user_id": u, "kind": "quiz",
                    "subject_id": SCIENCE_EN["subject_id"], "submissions": _submissions(SCIENCE_EN)}))
                for u in ids], lambda s, b: s == 200 and b.get("mastery_recorded")),
            await _burst("learner summary: every homepage at once", [
                (lambda u=u: ai.get(f"/ai/learners/{u}/summary")) for u in ids], _ok),
            await _burst("chat stream: every student at once", [
                (lambda u=u: _stream(ai, {"user_id": u, "user_type": "student",
                                          "subject_id": SCIENCE_EN["subject_id"],
                                          "message": "Explain isotopes briefly."}))
                for u in ids], lambda s, b: (b.get("final") or {}).get("event") == "result"),
            await _burst("progress report: every student at once", [
                (lambda u=u: ai.get(f"/ai/reports/{u}", params={"billed_user_id": u}))
                for u in ids], _report_written),
            await _burst("memory read: every student at once", [
                (lambda u=u: ai.get(f"/ai/memory/{u}")) for u in ids], _ok),
            await _burst("topic suggest: every teacher at once", [
                (lambda: ai.post("/ai/topics/suggest", json={
                    "subject_id": SCIENCE_EN["subject_id"],
                    "questions": [{"text": "What is the charge of an electron?"}]}))
                for _ in ids], _ok),
            await _burst("homework capture: every student at once", [
                (lambda u=u: ai.post("/ai/homework/capture", params={"billed_user_id": u},
                                     files={"image": ("hw.png", png, "image/png")}))
                for u in ids], _ok),
            await _burst("mixed: chat + grade + summary + report together", [
                *[(lambda u=u: ai.post("/ai/chat", json={
                    "user_id": u, "user_type": "student", "subject_id": SCIENCE_EN["subject_id"],
                    "message": "What is a mixture?"})) for u in ids[:10]],
                *[(lambda u=u: ai.post("/ai/grade", json={
                    "user_id": u, "billed_user_id": u, "submissions": _submissions(SCIENCE_EN)[:1]}))
                  for u in ids[10:20]],
                *[(lambda u=u: ai.get(f"/ai/learners/{u}/summary")) for u in ids[20:25]],
                *[(lambda u=u: ai.get(f"/ai/reports/{u}", params={"billed_user_id": u}))
                  for u in ids[25:30]]], _ok),
            await _burst("class analytics: whole roster", [
                (lambda: ai.post("/ai/classes/analytics", json={"user_ids": ids}))], _ok),
        ]
        stop.set()
        await prober
        _burst_checks(report, bursts)
        report.load = {"students": len(ids), "bursts": bursts,
                       "health_during_load": _latency(health_samples)}
        print(json.dumps({"health_during_load": report.load["health_during_load"]}), flush=True)
    return report


async def cleanup(args) -> Report:
    report = Report("cleanup")
    users = _users(args)
    async with httpx.AsyncClient(timeout=60) as client:
        ai = AI(client, args.ai, os.environ["AI_WEBHOOK_SHARED_SECRET"])
        for user in users:
            await check(report, "Forget test student", "DELETE /ai/memory/{id}",
                        ai.delete(f"/ai/memory/{user['user_id']}"), ok(lambda b: True,
                                                                     lambda b: _clip(b, 120)))
    return report


MODES = {"backend": run_backend, "direct": run_direct, "stress-backend": run_stress_backend,
         "stress-direct": run_stress_direct, "cleanup": cleanup}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=sorted(MODES))
    parser.add_argument("--api", default=os.environ.get("E2E_API"))
    parser.add_argument("--ai", default=os.environ.get("E2E_AI"))
    parser.add_argument("--students", type=int, default=int(os.environ.get("E2E_STUDENTS", "30")))
    parser.add_argument("--tag", default=os.environ.get("E2E_TAG", "run1"))
    parser.add_argument("--users", default=os.environ.get("E2E_USERS"))
    parser.add_argument("--users-file")
    parser.add_argument("--students-out", default="e2e_students.json")
    parser.add_argument("--out")
    args = parser.parse_args()
    report = asyncio.run(MODES[args.mode](args))
    failed = [c for c in report.checks if not c.passed]
    print(f"\n{args.mode}: {len(report.checks) - len(failed)}/{len(report.checks)} checks passed",
          flush=True)
    payload = {"mode": report.mode, "checks": [asdict(c) for c in report.checks], "load": report.load}
    if args.out:
        _write_json(args.out, payload)
    else:
        print("E2E_RESULT " + json.dumps(payload, ensure_ascii=False, default=str), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
