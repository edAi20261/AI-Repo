"""HTTP layer tests.

The runtime is stubbed, so these cover the transport contract only — auth,
validation, response shape, error mapping — with no keys and no services.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

import anyio
import pytest
from fastapi.testclient import TestClient

from agent.homework import HomeworkCapture
from shared.config import get_config
from shared.metering import Meter, record_usage
from system.errors import ConflictError, NotFoundError, UpstreamTimeoutError
from system.main import create_app
from system.billing import Billing
from system.runtime import Metered
from system.settings import Settings

TOKEN = "test-service-token"


class StubRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.billed: list[str] = []
        self.in_flight = 0
        self.config = get_config()
        self.homework = HomeworkCapture()

    async def chat(self, payload):
        self.calls.append(("chat", payload))
        meter = Meter()
        meter.record(0.003, 2000, 300)
        meter.record(0.0012, 1100, 150)
        return Metered({
            "response": "Combustion needs fuel, oxygen and heat [1].",
            "language": "en",
            "pathway": "student",
            "tasks": [{"type": "explain"}],
            "visuals": [{"kind": "chart", "s3_uri": "s3://bucket/chart.png"}],
            "prereq_topics": [{"topic_id": "t1", "label": "Oxygen", "strength": "required", "depth": 1}],
        }, meter)

    async def grade(self, payload, *, billed_user_id):
        self.calls.append(("grade", payload))
        self.billed.append(billed_user_id)
        return Metered({
            "graded": [{"correct": True, "partial_score": 1.0}],
            "summary": {"score": 1.0},
            "mastery_events": 1,
            "topics_updated": 1,
            "source": "practice",
            "degraded": False,
        }, Meter())

    async def report(self, payload, *, billed_user_id):
        self.calls.append(("report", payload))
        self.billed.append(billed_user_id)
        return Metered({
            "scope": "student_progress",
            "available": True,
            "narrative": "Steady progress.",
            "weak_topics": [],
            "root_cause": None,
            "trend": {"direction": "up", "count": 6},
            "degraded": False,
            "visual_spec": {"kind": "chart", "params": {}},
        }, Meter())

    async def capture_homework(self, content, *, language, billed_user_id):
        self.calls.append(("capture_homework", {"bytes": len(content), "language": language}))
        self.billed.append(billed_user_id)
        return Metered(self.homework, Meter())

    async def class_analytics(self, payload):
        self.calls.append(("class_analytics", payload))
        return {"available": True, "degraded": False,
                "students": {"total": 2, "with_evidence": 1},
                "at_risk": [{"user_id": "s1", "reasons": [{"code": "declining"}]}],
                "not_started": ["s2"]}

    async def class_report(self, payload, *, billed_user_id):
        self.calls.append(("class_report", payload))
        self.billed.append(billed_user_id)
        return Metered({"scope": "class_analytics", "available": True,
                        "narrative": "Most of the class is weak on Fractions.",
                        "claims": [], "analysis": {"students": {"total": 2}},
                        "degraded": False, "report_id": "r5", "persisted": True}, Meter())

    async def latest_report(self, payload):
        self.calls.append(("latest_report", payload))
        if payload.get("user_id") == "nobody" or payload.get("class_id") == "empty":
            return None
        return {"report_id": "r1", "generated_at": "2026-09-27T08:00:00+00:00",
                "payload": {"narrative": "Steady progress."}, "newer_evidence": 2,
                "stale": True}

    async def learner_summary(self, payload):
        self.calls.append(("learner_summary", payload))
        return {"available": True, "degraded": False,
                "subjects": [{"subject_id": "math", "topics": 4, "strong": 1,
                              "developing": 0, "weak": 2, "unknown": 0,
                              "not_practised": 1}],
                "recommendations": [{"topic_id": "t1", "action": "practise",
                                     "reasons": [{"code": "low_mastery", "score": 0.3}]}],
                "trend": {"direction": "flat", "count": 4}}

    async def suggest_topics(self, payload):
        self.calls.append(("suggest_topics", payload))
        return {"suggestions": [{"index": 0, "question_id": "q1", "resolved": True,
                                 "topics": [{"topic_id": "t1", "label": "Fractions",
                                             "score": 0.8, "lecture_id": "l1"}]}],
                "degraded": False}

    async def render(self, specs, **_):
        self.calls.append(("render", {"specs": specs}))
        return [{"kind": "chart", "s3_uri": "s3://bucket/trend.png"}]


class StubJobs:
    def __init__(self) -> None:
        self.submitted: list[dict] = []
        self.conflict = False

    async def submit(self, **kwargs):
        if self.conflict:
            raise ConflictError("This lecture is already being ingested.")
        self.submitted.append(kwargs)
        return _job(kwargs["lecture_id"], kwargs["subject_id"])

    async def submit_lesson(self, lesson_id):
        if self.conflict:
            raise ConflictError("This lecture is already being ingested.")
        self.submitted.append({"lesson_id": lesson_id})
        return _job(lesson_id, None)

    def get(self, job_id):
        if job_id == "missing":
            raise NotFoundError("No such ingestion job.")
        return _job("lec-1", "sub-1", job_id=job_id)

    def latest_for_lecture(self, lecture_id):
        return _job(lecture_id, "sub-1")

    def list(self, *, limit=50):
        return [_job("lec-1", "sub-1")]

    async def drain(self, timeout=30.0):
        return None


def _job(lecture_id, subject_id, job_id="job-1"):
    from system.ingestion_jobs import IngestionJob, JobStatus

    job = IngestionJob(job_id=job_id, lecture_id=lecture_id, subject_id=subject_id)
    job.status = JobStatus.QUEUED
    return job


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("AI_WEBHOOK_SHARED_SECRET", TOKEN)
    config = get_config(reload=True)

    app = create_app(config)
    app.state.settings = Settings.from_config(config)

    runtime, jobs = StubRuntime(), StubJobs()

    @asynccontextmanager
    async def _stub_lifespan(scoped_app):
        scoped_app.state.runtime = runtime
        scoped_app.state.jobs = jobs
        scoped_app.state.billing = Billing(None)
        yield

    app.router.lifespan_context = _stub_lifespan
    with TestClient(app) as test_client:
        test_client.runtime = runtime
        test_client.jobs = jobs
        yield test_client


def auth(extra=None):
    headers = {"x-service-token": TOKEN}
    headers.update(extra or {})
    return headers


CHAT_BODY = {"user_id": "u-1", "user_type": "student", "message": "Explain the fire triangle."}


# --- auth -------------------------------------------------------------------

def test_chat_without_token_is_rejected(client):
    response = client.post("/ai/chat", json=CHAT_BODY)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_chat_with_wrong_token_is_rejected(client):
    response = client.post("/ai/chat", json=CHAT_BODY, headers={"x-service-token": "nope"})
    assert response.status_code == 401


def test_health_needs_no_token(client):
    assert client.get("/health").status_code == 200


# --- chat -------------------------------------------------------------------

def test_chat_returns_composed_answer(client):
    response = client.post("/ai/chat", json=CHAT_BODY, headers=auth())
    assert response.status_code == 200

    body = response.json()
    assert body["response"].startswith("Combustion")
    assert body["language"] == "en"
    assert body["tasks"] == ["explain"]
    assert body["visuals"][0]["s3_uri"] == "s3://bucket/chart.png"
    assert body["request_id"]


def test_chat_reports_what_the_turn_cost(client):
    """The backend charges credits per message and records beside each charge what
    the message really cost. Without the cost in the response it can only guess,
    and a price table that has drifted from the bill is invisible until the invoice."""
    body = client.post("/ai/chat", json=CHAT_BODY, headers=auth()).json()
    assert body["usage"] == {"cost_usd": 0.0042, "prompt_tokens": 3100,
                             "completion_tokens": 450, "calls": 2}


def test_chat_passes_only_supplied_fields_to_the_graph(client):
    client.post("/ai/chat", json=CHAT_BODY, headers=auth())
    _, payload = client.runtime.calls[0]
    assert payload == {"user_id": "u-1", "user_type": "student",
                       "message": "Explain the fire triangle."}


def test_chat_forwards_grade_reference(client):
    body = {**CHAT_BODY, "grade_ref": {"assessment_id": "a-1", "kind": "quiz"}}
    client.post("/ai/chat", json=body, headers=auth())
    _, payload = client.runtime.calls[0]
    assert payload["grade_ref"] == {"assessment_id": "a-1", "kind": "quiz"}


def test_chat_rejects_an_unknown_role(client):
    body = {**CHAT_BODY, "user_type": "principal"}
    response = client.post("/ai/chat", json=body, headers=auth())
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_chat_rejects_an_empty_message(client):
    response = client.post("/ai/chat", json={**CHAT_BODY, "message": ""}, headers=auth())
    assert response.status_code == 422


def test_chat_rejects_unknown_fields(client):
    response = client.post("/ai/chat", json={**CHAT_BODY, "temperature": 0.9}, headers=auth())
    assert response.status_code == 422
    fields = response.json()["error"]["details"]["fields"]
    assert any("temperature" in field["field"] for field in fields)


def test_request_id_is_echoed_when_supplied(client):
    response = client.post(
        "/ai/chat", json=CHAT_BODY, headers=auth({"x-request-id": "trace-123"})
    )
    assert response.headers["x-request-id"] == "trace-123"
    assert response.json()["request_id"] == "trace-123"


# --- grading ----------------------------------------------------------------

GRADE_BODY = {
    "user_id": "u-1",
    "billed_user_id": "t-1",
    "submissions": [{"question": "What is 2+2?", "student_answer": "4"}],
}


def test_grade_returns_the_result(client):
    response = client.post("/ai/grade", json=GRADE_BODY, headers=auth())
    assert response.status_code == 200

    body = response.json()
    assert body["mastery_events"] == 1
    assert body["topics_updated"] == 1
    assert body["source"] == "practice"


def test_grade_requires_at_least_one_submission(client):
    response = client.post("/ai/grade", json={"user_id": "u-1", "billed_user_id": "t-1", "submissions": []}, headers=auth())
    assert response.status_code == 422


def test_grade_requires_kind_with_an_assessment_id(client):
    body = {**GRADE_BODY, "assessment_id": "a-1"}
    response = client.post("/ai/grade", json=body, headers=auth())
    assert response.status_code == 422


def test_grade_drops_unset_optional_fields(client):
    client.post("/ai/grade", json=GRADE_BODY, headers=auth())
    _, payload = client.runtime.calls[0]
    assert "correct_answer" not in payload["submissions"][0]


def test_the_latest_report_is_served_with_its_freshness(client):
    response = client.get("/ai/reports/u-1/latest", headers=auth())
    assert response.status_code == 200
    body = response.json()
    assert body["report_id"] == "r1" and body["stale"] is True
    assert body["report"]["narrative"] == "Steady progress."


def test_no_stored_report_is_a_404_not_an_empty_report(client):
    response = client.get("/ai/reports/nobody/latest", headers=auth())
    assert response.status_code == 404


# --- learners ---------------------------------------------------------------

def test_learner_summary_returns_the_analysis(client):
    response = client.get("/ai/learners/u-1/summary", headers=auth())
    assert response.status_code == 200
    body = response.json()
    assert body["subjects"][0]["weak"] == 2
    assert body["recommendations"][0]["reasons"][0]["code"] == "low_mastery"


def test_learner_summary_forwards_the_enrolled_subjects_and_current_lessons(client):
    client.get("/ai/learners/u-1/summary",
               params={"subject_ids": ["math", "bio"], "lecture_ids": ["l1"]}, headers=auth())
    assert client.runtime.calls[0] == ("learner_summary", {
        "user_id": "u-1", "subject_ids": ["math", "bio"], "lecture_ids": ["l1"]})


def test_learner_summary_needs_the_service_token(client):
    assert client.get("/ai/learners/u-1/summary").status_code == 401


# --- classes ----------------------------------------------------------------

def test_class_analytics_aggregates_the_roster_it_is_given(client):
    body = {"user_ids": ["s1", "s2"], "subject_ids": ["math"], "class_id": "c1"}
    response = client.post("/ai/classes/analytics", json=body, headers=auth())
    assert response.status_code == 200
    assert response.json()["at_risk"][0]["reasons"] == [{"code": "declining"}]
    assert client.runtime.calls[0] == ("class_analytics", body)


def test_class_analytics_refuses_an_empty_or_oversized_roster(client):
    assert client.post("/ai/classes/analytics", json={"user_ids": []},
                       headers=auth()).status_code == 422
    assert client.post("/ai/classes/analytics",
                       json={"user_ids": [f"s{i}" for i in range(501)]},
                       headers=auth()).status_code == 422


def test_a_class_report_is_written_for_the_roster_given(client):
    body = {"class_id": "c1", "user_ids": ["s1", "s2"], "language": "ar"}
    response = client.post("/ai/classes/report", json={**body, "billed_user_id": "t-1"},
                           headers=auth())
    assert response.status_code == 200
    assert response.json()["report_id"] == "r5"
    assert client.runtime.calls[0] == ("class_report", body)
    assert client.runtime.billed == ["t-1"]


def test_a_class_report_needs_its_class(client):
    response = client.post("/ai/classes/report", json={"user_ids": ["s1"], "billed_user_id": "t-1"},
                           headers=auth())
    assert response.status_code == 422


def test_the_latest_class_report_is_served_or_404(client):
    assert client.get("/ai/classes/c1/report/latest", headers=auth()).status_code == 200
    assert client.runtime.calls[-1] == ("latest_report", {"class_id": "c1",
                                                          "scope": "class_analytics"})
    assert client.get("/ai/classes/empty/report/latest", headers=auth()).status_code == 404


def test_a_parent_report_is_asked_for_by_scope(client):
    client.get("/ai/reports/u-1", params={"scope": "parent", "billed_user_id": "p-1"},
               headers=auth())
    assert client.runtime.calls[0][1]["scope"] == "parent"


# --- topics -----------------------------------------------------------------

def test_topic_suggestions_come_back_per_question(client):
    body = {"subject_id": "math", "questions": [{"text": "3/4 + 1/8", "question_id": "q1"}]}
    response = client.post("/ai/topics/suggest", json=body, headers=auth())
    assert response.status_code == 200
    assert response.json()["suggestions"][0]["topics"][0]["topic_id"] == "t1"
    assert client.runtime.calls[0] == ("suggest_topics", body)


def test_topic_suggestions_need_a_subject_and_a_question(client):
    assert client.post("/ai/topics/suggest", json={"questions": [{"text": "q"}]},
                       headers=auth()).status_code == 422
    assert client.post("/ai/topics/suggest", json={"subject_id": "s", "questions": []},
                       headers=auth()).status_code == 422


# --- reports ----------------------------------------------------------------

def test_report_renders_its_trend_chart(client):
    response = client.get("/ai/reports/u-1", params={"billed_user_id": "u-1"}, headers=auth())
    assert response.status_code == 200

    body = response.json()
    assert body["narrative"] == "Steady progress."
    assert body["trend"]["direction"] == "up"
    assert body["visual"]["s3_uri"] == "s3://bucket/trend.png"


def test_report_forwards_the_subject_scope(client):
    client.get("/ai/reports/u-1", params={"subject_id": "sub-9", "billed_user_id": "u-1"},
               headers=auth())
    _, payload = client.runtime.calls[0]
    assert payload["subject_id"] == "sub-9"


# --- ingestion --------------------------------------------------------------

INGEST_BODY = {
    "lecture_id": "lec-1",
    "subject_id": "sub-1",
    "source_uri": "file:///lessons/lec-1.pdf",
}


def test_ingest_is_accepted_not_executed(client):
    response = client.post("/ai/ingest", json=INGEST_BODY, headers=auth())
    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert client.jobs.submitted[0]["lecture_id"] == "lec-1"


def test_ingest_rejects_a_duplicate_run(client):
    client.jobs.conflict = True
    response = client.post("/ai/ingest", json=INGEST_BODY, headers=auth())
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "conflict"


def test_unknown_job_is_a_404(client):
    response = client.get("/ai/ingest/missing", headers=auth())
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_job_lookup_by_lecture(client):
    response = client.get("/ai/ingest/lectures/lec-7", headers=auth())
    assert response.status_code == 200
    assert response.json()["lecture_id"] == "lec-7"


# --- startup preflight ------------------------------------------------------

def test_production_refuses_to_start_without_a_service_token(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AI_WEBHOOK_SHARED_SECRET", "")
    config = get_config(reload=True)

    with pytest.raises(RuntimeError, match="AI_WEBHOOK_SHARED_SECRET"):
        create_app(config)


def test_production_refuses_to_start_without_stores(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AI_WEBHOOK_SHARED_SECRET", TOKEN)
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("QDRANT_URL", "")
    config = get_config(reload=True)

    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        create_app(config)


def test_development_starts_without_them(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("AI_WEBHOOK_SHARED_SECRET", "")
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("QDRANT_URL", "")

    assert create_app(get_config(reload=True)) is not None


# --- timeout ----------------------------------------------------------------

class _SlowGraph:
    def invoke(self, _payload):
        time.sleep(2)
        return {"response": "too late"}


def test_a_turn_past_the_timeout_is_abandoned(monkeypatch):
    """A hung model call must release the caller rather than hold the request open."""
    monkeypatch.setenv("APP_ENV", "test")
    from system.runtime import AgentRuntime

    runtime = AgentRuntime(
        get_config(reload=True), _SlowGraph(), None, billing=Billing(None), max_concurrency=2, timeout=0.2
    )

    async def run():
        with pytest.raises(UpstreamTimeoutError):
            await runtime.chat({"user_id": "u-1", "message": "hi"})

    anyio.run(run)


class _StreamingGraph:
    """A graph that yields node updates, the way LangGraph's astream does."""

    async def astream(self, _payload, stream_mode=None):
        yield {"plan": {"tasks": ["explain"]}}
        yield {"compose": {"response": "Jabir ibn Hayyan was a chemist."}}


def test_a_streamed_turn_reaches_the_student(monkeypatch):
    """POST /ai/chat/stream failed in milliseconds with internal_error, every time,
    in every environment: the turn was wrapped in `async with ... anyio.fail_after(...)`,
    and fail_after is a synchronous context manager. The endpoint had never worked,
    while /ai/chat beside it did — so nothing that only exercised the plain endpoint
    ever saw it."""
    monkeypatch.setenv("APP_ENV", "test")
    from system.runtime import AgentRuntime

    runtime = AgentRuntime(
        get_config(reload=True), _StreamingGraph(), None, billing=Billing(None), max_concurrency=2, timeout=5
    )

    async def run():
        updates = await runtime.stream_chat({"user_id": "u-1", "message": "hi"})
        return [pair async for pair in updates]

    updates = anyio.run(run)
    assert [stage for stage, _ in updates] == ["plan", "compose", "__final__"]
    # The final payload is the accumulation of every node's update, not the last one.
    assert updates[-1][1].result == {
        "tasks": ["explain"], "response": "Jabir ibn Hayyan was a chemist."}


class _BilledGraph:
    """Records usage from inside the graph, where the model callbacks run."""

    def invoke(self, _payload):
        record_usage({"cost": 0.002, "prompt_tokens": 900, "completion_tokens": 120})
        record_usage({"cost": 0.001, "prompt_tokens": 400, "completion_tokens": 60})
        return {"response": "Evaporation is ..."}

    async def astream(self, _payload, stream_mode=None):
        record_usage({"cost": 0.003, "prompt_tokens": 1000, "completion_tokens": 200})
        yield {"compose": {"response": "Evaporation is ..."}}


def test_a_turn_carries_the_cost_its_model_calls_reported(monkeypatch):
    """The graph runs in a worker thread, so a meter started anywhere but around
    the turn would see nothing and every message would be billed at zero."""
    monkeypatch.setenv("APP_ENV", "test")
    from system.runtime import AgentRuntime

    runtime = AgentRuntime(
        get_config(reload=True), _BilledGraph(), None, billing=Billing(None), max_concurrency=2, timeout=5
    )

    turn = anyio.run(runtime.chat, {"user_id": "u-1", "message": "hi"})
    assert turn.result == {"response": "Evaporation is ..."}
    assert turn.meter.as_dict() == {"cost_usd": 0.003, "prompt_tokens": 1300,
                                    "completion_tokens": 180, "calls": 2}


def test_a_streamed_turn_carries_the_cost_its_model_calls_reported(monkeypatch):
    """The streamed endpoint runs the graph on a different path from /ai/chat; a
    meter started for one and not the other bills every streamed message at zero."""
    monkeypatch.setenv("APP_ENV", "test")
    from system.runtime import AgentRuntime

    runtime = AgentRuntime(
        get_config(reload=True), _BilledGraph(), None, billing=Billing(None), max_concurrency=2, timeout=5
    )

    async def run():
        updates = await runtime.stream_chat({"user_id": "u-1", "message": "hi"})
        return [pair async for pair in updates]

    stage, turn = anyio.run(run)[-1]
    assert stage == "__final__"
    assert turn.meter.as_dict()["cost_usd"] == 0.003


def test_a_streamed_turn_past_the_timeout_is_abandoned(monkeypatch):
    """The timeout has to survive being made to work."""
    monkeypatch.setenv("APP_ENV", "test")
    from system.runtime import AgentRuntime

    class _SlowStream:
        async def astream(self, _payload, stream_mode=None):
            await anyio.sleep(2)
            yield {"plan": {}}

    runtime = AgentRuntime(
        get_config(reload=True), _SlowStream(), None, billing=Billing(None), max_concurrency=2, timeout=0.2
    )

    async def run():
        with pytest.raises(UpstreamTimeoutError):
            async for _ in await runtime.stream_chat({"user_id": "u-1", "message": "hi"}):
                pass

    anyio.run(run)


def test_a_service_pointed_at_an_empty_index_says_so(monkeypatch, caplog):
    """A whole afternoon of testing was spent against an empty index.

    APP_ENV=local points the vector store at eduai_chunks_dev, which held no
    points. The service started, answered every request, and declined or
    degraded each one — indistinguishable from the tutor being broken. Nothing
    at startup said the collection it was about to search was empty.
    """
    import logging

    from system import security

    monkeypatch.setattr(security, "collection_size", lambda settings: 0)
    with caplog.at_level(logging.WARNING):
        security.warn_if_index_empty(get_config(reload=True))

    assert any("empty" in r.message.lower() for r in caplog.records)


def test_a_service_pointed_at_a_populated_index_is_quiet(monkeypatch, caplog):
    import logging

    from system import security

    monkeypatch.setattr(security, "collection_size", lambda settings: 154)
    with caplog.at_level(logging.WARNING):
        security.warn_if_index_empty(get_config(reload=True))

    assert not caplog.records


# --- the platform's ingestion contract ---------------------------------------

def test_platform_trigger_queues_by_lesson_id(client):
    """The platform fires POST /ai/ingest/lesson with just {lesson_id}."""
    response = client.post(
        "/ai/ingest/lesson", json={"lecture_id": "lec-1"}, headers=auth()
    )
    assert response.status_code == 422  # it sends lesson_id, not lecture_id

    response = client.post("/ai/ingest/lesson", json={"lesson_id": "lec-1"}, headers=auth())
    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert client.jobs.submitted[0] == {"lesson_id": "lec-1"}


def test_platform_trigger_returns_before_resolving(client):
    """Resolution costs three calls back to the platform; the trigger is
    fire-and-forget on a short timeout, so it must not wait."""
    response = client.post("/ai/ingest/lesson", json={"lesson_id": "lec-2"}, headers=auth())
    assert response.status_code == 202
    assert response.json()["subject_id"] is None


def test_status_maps_to_the_platform_enum():
    from system.ingestion_jobs import IngestionJob, JobStatus
    from system.notifier import platform_status

    job = IngestionJob(job_id="j", lecture_id="l", subject_id="s")
    assert platform_status(job) == "pending"

    job.status = JobStatus.RUNNING
    assert platform_status(job) == "processing"

    job.status = JobStatus.SUCCEEDED
    assert platform_status(job) == "ready"

    job.ingestion_status = "needs_review"
    assert platform_status(job) == "ready", "a partial ingest is still usable"

    job.status = JobStatus.FAILED
    assert platform_status(job) == "failed"


# --- the manager talks to the notifier ---------------------------------------

class RecordingNotifier:
    """A real notifier stands in for the platform: if the manager calls anything
    other than `report`, these tests fail rather than a lesson silently sticking
    at 'pending' in production."""

    def __init__(self) -> None:
        self.reported: list[str] = []

    async def report(self, job) -> None:
        from system.notifier import platform_status

        self.reported.append(platform_status(job))


def _manager(notifier):
    from system.ingestion_jobs import IngestionJobManager
    from system.settings import IngestionSettings

    return IngestionJobManager(
        get_config(reload=True), IngestionSettings(max_workers=1), notifier=notifier
    )


def test_a_run_reports_processing_then_ready(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    notifier = RecordingNotifier()
    manager = _manager(notifier)
    monkeypatch.setattr(manager, "_ingest", lambda state: {"stats": {}, "ingestion_status": "ready"})

    async def run():
        job = await manager.submit(
            lecture_id="lec-1", subject_id="sub-1", source_uri="file:///x.pdf"
        )
        await manager.drain(timeout=5)
        return job

    job = anyio.run(run)
    assert notifier.reported == ["processing", "ready"]
    assert job.status.value == "succeeded"


def test_a_failed_run_reports_failed_and_releases_the_lecture(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    notifier = RecordingNotifier()
    manager = _manager(notifier)

    def boom(_state):
        raise RuntimeError("pdf unreadable")

    monkeypatch.setattr(manager, "_ingest", boom)

    async def run():
        await manager.submit(lecture_id="lec-2", subject_id="s", source_uri="file:///x.pdf")
        await manager.drain(timeout=5)
        # The lecture must be free again, or it can never be retried.
        await manager.submit(lecture_id="lec-2", subject_id="s", source_uri="file:///x.pdf")
        await manager.drain(timeout=5)

    anyio.run(run)
    assert notifier.reported == ["processing", "failed", "processing", "failed"]


def test_a_failed_resolve_is_reported_not_swallowed(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    notifier = RecordingNotifier()
    manager = _manager(notifier)

    def boom(_lesson_id):
        raise RuntimeError("no such lesson")

    monkeypatch.setattr(manager, "_resolve", boom)

    async def run():
        job = await manager.submit_lesson("lesson-9")
        await manager.drain(timeout=5)
        return job

    job = anyio.run(run)
    assert notifier.reported == ["failed"]
    assert "no such lesson" in job.errors[0]


def test_a_dependency_failing_is_not_reported_as_a_bug_in_this_service():
    """There were three disjoint error hierarchies, and only one of them could
    reach an HTTP status. S3 refusing a signed URL and the agent having a bug
    both came back as 500 "An unexpected error occurred", so a caller could not
    tell which one was worth retrying."""
    from shared.errors import ServiceError
    from shared.storage import StorageError
    from shared.structured import StructuredOutputError

    assert issubclass(StorageError, ServiceError)
    assert issubclass(StructuredOutputError, ServiceError)
    assert StorageError("s3 refused").status_code == 503


def test_unusable_source_material_is_not_a_server_error():
    """A lesson whose PDF is missing is the request's problem, not the
    service's, and a retry will not fix it."""
    from shared.source import SourceError
    from shared.errors import ServiceError

    assert issubclass(SourceError, ServiceError)
    assert SourceError("source file not found").status_code == 422


# --- homework capture -------------------------------------------------------

def _photo(client, content=b"\x89PNG\r\n", content_type="image/png"):
    return client.post(
        "/ai/homework/capture",
        params={"billed_user_id": "u-1"},
        files={"image": ("page.png", content, content_type)},
        headers=auth(),
    )


def test_a_homework_photo_comes_back_as_questions(client):
    client.runtime.homework = HomeworkCapture.model_validate(
        {"questions": [{"number": "1", "question": "ما هو السجع؟",
                        "student_answer": "تشابه"}]})

    body = _photo(client).json()
    assert body["questions"][0]["question"] == "ما هو السجع؟"
    assert body["questions"][0]["student_answer"] == "تشابه"
    assert body["unreadable"] is False
    assert body["request_id"]


def test_an_unreadable_photo_is_reported_not_treated_as_empty(client):
    client.runtime.homework = HomeworkCapture(unreadable=True)

    body = _photo(client).json()
    assert body["unreadable"] is True
    assert body["questions"] == []


def test_a_pdf_is_refused_before_a_vision_call_is_paid_for(client):
    response = _photo(client, content_type="application/pdf")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unprocessable"


def test_a_scanned_booklet_is_refused_by_size(client):
    from shared.config import Config

    limit = Config.load().get("api.homework.max_upload_bytes")
    response = _photo(client, content=b"x" * (limit + 1))
    assert response.status_code == 422
    assert response.json()["error"]["details"]["limit"] == limit


def test_an_empty_upload_is_refused(client):
    assert _photo(client, content=b"").status_code == 422


def test_capture_needs_the_service_token(client):
    response = client.post(
        "/ai/homework/capture",
        files={"image": ("page.png", b"\x89PNG", "image/png")},
        headers={"x-service-token": "nope"},
    )
    assert response.status_code == 401
