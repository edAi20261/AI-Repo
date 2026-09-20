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

from shared.config import get_config
from system.errors import ConflictError, NotFoundError, UpstreamTimeoutError
from system.main import create_app
from system.settings import Settings

TOKEN = "test-service-token"


class StubRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.in_flight = 0

    async def chat(self, payload):
        self.calls.append(("chat", payload))
        return {
            "response": "Combustion needs fuel, oxygen and heat [1].",
            "language": "en",
            "pathway": "student",
            "tasks": [{"type": "explain"}],
            "visuals": [{"kind": "chart", "s3_uri": "s3://bucket/chart.png"}],
            "prereq_topics": [{"topic_id": "t1", "label": "Oxygen", "strength": "required", "depth": 1}],
        }

    async def grade(self, payload):
        self.calls.append(("grade", payload))
        return {
            "graded": [{"correct": True, "partial_score": 1.0}],
            "summary": {"score": 1.0},
            "mastery_events": 1,
            "topics_updated": 1,
            "source": "practice",
            "degraded": False,
        }

    async def report(self, payload):
        self.calls.append(("report", payload))
        return {
            "scope": "student_progress",
            "available": True,
            "narrative": "Steady progress.",
            "weak_topics": [],
            "root_cause": None,
            "trend": {"direction": "up", "count": 6},
            "degraded": False,
            "visual_spec": {"kind": "chart", "params": {}},
        }

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
    response = client.post("/ai/grade", json={"user_id": "u-1", "submissions": []}, headers=auth())
    assert response.status_code == 422


def test_grade_requires_kind_with_an_assessment_id(client):
    body = {**GRADE_BODY, "assessment_id": "a-1"}
    response = client.post("/ai/grade", json=body, headers=auth())
    assert response.status_code == 422


def test_grade_drops_unset_optional_fields(client):
    client.post("/ai/grade", json=GRADE_BODY, headers=auth())
    _, payload = client.runtime.calls[0]
    assert "correct_answer" not in payload["submissions"][0]


# --- reports ----------------------------------------------------------------

def test_report_renders_its_trend_chart(client):
    response = client.get("/ai/reports/u-1", headers=auth())
    assert response.status_code == 200

    body = response.json()
    assert body["narrative"] == "Steady progress."
    assert body["trend"]["direction"] == "up"
    assert body["visual"]["s3_uri"] == "s3://bucket/trend.png"


def test_report_forwards_the_subject_scope(client):
    client.get("/ai/reports/u-1", params={"subject_id": "sub-9"}, headers=auth())
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
        get_config(reload=True), _SlowGraph(), None, max_concurrency=2, timeout=0.2
    )

    async def run():
        with pytest.raises(UpstreamTimeoutError):
            await runtime.chat({"message": "hi"})

    anyio.run(run)


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
    from ingestion.utils.source import SourceError
    from shared.errors import ServiceError

    assert issubclass(SourceError, ServiceError)
    assert SourceError("source file not found").status_code == 422


# --- homework capture -------------------------------------------------------

def _photo(client, content=b"\x89PNG\r\n", content_type="image/png"):
    return client.post(
        "/ai/homework/capture",
        files={"image": ("page.png", content, content_type)},
        headers=auth(),
    )


def test_a_homework_photo_comes_back_as_questions(client, monkeypatch):
    from agent.homework import HomeworkCapture

    import system.routers.homework as router
    monkeypatch.setattr(router, "capture", lambda *a, **k: HomeworkCapture.model_validate(
        {"questions": [{"number": "1", "question": "ما هو السجع؟",
                        "student_answer": "تشابه"}]}))

    body = _photo(client).json()
    assert body["questions"][0]["question"] == "ما هو السجع؟"
    assert body["questions"][0]["student_answer"] == "تشابه"
    assert body["unreadable"] is False
    assert body["request_id"]


def test_an_unreadable_photo_is_reported_not_treated_as_empty(client, monkeypatch):
    from agent.homework import HomeworkCapture

    import system.routers.homework as router
    monkeypatch.setattr(router, "capture", lambda *a, **k: HomeworkCapture(unreadable=True))

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
