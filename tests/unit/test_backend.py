"""Backend read-client unit tests. No network, no platform: the fixture client
reads a temp manifest and the HTTP client's httpx is monkeypatched."""

from __future__ import annotations

import httpx
import pytest

from shared.backend import (
    FixtureBackendClient,
    HttpBackendClient,
    get_backend_client,
)
from shared.config import Config

_MANIFEST = """
students:
  student-1:
    subjects:
      - { id: subjA, name: Science, grade_level: g8, language: en }
      - { id: subjB, name: Math, grade_level: g8, language: en }
subjects:
  subjA: { id: subjA, curriculum_id: cur1, grade_level: g8, name: Science, code: SCI8, language: en }
question_bank:
  subjA:
    - { id: q1, question_text: Q1, question_type: mcq, topic_ids: [t1], lecture_id: lecA, correct_answer_text: A, points: 1 }
    - { id: q2, question_text: Q2, question_type: short_answer, topic_ids: [t2], lecture_id: lecA, correct_answer_text: B, points: 2 }
homework:
  hw1:
    questions:
      - { id: h1, question_text: HQ1, correct_answer_text: "yes", points: 3, rubric: [{criterion: c, points: 3}] }
quizzes:
  qz1:
    questions:
      - { id: z1, question_text: ZQ1, correct_answer_text: oxygen, points: 1 }
"""


@pytest.fixture()
def manifest(tmp_path):
    p = tmp_path / "agent_fixtures.yaml"
    p.write_text(_MANIFEST, encoding="utf-8")
    return p


# --- fixture client ---------------------------------------------------------

def test_fixture_reads_student_subjects(manifest):
    c = FixtureBackendClient(manifest)
    subs = c.get_student_subjects("student-1")
    assert [s["id"] for s in subs] == ["subjA", "subjB"]


def test_fixture_subject_and_missing(manifest):
    c = FixtureBackendClient(manifest)
    assert c.get_subject("subjA")["code"] == "SCI8"
    with pytest.raises(KeyError):
        c.get_subject("nope")


def test_fixture_question_bank_filters_and_limits(manifest):
    c = FixtureBackendClient(manifest)
    assert len(c.get_question_bank("subjA")) == 2
    assert [q["id"] for q in c.get_question_bank("subjA", topic_ids=["t2"])] == ["q2"]
    assert len(c.get_question_bank("subjA", count=1)) == 1


def test_fixture_questions_answer_key(manifest):
    c = FixtureBackendClient(manifest)
    assert c.get_questions("homework", "hw1")[0]["correct_answer_text"] == "yes"
    assert c.get_questions("quiz", "qz1")[0]["correct_answer_text"] == "oxygen"
    with pytest.raises(ValueError):
        c.get_questions("exam", "x1")  # unknown kind


# --- factory ----------------------------------------------------------------

def test_factory_prefers_http_when_base_url_set(manifest, monkeypatch):
    monkeypatch.setenv("BACKEND_API_BASE_URL", "https://api.example/api/ai-integration")
    monkeypatch.setenv("AI_WEBHOOK_SHARED_SECRET", "tok")
    cfg = Config({"agent": {"backend": {"fixtures_path": str(manifest)}}}, "test")
    assert isinstance(get_backend_client(cfg), HttpBackendClient)


def test_factory_falls_back_to_fixture(manifest, monkeypatch):
    monkeypatch.delenv("BACKEND_API_BASE_URL", raising=False)
    cfg = Config({"agent": {"backend": {"fixtures_path": str(manifest)}}}, "test")
    assert isinstance(get_backend_client(cfg), FixtureBackendClient)


def test_factory_errors_without_source(monkeypatch):
    monkeypatch.delenv("BACKEND_API_BASE_URL", raising=False)
    cfg = Config({"agent": {"backend": {"fixtures_path": None}}}, "test")
    with pytest.raises(ValueError):
        get_backend_client(cfg)


# --- HTTP client ------------------------------------------------------------

class _Resp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else []

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "err", request=httpx.Request("GET", "http://x"),
                response=httpx.Response(self.status_code))

    def json(self):
        return self._payload


def test_http_sends_token_and_drops_none_params(monkeypatch):
    calls = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        calls["url"] = url
        calls["params"] = params
        calls["headers"] = headers
        return _Resp(200, [{"id": "subjA"}])

    monkeypatch.setattr(httpx, "get", fake_get)
    client = HttpBackendClient("https://api/api/ai-integration", "secret-tok")
    out = client.get_question_bank("subjA", topic_ids=None, count=5)

    assert out == [{"id": "subjA"}]
    assert calls["url"].endswith("/question-bank")
    assert calls["headers"]["x-service-token"] == "secret-tok"
    assert "topic_ids" not in calls["params"]  # None dropped
    assert calls["params"]["count"] == 5


def test_http_retries_transient_then_succeeds(monkeypatch):
    monkeypatch.setattr("shared.backend.time.sleep", lambda *_: None)
    attempts = {"n": 0}

    def flaky_get(url, params=None, headers=None, timeout=None):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.TransportError("boom")
        return _Resp(200, {"id": "subjA"})

    monkeypatch.setattr(httpx, "get", flaky_get)
    client = HttpBackendClient("https://api/api/ai-integration", "tok", max_retries=3)
    assert client.get_subject("subjA") == {"id": "subjA"}
    assert attempts["n"] == 3


def test_http_does_not_retry_4xx(monkeypatch):
    monkeypatch.setattr("shared.backend.time.sleep", lambda *_: None)
    attempts = {"n": 0}

    def not_found(url, params=None, headers=None, timeout=None):
        attempts["n"] += 1
        return _Resp(404)

    monkeypatch.setattr(httpx, "get", not_found)
    client = HttpBackendClient("https://api/api/ai-integration", "tok", max_retries=3)
    with pytest.raises(httpx.HTTPStatusError):
        client.get_subject("missing")
    assert attempts["n"] == 1  # 4xx raised immediately, no retry
