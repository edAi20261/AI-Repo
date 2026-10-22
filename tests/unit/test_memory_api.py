"""A student can be shown what the tutor holds about them, and have it destroyed.

Nothing this service remembered could be read back or erased by anybody. The
users are children; "what do you know about my child, and delete it" is a
question that gets asked, and the only answer was a database console.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from agent.config import get_config
from agent.profile import forget, held
from system.billing import Billing
from system.main import create_app
from system.runtime import AgentRuntime

USER = str(uuid.uuid4())
SESSION = str(uuid.uuid4())
WHEN = datetime(2026, 9, 22, 15, 56, tzinfo=timezone.utc)

HELD_ROWS = [
    ("conversation_turn", SESSION, '{"message":"q","reply":"a"}', WHEN, None),
    ("conversation_summary", SESSION, "Goals: understand evaporation", WHEN, None),
    ("conversation_note", SESSION, "We talked about evaporation.", WHEN, None),
    ("conversation_turn", str(uuid.uuid4()), '{"message":"q2","reply":"a2"}', WHEN, None),
]


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("AI_WEBHOOK_SHARED_SECRET", "test-token")
    app = create_app(get_config())
    app.state.runtime = AgentRuntime(get_config(), None, None, billing=Billing(None),
                                     max_concurrency=2, timeout=5)
    return TestClient(app)


def _auth(client):
    return {"x-service-token": "test-token"}


# --- what is held -------------------------------------------------------------

def test_the_summaries_are_shown_and_the_turns_are_not(fake_postgres):
    """The words of a conversation belong to the platform and the student is
    already shown them. What the tutor made of those words is the part nobody
    could see."""
    fake_postgres(fetchall=[HELD_ROWS], fetchone=[("short", True, False, None)])
    record = held(USER, config=get_config())
    assert record["conversations"] == 2
    assert [s["summary"] for s in record["summaries"]] == ["Goals: understand evaporation"]
    assert record["stated_preferences"] == {"length": "short", "examples": True}
    assert "q" not in str(record["summaries"])


def test_each_conversation_carries_the_plain_note_the_student_reads(fake_postgres):
    """The memory page showed the tutor's internal summary, headings and all."""
    older = str(uuid.uuid4())
    rows = [*HELD_ROWS, ("conversation_summary", older, "Covered: fractions", WHEN, None)]
    fake_postgres(fetchall=[rows], fetchone=[None])
    record = held(USER, config=get_config())
    by_conversation = {s["conversation_id"]: s for s in record["summaries"]}
    assert by_conversation[SESSION]["note"] == "We talked about evaporation."
    assert by_conversation[SESSION]["summary"] == "Goals: understand evaporation"
    assert by_conversation[older]["note"] is None
    assert len(record["summaries"]) == 2


def test_a_student_nothing_is_held_for_reads_as_empty(fake_postgres):
    fake_postgres(fetchall=[[]], fetchone=[None])
    record = held(USER, config=get_config())
    assert record["conversations"] == 0 and record["summaries"] == []
    assert record["stated_preferences"] is None


# --- destroying it ------------------------------------------------------------

def test_forgetting_one_conversation_leaves_the_preferences_alone(fake_postgres):
    db = fake_postgres(rowcount=4)
    forget(USER, conversation_id=SESSION, config=get_config())
    statements = [sql for sql, _ in db.statements]
    assert any("source_conversation_id" in sql for sql in statements)
    assert not any("adaptive_profile" in sql for sql in statements)


def test_forgetting_a_student_destroys_the_preferences_too(fake_postgres):
    """A preference is the one thing here never tied to a conversation, so it
    would otherwise survive a request to be forgotten."""
    db = fake_postgres(rowcount=9)
    assert forget(USER, config=get_config()) == {"facts_deleted": 9, "profile_deleted": True}
    statements = [sql for sql, _ in db.statements]
    assert any("DELETE FROM ai.user_memory WHERE user_id" in sql for sql in statements)
    assert any("DELETE FROM ai.adaptive_profile" in sql for sql in statements)
    assert db.committed


# --- the endpoints ------------------------------------------------------------

def test_reading_memory_needs_the_service_token(client):
    assert client.get(f"/ai/memory/{USER}").status_code in (401, 403)


def test_forgetting_needs_the_service_token(client):
    assert client.delete(f"/ai/memory/{USER}").status_code in (401, 403)


def test_the_endpoint_returns_what_is_held(client, fake_postgres, monkeypatch):
    fake_postgres(fetchall=[HELD_ROWS], fetchone=[("short", True, False, None)])
    body = client.get(f"/ai/memory/{USER}", headers=_auth(client)).json()
    assert body["conversations"] == 2
    assert body["summaries"][0]["summary"] == "Goals: understand evaporation"
    assert body["summaries"][0]["note"] == "We talked about evaporation."
    assert body["stated_preferences"]["length"] == "short"
    assert body["request_id"]


def test_the_endpoint_says_what_it_could_not_destroy(client, fake_postgres):
    """The platform owns the transcript and this cannot reach it. A student
    asking to be forgotten has to be forgotten there too, and a receipt that
    implied otherwise would be the wrong kind of wrong."""
    fake_postgres(rowcount=3)
    body = client.delete(f"/ai/memory/{USER}", headers=_auth(client)).json()
    assert body["facts_deleted"] == 3
    assert body["transcript_retained"] is True
    assert body["conversation_id"] is None


def test_one_conversation_can_be_forgotten_on_its_own(client, fake_postgres):
    fake_postgres(rowcount=2)
    body = client.delete(f"/ai/memory/{USER}?conversation_id={SESSION}",
                         headers=_auth(client)).json()
    assert body["conversation_id"] == SESSION and body["profile_deleted"] is False
