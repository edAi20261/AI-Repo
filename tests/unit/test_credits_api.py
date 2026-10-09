"""The credits endpoints, and what a paying caller sees when the balance runs out."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from agent.credits import Action, CreditCatalog, Credits, InMemoryCreditStore
from shared.config import get_config
from shared.metering import record_usage
from system.billing import Billing
from system.main import create_app
from system.runtime import AgentRuntime
from system.settings import Settings

TOKEN = "test-service-token"
HEADERS = {"x-service-token": TOKEN}
CHAT = {"user_id": "u-1", "user_type": "student", "message": "What is evaporation?"}


class _Graph:
    def invoke(self, _payload):
        record_usage({"cost": 0.004, "prompt_tokens": 3000, "completion_tokens": 400})
        return {"response": "Evaporation is ...", "language": "en"}

    async def astream(self, _payload, stream_mode=None):
        yield {"compose": {"response": "Evaporation is ...", "language": "en"}}


def _client(monkeypatch, credits):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("AI_WEBHOOK_SHARED_SECRET", TOKEN)
    config = get_config(reload=True)
    app = create_app(config)
    app.state.settings = Settings.from_config(config)
    billing = Billing(credits)

    @asynccontextmanager
    async def _lifespan(scoped_app):
        scoped_app.state.billing = billing
        scoped_app.state.runtime = AgentRuntime(config, _Graph(), None, billing=billing,
                                                max_concurrency=2, timeout=5)
        yield

    app.router.lifespan_context = _lifespan
    return TestClient(app)


@pytest.fixture
def credits():
    return Credits(InMemoryCreditStore(), CreditCatalog.from_config(get_config(reload=True)))


@pytest.fixture
def client(monkeypatch, credits):
    with _client(monkeypatch, credits) as test_client:
        yield test_client


def _exhaust(credits, user_id="u-1"):
    credits.assign_plan(user_id, "free", changed_by="admin")
    for _ in range(50):
        credits.charge(user_id, Action.CHAT_MESSAGE, units=1, reference_id=None)


def test_a_new_user_reads_as_the_default_plan_in_full(client):
    body = client.get("/ai/credits/u-1", headers=HEADERS).json()
    assert body["plan"] == "free" and body["balance"] == 50
    assert body["allowance"] == {"remaining": 50, "total": 50}


def test_a_chat_message_is_charged_and_shows_on_the_ledger(client):
    reply = client.post("/ai/chat", json=CHAT, headers=HEADERS)
    assert reply.status_code == 200 and reply.json()["usage"]["cost_usd"] == 0.004

    assert client.get("/ai/credits/u-1", headers=HEADERS).json()["balance"] == 49
    ledger = client.get("/ai/credits/u-1/ledger", headers=HEADERS).json()
    assert [entry["kind"] for entry in ledger["entries"]] == ["usage", "allowance_renewal"]
    assert ledger["entries"][0]["reference_id"] == reply.json()["request_id"]
    assert "cost_usd" not in ledger["entries"][0]


def test_an_empty_balance_is_a_402_the_platform_can_explain(client, credits):
    """A 402 with the renewal date is something the platform can turn into "your
    credits come back on the 29th"; a 500 or a 503 reads as the tutor being broken."""
    _exhaust(credits)
    response = client.post("/ai/chat", json=CHAT, headers=HEADERS)

    assert response.status_code == 402
    error = response.json()["error"]
    assert error["code"] == "insufficient_credits"
    assert error["details"]["required"] == 1 and "renews_at" in error["details"]


def test_an_empty_balance_is_a_402_on_the_streamed_endpoint_too(client, credits):
    """Refused inside the stream, the status would already be 200 and the refusal an
    SSE error event the client cannot tell from a crash."""
    _exhaust(credits)
    response = client.post("/ai/chat/stream", json=CHAT, headers=HEADERS)
    assert response.status_code == 402


def test_granted_credits_unblock_a_user_whose_allowance_is_spent(client, credits):
    _exhaust(credits)
    granted = client.post("/ai/credits/u-1/grants", headers=HEADERS,
                          json={"amount": 10, "granted_by": "admin-1", "note": "support"})
    assert granted.json()["extra"] == 10

    assert client.post("/ai/chat", json=CHAT, headers=HEADERS).status_code == 200


def test_a_plan_change_restarts_the_allowance_at_the_new_plan(client):
    body = client.put("/ai/credits/u-1/plan", headers=HEADERS,
                      json={"plan": "pro", "changed_by": "admin-1"}).json()
    assert (body["plan"], body["allowance"]["remaining"]) == ("pro", 2000)


def test_a_plan_that_does_not_exist_is_404(client):
    response = client.put("/ai/credits/u-1/plan", headers=HEADERS,
                          json={"plan": "platinum", "changed_by": "admin-1"})
    assert response.status_code == 404


def test_the_catalogue_lists_every_plan_and_every_priced_action(client):
    body = client.get("/ai/credits/plans", headers=HEADERS).json()
    assert body["default_plan"] == "free"
    assert {plan["code"] for plan in body["plans"]} == {"free", "basic", "pro"}
    assert {price["action"] for price in body["prices"]} == {action.value for action in Action}


def test_a_grant_must_be_positive(client):
    response = client.post("/ai/credits/u-1/grants", headers=HEADERS,
                           json={"amount": 0, "granted_by": "admin-1"})
    assert response.status_code == 422


def test_with_credits_disabled_the_credits_api_says_so(monkeypatch):
    """An empty balance served while credits are off would read as a real one."""
    with _client(monkeypatch, None) as client:
        response = client.get("/ai/credits/u-1", headers=HEADERS)
        assert response.status_code == 503
        assert client.post("/ai/chat", json=CHAT, headers=HEADERS).status_code == 200


@pytest.mark.parametrize("method, path, kwargs", [
    ("post", "/ai/grade", {"json": {"user_id": "s-1", "submissions": [
        {"question": "2+2?", "student_answer": "4"}]}}),
    ("post", "/ai/classes/report", {"json": {"class_id": "c1", "user_ids": ["s1"]}}),
    ("get", "/ai/reports/s-1", {}),
    ("post", "/ai/homework/capture", {"files": {"image": ("p.png", b"\x89PNG", "image/png")}}),
    ("post", "/ai/homework/mark", {"params": {"user_id": "s-1"},
                                   "files": {"image": ("p.png", b"\x89PNG", "image/png")}}),
])
def test_a_paid_endpoint_refuses_a_request_that_names_nobody_to_pay(client, method, path, kwargs):
    """Without the paying account the only choices are charging the wrong user or
    charging no one, and both are silent."""
    response = getattr(client, method)(path, headers=HEADERS, **kwargs)
    assert response.status_code == 422
