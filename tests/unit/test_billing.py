"""Charging around the runtime: when a credit is taken, kept, and given back."""

from __future__ import annotations

import time

import anyio
import pytest

from agent.credits import Action, CreditCatalog, Credits, InMemoryCreditStore, Kind
from shared.config import get_config
from shared.errors import InsufficientCreditsError, UpstreamTimeoutError
from shared.metering import record_usage


def _runtime(graph, credits, *, timeout=5):
    from system.billing import Billing
    from system.runtime import AgentRuntime

    return AgentRuntime(get_config(reload=True), graph, None, billing=Billing(credits),
                        max_concurrency=2, timeout=timeout)


def _credits():
    return Credits(InMemoryCreditStore(), CreditCatalog.from_config(get_config(reload=True)))


def _kinds(credits, user_id="u-1"):
    entries, _ = credits.history(user_id, page=1, page_size=50)
    return [entry.kind for entry in reversed(entries)]


class _Graph:
    def __init__(self, *, fail=False, sleep=0.0):
        self.fail, self.sleep, self.calls = fail, sleep, 0

    def invoke(self, _payload):
        self.calls += 1
        time.sleep(self.sleep)
        if self.fail:
            raise RuntimeError("model gateway down")
        record_usage({"cost": 0.004, "prompt_tokens": 3000, "completion_tokens": 400})
        return {"response": "Evaporation is ..."}

    async def astream(self, _payload, stream_mode=None):
        record_usage({"cost": 0.004, "prompt_tokens": 3000, "completion_tokens": 400})
        yield {"compose": {"response": "Evaporation is ..."}}


@pytest.fixture(autouse=True)
def _test_env(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")


def test_an_answered_turn_keeps_its_charge_and_records_what_it_cost():
    """The charge is the price; the cost beside it is the bill. Without both the
    price table can never be checked against what the platform actually pays."""
    credits = _credits()
    anyio.run(_runtime(_Graph(), credits).chat, {"user_id": "u-1", "message": "hi"})

    assert credits.summary("u-1").balance == 49
    entries, _ = credits.history("u-1", page=1, page_size=5)
    assert entries[0].kind is Kind.USAGE and entries[0].cost_usd == 0.004


def test_a_turn_that_fails_gives_the_credit_back():
    """A student who got an error instead of an answer must not pay for it."""
    credits = _credits()
    with pytest.raises(RuntimeError):
        anyio.run(_runtime(_Graph(fail=True), credits).chat, {"user_id": "u-1", "message": "hi"})

    assert credits.summary("u-1").balance == 50
    assert _kinds(credits) == [Kind.ALLOWANCE_RENEWAL, Kind.USAGE, Kind.REFUND]


def test_a_turn_that_times_out_gives_the_credit_back():
    credits = _credits()
    runtime = _runtime(_Graph(sleep=1), credits, timeout=0.1)
    with pytest.raises(UpstreamTimeoutError):
        anyio.run(runtime.chat, {"user_id": "u-1", "message": "hi"})

    assert credits.summary("u-1").balance == 50


def test_a_student_without_credits_is_refused_before_any_model_is_called():
    """Charging after the turn would mean paying OpenRouter for an answer the
    student is then refused."""
    credits = _credits()
    credits.assign_plan("u-1", "free", changed_by="admin")
    for _ in range(50):
        credits.charge("u-1", Action.CHAT_MESSAGE, units=1, reference_id=None)
    graph = _Graph()

    with pytest.raises(InsufficientCreditsError):
        anyio.run(_runtime(graph, credits).chat, {"user_id": "u-1", "message": "hi"})
    assert graph.calls == 0


def test_a_streamed_turn_is_refused_before_the_stream_starts():
    """Once the first event is sent the status is 200 and a refusal can only be an
    error event, which a client reads as a crash rather than an empty balance."""
    credits = _credits()
    credits.assign_plan("u-1", "free", changed_by="admin")
    for _ in range(50):
        credits.charge("u-1", Action.CHAT_MESSAGE, units=1, reference_id=None)

    with pytest.raises(InsufficientCreditsError):
        anyio.run(_runtime(_Graph(), credits).stream_chat, {"user_id": "u-1", "message": "hi"})


def test_a_streamed_turn_is_charged_once_and_keeps_the_charge():
    credits = _credits()
    runtime = _runtime(_Graph(), credits)

    async def run():
        updates = await runtime.stream_chat({"user_id": "u-1", "message": "hi"})
        return [stage async for stage, _ in updates]

    assert anyio.run(run)[-1] == "__final__"
    assert credits.summary("u-1").balance == 49


def test_a_degraded_grade_is_not_paid_for(monkeypatch):
    """Grading that fell back to string matching is a safe answer, not the one that
    was paid for, and it arrives as a result rather than an error."""
    import system.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "run_grade",
                        lambda payload, config: {"graded": [], "degraded": True})
    credits = _credits()
    runtime = _runtime(_Graph(), credits)

    async def run():
        return await runtime.grade({"user_id": "s-1", "submissions": [{}] * 11},
                                   billed_user_id="t-1")

    assert anyio.run(run).result["degraded"] is True
    assert _kinds(credits, "t-1") == [Kind.ALLOWANCE_RENEWAL, Kind.USAGE, Kind.REFUND]


def test_grading_is_charged_to_whoever_pays_by_the_batch(monkeypatch):
    import system.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "run_grade",
                        lambda payload, config: {"graded": [], "degraded": False})
    credits = _credits()
    runtime = _runtime(_Graph(), credits)

    async def run():
        await runtime.grade({"user_id": "s-1", "submissions": [{}] * 11}, billed_user_id="t-1")

    anyio.run(run)
    assert credits.summary("t-1").balance == 48
    assert _kinds(credits, "s-1") == []


def test_with_credits_disabled_nothing_is_charged_and_turns_still_run():
    graph = _Graph()
    turn = anyio.run(_runtime(graph, None).chat, {"user_id": "u-1", "message": "hi"})
    assert turn.result == {"response": "Evaporation is ..."} and graph.calls == 1


@pytest.mark.parametrize("marked, refunded", [
    ({"unreadable": True}, True),
    ({"degraded": True}, True),
    ({}, False),
])
def test_a_homework_page_is_paid_for_only_when_it_was_marked(monkeypatch, marked, refunded):
    """A photo nobody could read, or a page the grader failed on, came back with no
    marks on it. Charging for it is charging for nothing."""
    import system.runtime as runtime_module
    from agent.homework import MarkedHomework

    monkeypatch.setattr(runtime_module, "mark",
                        lambda content, config, **payload: MarkedHomework(**marked))
    credits = _credits()
    runtime = _runtime(_Graph(), credits)

    async def run():
        return await runtime.mark_homework(b"png", {"user_id": "s-1"}, billed_user_id="t-1")

    anyio.run(run)
    expected = [Kind.ALLOWANCE_RENEWAL, Kind.USAGE] + ([Kind.REFUND] if refunded else [])
    assert _kinds(credits, "t-1") == expected
