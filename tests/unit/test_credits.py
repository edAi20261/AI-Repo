"""Credits as a service: the operations the runtime and the credits API call."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agent.credits import Action, CreditCatalog, Credits, InMemoryCreditStore, PostgresCreditStore
from agent.credits.ledger import Kind
from shared.config import get_config
from shared.errors import InsufficientCreditsError, NotFoundError, ServiceUnavailableError


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 9, 29, 10, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now


def _credits(clock: _Clock | None = None) -> Credits:
    return Credits(InMemoryCreditStore(), CreditCatalog.from_config(get_config()),
                   clock=clock or _Clock())


def test_a_user_seen_for_the_first_time_starts_on_the_default_plan():
    summary = _credits().summary("u-1")
    assert (summary.plan, summary.balance, summary.allowance_total) == ("free", 50, 50)


def test_a_charge_takes_the_price_of_the_action_and_is_written_to_the_ledger():
    credits = _credits()
    charge = credits.charge("u-1", Action.GRADE_ANSWERS, units=11, reference_id="req-1")

    assert charge.kind is Kind.USAGE and charge.allowance_delta == -2
    assert credits.summary("u-1").balance == 48
    entries, total = credits.history("u-1", page=1, page_size=10)
    assert total == 2 and entries[0].reference_id == "req-1"


def test_a_user_with_too_few_credits_is_refused_before_anything_runs():
    """The refusal carries what is needed and when the allowance comes back, so the
    platform can tell the student something more useful than 'no'."""
    credits = _credits()
    credits.assign_plan("u-1", "free", changed_by="admin")
    for _ in range(50):
        credits.charge("u-1", Action.CHAT_MESSAGE, units=1, reference_id=None)

    with pytest.raises(InsufficientCreditsError) as refused:
        credits.charge("u-1", Action.CHAT_MESSAGE, units=1, reference_id=None)
    assert refused.value.status_code == 402
    assert refused.value.details["balance"] == 0 and refused.value.details["required"] == 1


def test_a_refund_restores_the_balance_and_records_what_it_reverses():
    credits = _credits()
    charge = credits.charge("u-1", Action.CHAT_MESSAGE, units=1, reference_id="req-1")
    refund = credits.refund(charge)

    assert credits.summary("u-1").balance == 50
    assert refund.kind is Kind.REFUND and refund.reverses_id == charge.id


def test_granted_credits_survive_the_monthly_renewal():
    clock = _Clock()
    credits = _credits(clock)
    credits.grant("u-1", 30, granted_by="admin", note="outage")
    credits.charge("u-1", Action.CHAT_MESSAGE, units=1, reference_id=None)

    clock.now += timedelta(days=31)
    summary = credits.summary("u-1")
    assert (summary.allowance_remaining, summary.extra) == (50, 30)


def test_moving_to_another_plan_restarts_the_allowance_at_that_plan():
    credits = _credits()
    credits.charge("u-1", Action.CHAT_MESSAGE, units=1, reference_id=None)
    summary = credits.assign_plan("u-1", "basic", changed_by="admin")
    assert (summary.plan, summary.allowance_remaining) == ("basic", 500)


def test_a_plan_the_catalogue_does_not_have_is_not_found():
    with pytest.raises(NotFoundError):
        _credits().assign_plan("u-1", "platinum", changed_by="admin")


def test_the_postgres_store_serialises_each_user_behind_one_lock(fake_postgres):
    """Two turns for one student read the same balance and both spend it unless the
    second waits for the first. The lock is also what makes opening a wallet for a new
    user safe when two of their requests arrive together."""
    created = datetime(2026, 9, 29, tzinfo=timezone.utc)
    db = fake_postgres(fetchone=[None, ("entry-1", created)])
    store = PostgresCreditStore(get_config())
    credits = Credits(store, CreditCatalog.from_config(get_config()), clock=_Clock())

    credits.summary("u-1")

    assert db.statements[0][0].startswith("SELECT pg_advisory_xact_lock")
    assert any("INSERT INTO ai.credit_wallets" in sql for sql, _ in db.statements)
    assert db.committed


def test_an_unreachable_credits_database_refuses_the_turn_rather_than_giving_it_away(
        fake_postgres):
    """Failing open would hand every student free usage for as long as the database
    is down, and nobody would see it until the invoice."""
    fake_postgres(explode=True)
    credits = Credits(PostgresCreditStore(get_config()), CreditCatalog.from_config(get_config()),
                      clock=_Clock())

    with pytest.raises(ServiceUnavailableError):
        credits.charge("u-1", Action.CHAT_MESSAGE, units=1, reference_id=None)
