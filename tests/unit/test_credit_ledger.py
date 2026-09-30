"""The credit arithmetic, with no store and no clock but the one passed in."""

from __future__ import annotations

from datetime import datetime, timezone

from agent.credits import ledger
from agent.credits.catalog import Plan, Price
from agent.credits.ledger import Kind

NOW = datetime(2026, 9, 29, 10, 0, tzinfo=timezone.utc)
FREE = Plan("free", 50)


def _wallet(allowance: int, extra: int, *, ends: datetime | None = None) -> ledger.Wallet:
    return ledger.Wallet(
        user_id="u-1", plan="free", allowance=allowance, extra=extra,
        period_started_at=NOW, period_ends_at=ends or datetime(2026, 10, 29, tzinfo=timezone.utc))


def test_a_new_wallet_opens_with_its_plan_allowance_on_the_ledger():
    """A balance that did not come from a ledger entry cannot be audited: the sum of
    a user's entries must always equal what the wallet says they hold."""
    opened = ledger.open_wallet("u-1", FREE, NOW)
    assert opened.wallet.balance == 50
    assert [(e.kind, e.allowance_delta, e.balance_after) for e in opened.entries] == [
        (Kind.ALLOWANCE_RENEWAL, 50, 50)]


def test_usage_spends_the_monthly_allowance_before_bonus_credits():
    """Bonus credits never expire and the allowance does, so spending bonus first
    would let a student lose credits at renewal that they could have used."""
    charged = ledger.debit(_wallet(allowance=1, extra=5), 3, action="chat_message",
                           reference_id="req-1")
    assert charged is not None
    assert (charged.wallet.allowance, charged.wallet.extra) == (0, 3)
    assert (charged.entries[0].allowance_delta, charged.entries[0].extra_delta) == (-1, -2)


def test_a_debit_larger_than_the_whole_balance_is_refused():
    assert ledger.debit(_wallet(allowance=1, extra=1), 3, action="chat_message",
                        reference_id=None) is None


def test_a_refund_returns_credits_to_the_buckets_they_came_from():
    """A refund into the wrong bucket turns an expiring credit into a permanent one."""
    charged = ledger.debit(_wallet(allowance=1, extra=5), 3, action="chat_message",
                           reference_id="req-1")
    assert charged is not None
    refunded = ledger.reverse(charged.wallet, charged.entries[0])
    assert (refunded.wallet.allowance, refunded.wallet.extra) == (1, 5)
    assert refunded.entries[0].kind is Kind.REFUND


def test_the_allowance_renews_only_once_its_period_has_ended():
    current = _wallet(allowance=3, extra=2, ends=datetime(2026, 10, 1, tzinfo=timezone.utc))
    assert ledger.renew_if_due(current, FREE, NOW).entries == ()

    ended = _wallet(allowance=3, extra=2, ends=NOW)
    renewed = ledger.renew_if_due(ended, FREE, NOW)
    assert (renewed.wallet.allowance, renewed.wallet.extra) == (50, 2)
    assert renewed.entries[0].allowance_delta == 47
    assert renewed.wallet.period_ends_at == datetime(2026, 10, 29, 10, 0, tzinfo=timezone.utc)


def test_a_period_starting_on_the_31st_ends_on_the_last_day_of_a_shorter_month():
    """Adding a month to 31 January naively lands in March and skips February's renewal."""
    _, ends = ledger.period_starting_at(datetime(2026, 1, 31, tzinfo=timezone.utc))
    assert ends == datetime(2026, 2, 28, tzinfo=timezone.utc)


def test_a_price_per_batch_rounds_a_partial_batch_up():
    """Eleven answers graded at one credit per ten must cost two, not one."""
    price = Price(credits=1, per=10)
    assert [price.of(units) for units in (1, 10, 11, 200)] == [1, 1, 2, 20]
