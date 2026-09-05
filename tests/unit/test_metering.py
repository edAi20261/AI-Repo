"""LLM usage/cost metering — offline (no API): feed usage dicts and assert totals."""

from __future__ import annotations

import contextvars

from shared.metering import (
    MeteringCallback,
    record_usage,
    start_meter,
)


def test_record_usage_accumulates_cost_and_tokens():
    meter = start_meter()
    record_usage({"cost": 0.012, "prompt_tokens": 100, "completion_tokens": 40})
    record_usage({"cost": 0.008, "prompt_tokens": 50, "completion_tokens": 10})
    assert meter.as_dict() == {
        "cost_usd": 0.02, "prompt_tokens": 150, "completion_tokens": 50, "calls": 2}


def test_record_usage_reads_alternate_token_keys():
    meter = start_meter()
    record_usage({"total_cost": 0.005, "input_tokens": 30, "output_tokens": 7})
    d = meter.as_dict()
    assert d["cost_usd"] == 0.005 and d["prompt_tokens"] == 30 and d["completion_tokens"] == 7


def test_record_usage_ignores_non_dict_but_counts_empty_call():
    meter = start_meter()
    record_usage(None)          # not a dict -> fully ignored
    record_usage("nope")        # not a dict -> ignored
    assert meter.as_dict()["calls"] == 0
    record_usage({})            # dict, no fields -> a zero-cost call is still recorded
    assert meter.as_dict() == {
        "cost_usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "calls": 1}


def test_record_usage_noop_in_context_without_meter():
    # In a fresh context no meter is installed; recording must not raise.
    contextvars.Context().run(record_usage, {"cost": 1.0})


def test_metering_callback_reads_llm_output_token_usage():
    meter = start_meter()

    class _Resp:
        llm_output = {"token_usage": {
            "cost": 0.03, "prompt_tokens": 200, "completion_tokens": 80}}

    MeteringCallback().on_llm_end(_Resp())
    d = meter.as_dict()
    assert d["cost_usd"] == 0.03 and d["calls"] == 1 and d["prompt_tokens"] == 200
