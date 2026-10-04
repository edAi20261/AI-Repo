"""Pages are read several at a time, and come back in the order they were in.

A page is seconds of arithmetic and a minute of waiting on HTTP. Fargate bills
for the container existing rather than working, so reading one page at a time
held a machine for ten hours to do twenty minutes of work.
"""

from __future__ import annotations

import threading
import time

import pytest

from ingestion.concurrency import DEFAULT_WORKERS, in_order, windows


def test_results_come_back_in_the_order_they_went_in():
    """The whole point. A page carries its position in the book, and results in
    completion order would reassemble the book in whatever order the model
    happened to finish."""
    # Reverse the durations, so finishing order is the opposite of input order.
    items = [0.05, 0.04, 0.03, 0.02, 0.01]

    def work(delay):
        time.sleep(delay)
        return delay

    assert in_order(items, work, workers=5) == items


def test_the_calls_actually_overlap():
    started = threading.Barrier(4, timeout=5)

    def work(_):
        started.wait()        # only passes if four are in flight together
        return True

    assert in_order([1, 2, 3, 4], work, workers=4) == [True] * 4


def test_one_worker_runs_inline_so_a_deployment_can_turn_it_off():
    """Not an optimisation — it is how this is disabled without a second code
    path that goes stale."""
    threads = []

    def work(item):
        threads.append(threading.current_thread().name)
        return item

    assert in_order([1, 2, 3], work, workers=1) == [1, 2, 3]
    assert set(threads) == {threading.current_thread().name}


def test_a_failure_in_one_page_is_not_swallowed():
    """A page that raises must surface, not come back as a quietly missing
    page that is then indexed as an empty one."""
    def work(item):
        if item == 2:
            raise ValueError("this page exploded")
        return item

    with pytest.raises(ValueError, match="exploded"):
        in_order([1, 2, 3], work, workers=3)


def test_an_empty_list_does_no_work():
    assert in_order([], lambda x: x, workers=8) == []


# --- windowing ---------------------------------------------------------------

def test_pages_are_handed_over_a_window_at_a_time():
    """A rendered page is megabytes; a four hundred page book rendered up front
    is gigabytes held on a task with eight to its name."""
    assert list(windows(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]


def test_a_window_larger_than_the_book_is_one_window():
    assert list(windows([1, 2], 8)) == [[1, 2]]


def test_nothing_in_produces_nothing_out():
    assert list(windows([], 4)) == []


def test_the_shipped_default_is_what_was_measured():
    """8 workers: 1.6s a page against 6.9s sequential, 8/8 answered, no rate
    limiting — measured on Vertex against a scanned Arabic question bank."""
    assert DEFAULT_WORKERS == 8
