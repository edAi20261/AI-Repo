"""Running a page's model calls while the next page's are already in flight.

A page costs seconds of arithmetic and a minute or more of waiting. The waiting
is a model answering over HTTP, and during it the task holds a whole vCPU and
eight gigabytes doing nothing — Fargate bills for the container existing, not
for it working, so idle time is paid time. Reading one page at a time meant a
three hundred page book held a machine for ten hours to do perhaps twenty
minutes of actual work.

So the calls overlap. What cannot overlap is PyMuPDF: a ``Document`` is not
thread-safe, and two threads rendering from one is a crash, not a slow path.
Everything that touches the document therefore stays on the calling thread, and
only the network calls are handed out. That is also the honest split — the
document work is milliseconds and the network is the minute.

Measured against Vertex on eight pages of a scanned Arabic question bank:
6.9s a page sequentially, 1.6s a page at eight in flight, 8/8 answered, no
rate limiting. 4.2x rather than the 8x that the worker count suggests, because
concurrent requests are individually slower — which is worth knowing before
anyone sets the number to fifty and expects fifty.
"""

from __future__ import annotations

import concurrent.futures as futures
import logging
from typing import Callable, Iterable, Iterator, Sequence, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

DEFAULT_WORKERS = 8


def in_order(
    items: Sequence[T],
    work: Callable[[T], R],
    *,
    workers: int = DEFAULT_WORKERS,
    label: str = "work",
) -> list[R]:
    """``[work(i) for i in items]``, computed concurrently, returned in order.

    Order is the whole point: pages carry their position in the book, and a
    result list that arrives in completion order would reassemble the book in
    the order the model happened to finish. ``ThreadPoolExecutor.map`` already
    guarantees this; it is named here so nobody has to remember that it does.

    One worker runs the items inline, which is not an optimisation — it is how
    a deployment turns this off without a separate code path to go stale.
    """
    if workers <= 1 or len(items) <= 1:
        return [work(item) for item in items]

    with futures.ThreadPoolExecutor(max_workers=workers,
                                    thread_name_prefix=label) as pool:
        return list(pool.map(work, items))


def streamed(
    items: Iterable[T],
    prepare: Callable[[T], R],
    work: Callable[[R], object],
    *,
    workers: int = DEFAULT_WORKERS,
    label: str = "work",
) -> list[object]:
    """``prepare`` on this thread, ``work`` on several, results in order.

    One pool for the whole run rather than a pool per batch. Batching looked
    equivalent and is not: a batch waits for its slowest member before the next
    one starts, so a single page that retries onto the fallback model holds
    seven idle workers while it finishes. Measured over twelve pages that cost
    1.20x against the 5.67x the same calls reach with no barrier between them.

    ``prepare`` stays on the calling thread because it touches PyMuPDF, which
    is not thread-safe. It also paces the submission: a page is only rendered
    when a worker is free to take it, so the number of rendered pages held in
    memory is bounded by the pool rather than by the length of the book.
    """
    if workers <= 1:
        return [work(prepare(item)) for item in items]

    submitted: list[futures.Future] = []
    with futures.ThreadPoolExecutor(max_workers=workers,
                                    thread_name_prefix=label) as pool:
        # A bounded queue: never let more than twice the worker count sit
        # rendered and waiting, or a long book renders itself into the heap.
        for item in items:
            while sum(1 for f in submitted if not f.done()) >= workers * 2:
                futures.wait(submitted, return_when=futures.FIRST_COMPLETED)
            submitted.append(pool.submit(work, prepare(item)))
        return [f.result() for f in submitted]


def windows(items: Iterable[T], size: int) -> Iterator[list[T]]:
    """``items`` in chunks of ``size``, the last one short.

    Pages are processed a window at a time rather than all at once because a
    rendered page is megabytes: a four hundred page book rendered up front is
    gigabytes of images held to save seconds of rendering, on a task with eight
    gigabytes to its name.
    """
    window: list[T] = []
    for item in items:
        window.append(item)
        if len(window) >= size:
            yield window
            window = []
    if window:
        yield window
