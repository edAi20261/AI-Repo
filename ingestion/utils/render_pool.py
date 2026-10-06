"""Rendering a book's pages on every core instead of one.

A page at 300 dpi is most of a second of CPU, not milliseconds. Measured on an
idle 2-vCPU machine over twenty pages of a scanned 1033 x 1375 pt textbook:
0.83 s a page on one thread, 0.45 s a page through two processes. The extract
stage renders on the thread that owns the document, because a PyMuPDF
``Document`` is not thread-safe — which is right for safety and leaves every
core but one idle while three hundred pages queue for it.

Processes, not threads: each opens its own copy of the book, so no document is
ever shared, and rendering runs in parallel without holding the interpreter
lock. Spawned rather than forked, because the parent already has HTTP threads
running and a forked copy of a lock held by one of them is a deadlock.
"""

from __future__ import annotations

import concurrent.futures as futures
import multiprocessing
import os
import threading
from typing import Optional

from . import pdf

_DOC = None
_DPI = 300


def _open(path: str, dpi: int) -> None:
    global _DOC, _DPI
    _DOC = pdf.open_document(path)
    _DPI = dpi


def _render(page_no: int) -> bytes:
    return pdf.render_page_png(_DOC[page_no - 1], dpi=_DPI)


def default_processes(configured: int) -> int:
    """``configured`` render processes, or one per core when it is 0."""
    return configured if configured > 0 else (os.cpu_count() or 1)


class PageRenderer:
    """``render(page_no)`` from any thread, on up to ``processes`` cores at once.

    One process renders in this one, behind a lock: the same pixels by the same
    function, without the start-up cost of a pool that would hold one worker.
    """

    def __init__(self, path: str, *, dpi: int, processes: int) -> None:
        self._pool: Optional[futures.ProcessPoolExecutor] = None
        self._doc = None
        self._lock = threading.Lock()
        self._dpi = dpi
        if processes > 1:
            self._pool = futures.ProcessPoolExecutor(
                max_workers=processes,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_open, initargs=(path, dpi))
        else:
            self._doc = pdf.open_document(path)

    def render(self, page_no: int) -> bytes:
        if self._pool is not None:
            return self._pool.submit(_render, page_no).result()
        with self._lock:
            return pdf.render_page_png(self._doc[page_no - 1], dpi=self._dpi)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
        if self._doc is not None:
            self._doc.close()

    def __enter__(self) -> "PageRenderer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
