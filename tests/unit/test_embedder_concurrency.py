"""One embedder serves every turn in the process, from several threads at once.

The service runs up to ``api.max_concurrency`` turns in worker threads and they
share one BGE-M3. Its fast tokenizer is Rust behind a RefCell: a second thread
entering it while the first is inside raises ``RuntimeError: Already borrowed``.
With thirty students asking at once in production, one turn in thirty failed
with a 500 at the retrieve step.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from shared.embeddings import LocalSentenceTransformerEmbedder


class _Borrowed:
    def __init__(self) -> None:
        self._inside = False

    def __enter__(self):
        if self._inside:
            raise RuntimeError("Already borrowed")
        self._inside = True
        time.sleep(0.01)

    def __exit__(self, *exc):
        self._inside = False


class _Tokenizer:
    def __init__(self, borrow: _Borrowed) -> None:
        self._borrow = borrow

    def __call__(self, text, add_special_tokens=False):
        with self._borrow:
            return {"input_ids": [len(word) for word in text.split()]}


class _Model:
    def __init__(self, name, device=None) -> None:
        self._borrow = _Borrowed()
        self.tokenizer = _Tokenizer(self._borrow)

    def get_embedding_dimension(self) -> int:
        return 4

    def encode(self, texts, **kwargs):
        with self._borrow:
            return np.ones((len(texts), 4))


def test_concurrent_turns_never_enter_the_tokenizer_together(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=_Model))
    embedder = LocalSentenceTransformerEmbedder("bge-m3")
    start = threading.Barrier(16)

    def turn(i: int) -> None:
        start.wait()
        embedder.embed_query(f"what is an isotope {i}")
        embedder.embed_sparse([f"what is an isotope {i}"])

    with ThreadPoolExecutor(max_workers=16) as pool:
        for future in [pool.submit(turn, i) for i in range(16)]:
            future.result()
