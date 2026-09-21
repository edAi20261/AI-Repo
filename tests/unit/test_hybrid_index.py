"""Named dense and sparse vectors in one collection — Stage 5.

A dense vector finds a passage that means the same thing. A sparse one finds the
passage that uses the same words: the name, the formula, the date a student
types exactly. Keeping both in one Qdrant collection lets the server fuse them,
instead of a dense search fetching candidates and Python re-ranking whatever
came back.
"""

from __future__ import annotations

from typing import Any

from ingestion.state import Chunk
from shared.config import get_config


class _FakeQdrant:
    """Records what a real client would have been asked to do."""

    def __init__(self, exists: bool = False) -> None:
        self.exists = exists
        self.created: dict[str, Any] = {}
        self.points: list = []

    def collection_exists(self, _name):
        return self.exists

    def create_collection(self, collection_name, **kwargs):
        self.created = {"name": collection_name, **kwargs}
        self.exists = True

    def get_collection(self, _name):
        return type("Info", (), {"payload_schema": {}})()

    def create_payload_index(self, *a, **k):
        pass

    def upsert(self, collection_name, points, wait=False):
        self.points.extend(points)


def _store(monkeypatch, hybrid: bool):
    from ingestion import stores

    config = get_config(reload=True)
    config._data.setdefault("vector_store", {})["hybrid"] = hybrid
    fake = _FakeQdrant()
    monkeypatch.setattr(stores, "upsert_client", lambda *a, **k: fake)
    return stores.QdrantVectorStore(config, "http://qdrant.invalid"), fake


def _chunk(**kw):
    base = dict(lecture_id="lec1", text="photosynthesis", heading_path="", page_from=1,
                page_to=1, language="en", order_index=0, chunk_id="c1",
                embedding=[0.1] * 8)
    base.update(kw)
    return Chunk(**base)


def test_a_hybrid_collection_carries_both_vectors(monkeypatch):
    store, fake = _store(monkeypatch, hybrid=True)
    store.ensure_collection(8)

    assert "dense" in fake.created["vectors_config"]
    assert "sparse" in fake.created["sparse_vectors_config"]


def test_the_collection_we_already_have_is_left_as_it_was(monkeypatch):
    """154 points live in an unnamed-vector collection. Naming the vector is a
    migration, not a config flip: every existing point would stop being found."""
    store, fake = _store(monkeypatch, hybrid=False)
    store.ensure_collection(8)

    assert fake.created["vectors_config"].__class__.__name__ == "VectorParams"
    assert "sparse_vectors_config" not in fake.created


def test_a_point_written_to_a_hybrid_collection_carries_its_terms(monkeypatch):
    store, fake = _store(monkeypatch, hybrid=True)
    store.upsert([_chunk(sparse={7: 1.5, 11: 0.5})])

    vector = fake.points[0].vector
    assert vector["dense"] == [0.1] * 8
    assert sorted(zip(vector["sparse"].indices, vector["sparse"].values)) == [
        (7, 1.5), (11, 0.5)]


def test_a_point_with_no_terms_is_still_written(monkeypatch):
    """A chunk of pure notation has no words to match on. It must still be
    findable by meaning rather than dropped from the index."""
    store, fake = _store(monkeypatch, hybrid=True)
    store.upsert([_chunk(sparse={})])

    assert len(fake.points) == 1
    assert fake.points[0].vector["dense"] == [0.1] * 8


def test_a_point_written_to_the_old_collection_is_unchanged(monkeypatch):
    store, fake = _store(monkeypatch, hybrid=False)
    store.upsert([_chunk(sparse={7: 1.5})])

    assert fake.points[0].vector == [0.1] * 8
