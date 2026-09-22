"""Searching a collection that holds both vectors.

The gain from a sparse index is not the fusion — dense candidates were already
re-ranked lexically in Python. It is that the lexical half can now *find* a
chunk the dense half missed entirely, instead of only reordering what dense
brought back.

The score must stay the dense cosine. ``agent.min_score`` is measured against a
similarity, and a fused rank score in its place would silently disable the floor
that stops the tutor answering what the syllabus cannot.
"""

from __future__ import annotations

from typing import Any

from shared.config import get_config


class _Hit:
    def __init__(self, id: str, score: float, payload: dict | None = None) -> None:
        self.id = id
        self.score = score
        self.payload = payload or {}


class _FakeQdrant:
    """Answers a dense query, a sparse query, and a dense query by id."""

    def __init__(self, dense: list[_Hit], sparse: list[_Hit], by_id: list[_Hit],
                 named: bool = True) -> None:
        self._dense, self._sparse, self._by_id = dense, sparse, by_id
        self._named = named
        self.queries: list[dict[str, Any]] = []

    def get_collection(self, _name):
        params = type("P", (), {"vectors": {"dense": object()} if self._named
                                else object()})()
        return type("Info", (), {"config": type("C", (), {"params": params})()})()

    def query_points(self, collection, **kwargs):
        self.queries.append(kwargs)
        if kwargs.get("using") == "sparse":
            points = self._sparse
        elif len(self.queries) > 2:
            points = self._by_id          # the rescore of what only words found
        else:
            points = self._dense
        return type("R", (), {"points": points})()


def _config(hybrid: bool):
    from shared import vectors

    vectors.has_named_vectors.cache_clear()
    config = get_config(reload=True)
    config._data.setdefault("vector_store", {})["hybrid"] = hybrid
    return config


def test_a_chunk_only_the_words_can_find_enters_the_pool(monkeypatch):
    """A student types a name the dense vector averaged away. Before a sparse
    index that chunk was never a candidate, so no amount of re-ranking could
    surface it."""
    from agent import tools

    fake = _FakeQdrant(
        dense=[_Hit("a", 0.61), _Hit("b", 0.55)],
        sparse=[_Hit("c", 8.2), _Hit("a", 6.0)],
        by_id=[_Hit("c", 0.47)],
    )
    monkeypatch.setattr(tools, "query_client", lambda config: fake)
    monkeypatch.setattr(tools, "get_embedder",
                        lambda config: type("E", (), {
                            "embed_query": lambda self, q: [0.1] * 8,
                            "embed_sparse": lambda self, t: [{7: 1.0}]})())

    scores = tools._candidate_scores(
        "جابر بن حيان", [], _config(True), collection="eduai_chunks", limit=10)

    assert set(scores) == {"a", "b", "c"}
    # The sparse-only chunk carries its dense cosine, not its sparse score.
    assert scores["c"] == 0.47


def test_the_score_stays_a_similarity(monkeypatch):
    """0.47 is comparable to agent.min_score. A fused rank score is not."""
    from agent import tools

    fake = _FakeQdrant(dense=[_Hit("a", 0.61)], sparse=[_Hit("a", 9.9)], by_id=[])
    monkeypatch.setattr(tools, "query_client", lambda config: fake)
    monkeypatch.setattr(tools, "get_embedder",
                        lambda config: type("E", (), {
                            "embed_query": lambda self, q: [0.1] * 8,
                            "embed_sparse": lambda self, t: [{7: 1.0}]})())

    scores = tools._candidate_scores(
        "q", [], _config(True), collection="eduai_chunks", limit=10)
    assert scores["a"] == 0.61


def test_a_collection_with_one_vector_is_searched_the_way_it_always_was(monkeypatch):
    """154 points live in an unnamed collection. Naming a vector in the query
    finds none of them."""
    from agent import tools

    fake = _FakeQdrant(dense=[_Hit("a", 0.61)], sparse=[], by_id=[], named=False)
    monkeypatch.setattr(tools, "query_client", lambda config: fake)
    monkeypatch.setattr(tools, "get_embedder",
                        lambda config: type("E", (), {
                            "embed_query": lambda self, q: [0.1] * 8,
                            "embed_sparse": lambda self, t: [{7: 1.0}]})())

    tools._candidate_scores("q", [], _config(False), collection="eduai_chunks", limit=10)

    assert len(fake.queries) == 1
    assert "using" not in fake.queries[0]


# --- the collection knows its own shape -------------------------------------

class _Shaped:
    """A client whose collection reports how its vectors are configured."""

    def __init__(self, named: bool) -> None:
        self.named = named
        self.queries: list[dict] = []

    def get_collection(self, _name):
        params = type("P", (), {"vectors": {"dense": object()} if self.named
                                else object()})()
        return type("Info", (), {"config": type("C", (), {"params": params})()})()

    def query_points(self, collection, **kwargs):
        self.queries.append(kwargs)
        return type("R", (), {"points": []})()


def test_a_collection_without_named_vectors_is_not_queried_by_name(monkeypatch):
    """APP_ENV=local points at eduai_chunks_dev, which has one unnamed vector.
    vector_store.hybrid is true in the base config, so every query named a
    vector that collection does not have and Qdrant answered 400 — a 500 on
    every turn, where the honest outcome was "nothing found". A setting cannot
    be the authority on a collection's shape; the collection is."""
    from agent import tools
    from shared import vectors

    vectors.has_named_vectors.cache_clear()
    fake = _Shaped(named=False)
    monkeypatch.setattr(tools, "query_client", lambda config: fake)
    monkeypatch.setattr(tools, "get_embedder",
                        lambda config: type("E", (), {
                            "embed_query": lambda self, q: [0.1] * 8,
                            "embed_sparse": lambda self, t: [{7: 1.0}]})())

    tools._candidate_scores("q", [], _config(True),
                            collection="eduai_chunks_dev", limit=10)

    assert "using" not in fake.queries[0]
    assert len(fake.queries) == 1, "no sparse query against a collection without one"


def test_a_collection_with_named_vectors_is_queried_by_name(monkeypatch):
    from agent import tools
    from shared import vectors

    vectors.has_named_vectors.cache_clear()
    fake = _Shaped(named=True)
    monkeypatch.setattr(tools, "query_client", lambda config: fake)
    monkeypatch.setattr(tools, "get_embedder",
                        lambda config: type("E", (), {
                            "embed_query": lambda self, q: [0.1] * 8,
                            "embed_sparse": lambda self, t: [{7: 1.0}]})())

    tools._candidate_scores("q", [], _config(True),
                            collection="eduai_chunks_v2", limit=10)

    assert fake.queries[0]["using"] == "dense"
