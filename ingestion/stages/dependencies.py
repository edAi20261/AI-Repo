"""dependencies stage — derive topic -> prerequisite edges.

Runs after ``topics``. For each of the lesson's topics an LLM proposes
prerequisites from the subject's topic catalog (one call per topic); the raw
edges are cleaned deterministically (self-edges dropped, out-degree capped,
cycles broken) and written to ``ai.topic_dependencies``. Stub returns none when
no chat key is set.
"""

from __future__ import annotations

from ..context import PipelineContext
from ..state import Dependency, IngestionState


def run(state: IngestionState, ctx: PipelineContext) -> IngestionState:
    topics = state.get("topics", [])
    stats = {"edges": 0}
    deriver = ctx.dependency_deriver
    if deriver.is_stub:
        # Stub means no chat key: derive nothing and leave existing edges alone.
        state["dependencies"] = []
        state.setdefault("stats", {})["dependencies"] = stats
        return state

    # Real run re-derives this lecture's edges, so drop its old ones first
    # (edges are ingestion-owned facts, not student history — safe to replace).
    ctx.chunk_store.delete_lecture_dependencies(state["lecture_id"])
    if len(topics) < 2:
        # Need >= 2 topics for an edge.
        state["dependencies"] = []
        state.setdefault("stats", {})["dependencies"] = stats
        return state

    # Candidate pool: this lesson's topics + the subject's existing catalog.
    label_to_id: dict[str, str] = {}
    for t in topics:
        label_to_id[t["label"]] = t["topic_id"]
    for topic_id, label, _emb in ctx.chunk_store.read_topic_catalog(state["subject_id"]):
        label_to_id.setdefault(label, topic_id)

    cfg = ctx.config.get("ingestion.dependencies", {})
    max_candidates = cfg.get("max_candidates", 0)
    vecs = _candidate_vectors(ctx, list(label_to_id), max_candidates)

    raw: list[Dependency] = []
    for t in topics:
        candidates = [label for label in label_to_id if label != t["label"]]
        if vecs and len(candidates) > max_candidates:
            candidates = _nearest(t["label"], candidates, vecs, max_candidates)
        for proposed in deriver.derive(t["label"], candidates):
            prereq_id = label_to_id.get(proposed["prereq"])
            if prereq_id:
                raw.append(
                    Dependency(
                        topic_id=t["topic_id"],
                        prereq_topic_id=prereq_id,
                        strength=proposed.get("strength", "required"),
                        rationale=proposed.get("rationale"),
                    )
                )

    edges = _clean(
        raw,
        drop_self=cfg.get("drop_self_edges", True),
        max_out_degree=cfg.get("max_out_degree", 5),
        break_cycles=cfg.get("break_cycles", True),
    )
    ctx.chunk_store.write_dependencies(edges)

    state["dependencies"] = [
        {"topic_id": e.topic_id, "prereq_topic_id": e.prereq_topic_id} for e in edges
    ]
    stats["edges"] = len(edges)
    state.setdefault("stats", {})["dependencies"] = stats
    return state


def _clean(
    edges: list[Dependency],
    *,
    drop_self: bool,
    max_out_degree: int,
    break_cycles: bool,
) -> list[Dependency]:
    """Drop self-edges and duplicates, cap out-degree, and skip edges that would
    close a cycle (an edge means topic -> prereq; prereq must not already reach
    topic)."""
    adjacency: dict[str, set[str]] = {}   # topic -> prereqs, for cycle checks
    out_degree: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    kept: list[Dependency] = []

    for edge in edges:
        pair = (edge.topic_id, edge.prereq_topic_id)
        if drop_self and edge.topic_id == edge.prereq_topic_id:
            continue
        if pair in seen:
            continue
        if out_degree.get(edge.topic_id, 0) >= max_out_degree:
            continue
        if break_cycles and _reaches(adjacency, edge.prereq_topic_id, edge.topic_id):
            continue
        seen.add(pair)
        adjacency.setdefault(edge.topic_id, set()).add(edge.prereq_topic_id)
        out_degree[edge.topic_id] = out_degree.get(edge.topic_id, 0) + 1
        kept.append(edge)
    return kept


def _candidate_vectors(ctx, labels: list[str], max_candidates: int) -> dict[str, list[float]]:
    """Embed the candidate labels once, but only when a cap is actually in force and
    an embedder is available; otherwise ``{}`` (no capping)."""
    embedder = getattr(ctx, "embedder", None)
    if not max_candidates or embedder is None:
        return {}
    return {lbl: vec for lbl, vec in zip(labels, embedder.embed(labels))}


def _nearest(
    target: str, candidates: list[str], vecs: dict[str, list[float]], k: int
) -> list[str]:
    """The ``k`` candidates whose label embedding is closest to ``target`` (cosine on
    normalised vectors). Keeps prerequisite candidates plausible as the catalog grows."""
    tv = vecs[target]
    ranked = sorted(candidates, key=lambda c: _dot(tv, vecs[c]), reverse=True)
    return ranked[:k]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _reaches(adjacency: dict[str, set[str]], start: str, target: str) -> bool:
    """True if ``target`` is reachable from ``start`` along the edges so far."""
    stack = [start]
    seen: set[str] = set()
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adjacency.get(node, ()))
    return False
