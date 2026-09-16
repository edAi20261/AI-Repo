"""Re-embed the indexed chunks in place, without re-running ingestion.

Changing what we embed — a new model, a new normalisation — invalidates every
stored vector, because a query and a document must be embedded the same way to
be comparable. Re-running the pipeline would do it, but at a price: vision
transcription for every page, a caption for every figure, topic extraction and
one LLM call per topic. And ``ingestion/stages/index.py`` deletes the lecture
first, so every chunk_id is re-minted and the topic tags, figure links and
citation history that point at them are orphaned.

None of that is needed. The chunk text is already in ``ai.chunks``, and Qdrant
upserts by point id. Reading the text, embedding it again and upserting under
the SAME chunk_id costs nothing and changes nothing else.

    .venv/bin/python scripts/reembed_chunks.py                # report only
    .venv/bin/python scripts/reembed_chunks.py --apply        # write

Payload labels (subject, lesson title, lesson order) live only on the Qdrant
point, so they are read back from the existing point and rewritten unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg
from qdrant_client import QdrantClient, models

from agent.config import Config
from agent.tools import pg_url

PAYLOAD_ONLY_FIELDS = ("subject_name", "lesson_title", "lesson_order")


def _existing_points(client: QdrantClient, collection: str) -> dict[str, dict]:
    """Every point's payload, keyed by chunk_id, so nothing is lost on rewrite."""
    found: dict[str, dict] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection, limit=256, with_payload=True, with_vectors=False,
            offset=offset)
        for point in points:
            found[str(point.id)] = dict(point.payload or {})
        if offset is None:
            return found


def _chunk_texts(config: Config, chunk_ids: list[str]) -> dict[str, str]:
    with psycopg.connect(pg_url(config.secret("DATABASE_URL")),
                         connect_timeout=15) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT chunk_id::text, text FROM ai.chunks "
            "WHERE chunk_id::text = ANY(%s)", (chunk_ids,))
        return dict(cur.fetchall())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="write the new vectors; otherwise only report")
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    config = Config.load()
    collection = config.vector_store.get("collection", "eduai_chunks")
    client = QdrantClient(url=config.secret("QDRANT_URL"),
                          api_key=config.secret("QDRANT_API_KEY"), timeout=120)

    payloads = _existing_points(client, collection)
    print(f"collection {collection}: {len(payloads)} points")
    if not payloads:
        return 0

    texts = _chunk_texts(config, list(payloads))
    missing = [cid for cid in payloads if cid not in texts]
    if missing:
        # A point with no row behind it cannot be re-embedded from source. Leave
        # it exactly as it is rather than guessing at its text.
        print(f"  {len(missing)} point(s) have no ai.chunks row — leaving untouched")

    todo = [cid for cid in payloads if cid in texts]
    print(f"  {len(todo)} to re-embed")
    if not args.apply:
        print("\nreport only — pass --apply to write")
        return 0

    from shared.embeddings import get_embedder
    embedder = get_embedder(config)

    written = 0
    for start in range(0, len(todo), args.batch):
        batch = todo[start:start + args.batch]
        vectors = embedder.embed([texts[cid] for cid in batch])
        client.upsert(collection, points=[
            models.PointStruct(id=cid, vector=vector, payload=payloads[cid])
            for cid, vector in zip(batch, vectors)
        ], wait=True)
        written += len(batch)
        print(f"  {written}/{len(todo)}")

    labelled = sum(1 for cid in todo
                   if any(payloads[cid].get(f) for f in PAYLOAD_ONLY_FIELDS))
    print(f"\nre-embedded {written} chunks under their existing ids "
          f"({labelled} still carry their citation labels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
