"""Verify ingested lessons in the live stores against the pipeline invariants.

    python scripts/verify_ingest.py [--lecture-id <uuid>]

Per lecture (all lectures in ai.chunks when no id is given) it checks:
  - chunk / manifest / Qdrant point parity (ai.chunks == ai.vector_index == points)
  - a topic_id on every chunk (invariant 12)
  - the vision stub sentinel absent from chunk text
  - chunk text is NFKC-normalised
  - namespace / subject_id / grade / language on every Qdrant point
  - re-ingestion duplication (chunks beyond one per order_index+version)
"""

from __future__ import annotations

import argparse
import os
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.config import Config
from ingestion.llm_provider import VISION_STUB_SENTINEL


def _pg_url(raw: str) -> str:
    return raw.replace("+asyncpg", "").replace("+psycopg", "")


def _lectures(cur, lecture_id: str | None) -> list[str]:
    if lecture_id:
        return [lecture_id]
    cur.execute("SELECT DISTINCT lecture_id::text FROM ai.chunks ORDER BY 1")
    return [r[0] for r in cur.fetchall()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lecture-id", default=None)
    args = ap.parse_args()

    cfg = Config.load()
    import psycopg
    from qdrant_client import QdrantClient, models

    collection = cfg.vector_store.get("collection", "eduai_chunks")
    qc = QdrantClient(url=cfg.secret("QDRANT_URL", "http://localhost:6333"),
                      api_key=cfg.secret("QDRANT_API_KEY"))
    conn = psycopg.connect(_pg_url(cfg.secret("DATABASE_URL")))
    cur = conn.cursor()

    print(f"collection={collection}  app_env={cfg.app_env}\n")
    all_ok = True
    for lid in _lectures(cur, args.lecture_id):
        cur.execute("SELECT count(*), count(DISTINCT (order_index, version)) "
                    "FROM ai.chunks WHERE lecture_id = %s", (lid,))
        n_chunks, n_distinct = cur.fetchone()
        cur.execute("SELECT count(*) FROM ai.chunks WHERE lecture_id = %s "
                    "AND (topic_ids IS NULL OR cardinality(topic_ids) = 0)", (lid,))
        no_topic = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM ai.chunks WHERE lecture_id = %s "
                    "AND text LIKE %s", (lid, f"%{VISION_STUB_SENTINEL}%"))
        stubbed = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM ai.vector_index WHERE lecture_id = %s", (lid,))
        n_manifest = cur.fetchone()[0]

        cur.execute("SELECT text FROM ai.chunks WHERE lecture_id = %s", (lid,))
        not_nfkc = sum(1 for (t,) in cur.fetchall()
                       if unicodedata.normalize("NFKC", t) != t)

        flt = models.Filter(must=[models.FieldCondition(
            key="lecture_id", match=models.MatchValue(value=lid))])
        n_points = qc.count(collection, count_filter=flt, exact=True).count
        pts, _ = qc.scroll(collection, scroll_filter=flt, limit=10000, with_payload=True)
        need = ("namespace", "subject_id", "grade", "language")
        bad_payload = sum(1 for p in pts if any(k not in p.payload for k in need))

        cur.execute("SELECT status, version, chunk_count, topic_count, ingested_at "
                    "FROM ai.lecture_ingestion WHERE lecture_id = %s", (lid,))
        rec = cur.fetchone()

        dup = n_chunks - n_distinct
        checks = {
            "parity chunks==manifest==points": n_chunks == n_manifest == n_points,
            "topic_id on every chunk": no_topic == 0,
            "no stub sentinel": stubbed == 0,
            "NFKC-clean text": not_nfkc == 0,
            "payload complete on points": bad_payload == 0,
            "no re-ingest duplicates": dup == 0,
        }
        ok = all(checks.values())
        all_ok = all_ok and ok
        print(f"{'OK ' if ok else 'FAIL'}  {lid}")
        if rec:
            print(f"      record: status={rec[0]} version={rec[1]} "
                  f"chunk_count={rec[2]} topic_count={rec[3]} ingested_at={rec[4]:%Y-%m-%d %H:%M}")
        else:
            print("      record: (none in ai.lecture_ingestion)")
        print(f"      chunks={n_chunks} manifest={n_manifest} points={n_points} "
              f"distinct={n_distinct} dup={dup} no_topic={no_topic} "
              f"stubbed={stubbed} not_nfkc={not_nfkc} bad_payload={bad_payload}")
        for name, passed in checks.items():
            if not passed:
                print(f"        - FAIL: {name}")

    conn.close()
    print("\n" + ("ALL LECTURES OK" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
