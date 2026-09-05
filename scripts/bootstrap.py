"""Provision local stores for ingestion testing.

    python scripts/bootstrap.py                # both, from config/.env
    python scripts/bootstrap.py --postgres     # apply dev_ai_schema.sql only
    python scripts/bootstrap.py --qdrant       # create the Qdrant collection only

- Postgres: applies scripts/dev_ai_schema.sql to ``DATABASE_URL``. Skipped when
  DATABASE_URL is unset / the placeholder.
- Qdrant: creates the ``vector_store.collection``, sized to ``embeddings.dimensions``.
  Skipped when QDRANT_URL is unset / unreachable.

Idempotent. Dev/test only — in production the backend owns the DB schema and the
pipeline auto-creates the Qdrant collection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion.config import Config, get_config  # noqa: E402
from ingestion.stores import DryRunVectorStore, _sync_url, get_vector_store  # noqa: E402

_SQL = Path(__file__).with_name("dev_ai_schema.sql")
_PLACEHOLDER_DB = "postgresql+asyncpg://user:password@localhost"


def _strip_line_comments(text: str) -> str:
    """Drop ``--`` comments (to end of line). Our SQL has no ``--`` inside a
    string or the dollar-quoted block, so a per-line cut is safe — and it must
    run before splitting, since a comment like ``schema.sql; ingestion`` contains
    a semicolon that would otherwise fake a statement break."""
    lines = []
    for line in text.splitlines():
        idx = line.find("--")
        lines.append(line if idx == -1 else line[:idx])
    return "\n".join(lines)


def _split_sql(text: str) -> list[str]:
    """Split a SQL script on ';' while respecting ``$$``-dollar-quoted blocks
    (the DO block), so the enum guard is not chopped in half."""
    text = _strip_line_comments(text)
    statements: list[str] = []
    buf: list[str] = []
    in_dollar = False
    i = 0
    while i < len(text):
        if text[i : i + 2] == "$$":
            in_dollar = not in_dollar
            buf.append("$$")
            i += 2
            continue
        ch = text[i]
        if ch == ";" and not in_dollar:
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
        else:
            buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


def bootstrap_postgres(config: Config) -> None:
    url = config.secret("DATABASE_URL")
    if not url or url.startswith(_PLACEHOLDER_DB):
        print("postgres: skipped (DATABASE_URL not set or still the placeholder)")
        return

    from sqlalchemy import create_engine, text

    engine = create_engine(_sync_url(url), future=True)
    statements = _split_sql(_SQL.read_text(encoding="utf-8"))
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
    print(f"postgres: applied {len(statements)} statements from {_SQL.name}")


def bootstrap_qdrant(config: Config) -> None:
    store = get_vector_store(config)
    if isinstance(store, DryRunVectorStore):
        print("qdrant: skipped (QDRANT_URL not set — dry-run store)")
        return
    dimension = config.get("embeddings.dimensions") or 1024
    collection = config.get("vector_store.collection", "eduai_chunks")
    try:
        store.ensure_collection(dimension)
    except Exception as exc:  # unreachable server, auth, etc.
        print(f"qdrant: could not create collection '{collection}': {exc}")
        return
    print(f"qdrant: collection '{collection}' ready (dim={dimension})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/bootstrap.py",
        description="Create the ai.* tables and the Qdrant collection for testing.",
    )
    parser.add_argument("--postgres", action="store_true", help="apply dev_ai_schema.sql")
    parser.add_argument("--qdrant", action="store_true", help="create the Qdrant collection")
    args = parser.parse_args(argv)

    if not (args.postgres or args.qdrant):
        args.postgres = args.qdrant = True  # default: both

    config = get_config()
    if args.postgres:
        bootstrap_postgres(config)
    if args.qdrant:
        bootstrap_qdrant(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
