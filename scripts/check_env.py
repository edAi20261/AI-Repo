"""Preflight: prove every configured credential actually works.

Each check makes one real call and reports what the service said. Secrets are
never printed — only whether they are present and whether they were accepted.

    python scripts/check_env.py

Exit code is the number of blocking failures, so it is usable in CI.
"""

from __future__ import annotations

import os
import sys
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import Config

OK, FAIL, SKIP, WARN = "PASS", "FAIL", "SKIP", "WARN"


class Report:
    """Collects results; blocking failures set the exit code."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []
        self.blocking = 0

    def add(self, name: str, state: str, detail: str, *, blocking: bool = False) -> None:
        self.rows.append((name, state, detail))
        if state == FAIL and blocking:
            self.blocking += 1

    def render(self) -> None:
        width = max(len(n) for n, _, _ in self.rows)
        print()
        for name, state, detail in self.rows:
            print(f"  [{state}] {name:<{width}}  {detail}")
        print()
        if self.blocking:
            print(f"{self.blocking} blocking failure(s) — the service will not work yet.")
        else:
            print("No blocking failures.")


def check_backend(report: Report) -> None:
    base = Config.secret("BACKEND_API_BASE_URL")
    token = Config.secret("AI_WEBHOOK_SHARED_SECRET")
    if not base:
        report.add("platform backend", SKIP, "BACKEND_API_BASE_URL unset")
        return
    if not token:
        report.add("platform backend", FAIL, "AI_WEBHOOK_SHARED_SECRET unset", blocking=True)
        return

    import httpx

    # A known-absent subject: 404 proves the token passed the guard, 401 proves it did not.
    url = f"{base.rstrip('/')}/subjects/11111111-1111-4111-8111-111111111111"
    try:
        response = httpx.get(url, headers={"x-service-token": token}, timeout=30)
    except Exception as exc:
        report.add("platform backend", FAIL, f"unreachable: {type(exc).__name__}", blocking=True)
        return

    if response.status_code == 401:
        report.add("platform backend", FAIL, "token rejected (401)", blocking=True)
    elif response.status_code in (200, 400, 404):
        report.add("platform backend", OK, f"token accepted (HTTP {response.status_code})")
    else:
        report.add("platform backend", WARN, f"unexpected HTTP {response.status_code}")


def check_openrouter(report: Report) -> None:
    key = Config.secret("OPENROUTER_API_KEY")
    if not key:
        report.add(
            "openrouter (llm)", FAIL,
            "unset — the service runs but answers with stub text", blocking=True,
        )
        return

    import httpx

    try:
        response = httpx.get(
            "https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {key}"},
            timeout=30,
        )
    except Exception as exc:
        report.add("openrouter (llm)", FAIL, f"unreachable: {type(exc).__name__}", blocking=True)
        return

    if response.status_code == 200:
        data = (response.json() or {}).get("data", {})
        limit, usage = data.get("limit"), data.get("usage")
        budget = "unlimited" if limit is None else f"limit {limit}"
        report.add("openrouter (llm)", OK, f"key valid — usage {usage}, {budget}")
    else:
        report.add(
            "openrouter (llm)", FAIL,
            f"key rejected (HTTP {response.status_code})", blocking=True,
        )


def check_database(report: Report) -> None:
    url = Config.secret("DATABASE_URL")
    if not url:
        report.add("postgres", FAIL, "DATABASE_URL unset", blocking=True)
        return

    parsed = urlparse(url)
    if not parsed.hostname:
        report.add(
            "postgres", FAIL,
            "malformed — expected postgresql+asyncpg://user:pass@host:5432/db",
            blocking=True,
        )
        return

    try:
        import psycopg
    except ImportError:
        report.add("postgres", SKIP, "psycopg not installed")
        return

    dsn = url.replace("+asyncpg", "").replace("+psycopg", "")
    try:
        with psycopg.connect(dsn, connect_timeout=8) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'ai'"
            )
            tables = cur.fetchone()[0]
    except Exception as exc:
        report.add(
            "postgres", FAIL,
            f"{parsed.hostname}: {type(exc).__name__} — {str(exc).splitlines()[0][:90]}",
            blocking=True,
        )
        return

    if tables:
        report.add("postgres", OK, f"connected — {tables} ai.* tables present")
    else:
        report.add(
            "postgres", WARN,
            "connected but no ai.* tables — run scripts/bootstrap.py",
        )


def check_qdrant(report: Report) -> None:
    url = Config.secret("QDRANT_URL")
    if not url:
        report.add("qdrant (vectors)", FAIL, "QDRANT_URL unset", blocking=True)
        return

    try:
        from qdrant_client import QdrantClient
    except ImportError:
        report.add("qdrant (vectors)", SKIP, "qdrant-client not installed")
        return

    try:
        client = QdrantClient(url=url, api_key=Config.secret("QDRANT_API_KEY"), timeout=15)
        names = [c.name for c in client.get_collections().collections]
    except Exception as exc:
        report.add(
            "qdrant (vectors)", FAIL,
            f"{type(exc).__name__} — {str(exc).splitlines()[0][:90]}", blocking=True,
        )
        return

    report.add("qdrant (vectors)", OK, f"connected — collections: {names or 'none yet'}")


def check_s3(report: Report) -> None:
    bucket = Config.secret("S3_BUCKET")
    if not bucket:
        report.add("s3 (figures, visuals)", SKIP, "S3_BUCKET unset — visuals fall back to disk")
        return

    key_id = Config.secret("AWS_ACCESS_KEY_ID") or ""
    secret = Config.secret("AWS_SECRET_ACCESS_KEY") or ""
    if secret and len(secret) != 40:
        report.add(
            "s3 (figures, visuals)", FAIL,
            f"AWS_SECRET_ACCESS_KEY is {len(secret)} chars; AWS keys are 40 — looks truncated",
        )
        return
    if key_id and len(key_id) != 20:
        report.add(
            "s3 (figures, visuals)", WARN,
            f"AWS_ACCESS_KEY_ID is {len(key_id)} chars; AWS keys are 20",
        )

    try:
        import boto3
    except ImportError:
        report.add("s3 (figures, visuals)", SKIP, "boto3 not installed")
        return

    try:
        boto3.client(
            "s3",
            region_name=Config.secret("S3_REGION"),
            endpoint_url=Config.secret("S3_ENDPOINT_URL") or None,
        ).head_bucket(Bucket=bucket)
    except Exception as exc:
        report.add("s3 (figures, visuals)", FAIL, f"{type(exc).__name__} — {str(exc)[:90]}")
        return

    report.add("s3 (figures, visuals)", OK, f"bucket '{bucket}' reachable, read/write assumed")


def main() -> int:
    Config.load()  # loads .env
    print(f"APP_ENV = {os.environ.get('APP_ENV', 'local')}")

    report = Report()
    check_backend(report)
    check_openrouter(report)
    check_database(report)
    check_qdrant(report)
    check_s3(report)
    report.render()
    return report.blocking


if __name__ == "__main__":
    raise SystemExit(main())
