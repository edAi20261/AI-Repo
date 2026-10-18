"""Bearer tokens for Google Cloud, minted from a service account.

Vertex is the only gateway here that will not take a static key. It wants an
OAuth access token, those expire about an hour after they are issued, and a
book takes longer than that to read — so a token fetched when a job starts is
not a token that still works when the job ends.

So the token is minted on demand and cached until shortly before it expires.
The margin is deliberate: a token that is valid when we check it and expired
when the request lands is the kind of failure that appears once in a hundred
pages and is never reproducible.

Why Vertex and not the simpler Gemini API: they bill from different places.
The Gemini API draws on AI Studio prepay credit; Vertex draws on the Cloud
billing account, which is where this project's credit actually sits. The
transport is the same OpenAI-shaped request either way — only the address and
the kind of credential differ.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

SCOPE = "https://www.googleapis.com/auth/cloud-platform"
DEFAULT_CREDENTIALS_ENV = "GOOGLE_APPLICATION_CREDENTIALS"
ATTACHED = "attached"
CLOUD_RUN_JOB_ENV = "CLOUD_RUN_JOB"

# Refresh this long before the token actually expires. Google issues them for
# an hour; five minutes is enough for a slow page read to finish on the old one.
REFRESH_MARGIN_SECONDS = 300

_LOCK = threading.Lock()
_CREDENTIALS: Any = None
_SOURCE: Optional[str] = None


class CredentialsUnavailable(RuntimeError):
    """No service account could be loaded, so no Google call can be signed."""


def credential_source(config: Any) -> Optional[str]:
    configured = config.secret(config.get("vision.credentials_env") or DEFAULT_CREDENTIALS_ENV)
    if configured:
        return configured
    return ATTACHED if os.environ.get(CLOUD_RUN_JOB_ENV) else None


def _attached() -> tuple[Any, Optional[str]]:
    try:
        import google.auth
    except ImportError as error:  # pragma: no cover - import guard
        raise CredentialsUnavailable(
            "google-auth is not installed; it is needed to reach Vertex.") from error
    try:
        return google.auth.default(scopes=[SCOPE])
    except Exception as error:
        raise CredentialsUnavailable(
            "no service account is attached to this job") from error


def _load(source: str) -> Any:
    """Credentials from a path on disk, or from the JSON itself.

    Both, because the two places this runs hand it over differently. A laptop
    has a file. ECS injects a secret as an environment variable, so there is no
    file to point at — and writing one to disk at startup just to read it back
    puts a private key on a filesystem for no reason.
    """
    try:
        from google.oauth2 import service_account  # lazy: only Vertex needs it
    except ImportError as error:  # pragma: no cover - import guard
        raise CredentialsUnavailable(
            "google-auth is not installed; it is needed to reach Vertex.") from error

    if source == ATTACHED:
        return _attached()[0]
    text = source.strip()
    if text.startswith("{"):
        import json

        try:
            return service_account.Credentials.from_service_account_info(
                json.loads(text), scopes=[SCOPE])
        except Exception as error:
            raise CredentialsUnavailable(
                "the service account JSON could not be read") from error
    try:
        return service_account.Credentials.from_service_account_file(text, scopes=[SCOPE])
    except Exception as error:
        raise CredentialsUnavailable(
            f"the service account at {text} could not be read") from error


def project_of(source: str) -> Optional[str]:
    """The project a service account belongs to, from a path or from the JSON."""
    import json

    if source == ATTACHED:
        return _attached()[1]
    text = source.strip()
    try:
        if text.startswith("{"):
            return json.loads(text).get("project_id")
        with open(text, encoding="utf-8") as handle:
            return json.load(handle).get("project_id")
    except Exception:
        logger.warning("could not read the project id from the service account",
                       exc_info=True)
        return None


def access_token(credentials_path: str) -> str:
    """A valid bearer token for ``credentials_path``, minted or reused.

    Reused while it has more than ``REFRESH_MARGIN_SECONDS`` left, so a book of
    three hundred pages pays for one token rather than three hundred.
    """
    global _CREDENTIALS, _SOURCE

    with _LOCK:
        if _CREDENTIALS is None or _SOURCE != credentials_path:
            _CREDENTIALS = _load(credentials_path)
            _SOURCE = credentials_path
            logger.info("google service account loaded",
                        extra={"client_email": getattr(_CREDENTIALS, "service_account_email",
                                                       None)})
        if _needs_refresh(_CREDENTIALS):
            import google.auth.transport.requests as transport  # lazy

            _CREDENTIALS.refresh(transport.Request())
            logger.info("google access token refreshed",
                        extra={"expires_at": str(getattr(_CREDENTIALS, "expiry", None))})
        token = getattr(_CREDENTIALS, "token", None)
    if not token:
        raise CredentialsUnavailable("the service account produced no access token")
    return token


def _needs_refresh(credentials: Any) -> bool:
    if not getattr(credentials, "token", None):
        return True
    expiry = getattr(credentials, "expiry", None)
    if expiry is None:
        return True
    # google-auth stores expiry as a naive UTC datetime.
    from datetime import datetime, timezone

    moment = expiry.replace(tzinfo=timezone.utc) if expiry.tzinfo is None else expiry
    return (moment - datetime.now(timezone.utc)).total_seconds() <= REFRESH_MARGIN_SECONDS


def reset() -> None:
    """Forget the cached credentials. For tests, and for a rotated key."""
    global _CREDENTIALS, _SOURCE
    with _LOCK:
        _CREDENTIALS, _SOURCE = None, None


BILLED_PROJECT_ENV = "VERTEX_PROJECT"


def billed_project(config: Any, source: Optional[str]) -> Optional[str]:
    """The project a Vertex call is made in, which is the project it is billed to.

    Usually the credential's own. ``VERTEX_PROJECT`` moves the bill to another
    project the credential may call, without a rebuild: ingestion spends one
    project's free credit, then is pointed at the next."""
    return (config.secret(BILLED_PROJECT_ENV) or config.get("vision.project")
            or (project_of(source) if source else None))


def vertex_base_url(project: str, location: str) -> str:
    """Vertex's OpenAI-compatible endpoint for a project and region.

    ``global`` has no regional prefix; every other location does.
    """
    host = ("aiplatform.googleapis.com" if location == "global"
            else f"{location}-aiplatform.googleapis.com")
    return f"https://{host}/v1/projects/{project}/locations/{location}/endpoints/openapi"
