from __future__ import annotations

import pytest

from shared import gcp
from shared.config import Config


@pytest.fixture
def config(monkeypatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv(gcp.CLOUD_RUN_JOB_ENV, raising=False)
    gcp.reset()
    yield Config.load()
    gcp.reset()


def test_a_configured_key_is_used_wherever_the_code_runs(config, monkeypatch):
    """A key given explicitly must keep winning, or moving a job to Cloud Run
    would silently switch which project pays for its model calls."""
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/keys/sa.json")
    monkeypatch.setenv(gcp.CLOUD_RUN_JOB_ENV, "ingest-read")
    assert gcp.credential_source(config) == "/keys/sa.json"


def test_without_a_key_a_cloud_run_job_uses_its_attached_service_account(config, monkeypatch):
    """The ingestion project forbids service account keys, so a Cloud Run job
    has no key to read: its own attached account is the only credential."""
    monkeypatch.setenv(gcp.CLOUD_RUN_JOB_ENV, "ingest-read")
    assert gcp.credential_source(config) == gcp.ATTACHED


def test_without_a_key_anywhere_else_there_is_no_credential(config):
    """On a laptop or in CI no ambient Google login may be picked up, or a test
    run could start calling and billing a real project."""
    assert gcp.credential_source(config) is None


def test_vertex_counts_as_available_for_a_cloud_run_job(config, monkeypatch):
    """Availability decides between the real model and the stub; a job with an
    attached account reported none and would have read every page with a stub."""
    from shared.llm import llm_key_available

    spec = {"provider": "vertex", "model": "google/gemini-2.5-flash"}
    assert not llm_key_available(config, spec)
    monkeypatch.setenv(gcp.CLOUD_RUN_JOB_ENV, "ingest-finish")
    assert llm_key_available(config, spec)


def test_the_attached_account_supplies_both_the_token_and_the_project(config, monkeypatch):
    """Vertex's address needs the project as well as the token, and with no key
    file there is no JSON to read the project from."""
    class _Credentials:
        token = None
        expiry = None
        service_account_email = "ingestion-deployer@example.iam.gserviceaccount.com"

        def refresh(self, _request):
            self.token = "attached-token"

    monkeypatch.setattr(gcp, "_attached", lambda: (_Credentials(), "ingest-project"))
    assert gcp.access_token(gcp.ATTACHED) == "attached-token"
    assert gcp.project_of(gcp.ATTACHED) == "ingest-project"


def test_a_cloud_run_job_reads_pages_with_the_model_not_the_stub(config, monkeypatch):
    """The first book run on Cloud Run stopped on every reader: the vision
    client asked for a key file itself, found none, and fell back to the stub."""
    from ingestion.llm_provider import get_vision_client

    monkeypatch.setenv(gcp.CLOUD_RUN_JOB_ENV, "ingest-read")
    assert not get_vision_client(config).is_stub
