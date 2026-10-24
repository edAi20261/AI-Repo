# ---------------------------------------------------------------------------
# EduAI — system/ HTTP service image.
#
# Builds the AI service that the platform backend calls (system/asgi:app).
# Ingestion runs from the same image via `docker run ... python -m ingestion ...`
# if needed, since agent/, ingestion/ and shared/ all ship in the image.
#
# Multi-stage: packages install into a venv in the builder stage; only the
# venv + app code cross into the runtime stage, so pip's own footprint and
# any install-time state never end up in the final image.

# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PIP_NO_CACHE_DIR=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt .
# torch's default PyPI wheel bundles CUDA (multi-GB, unreliable to pull); this
# service only runs the embedder on CPU, so pull the much smaller CPU wheel
# first from PyTorch's own index, then satisfy the rest from PyPI as normal.
# --no-compile: skip .pyc generation, they're regenerated (and cached) on
# first import at runtime anyway.
RUN pip install --no-compile --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 \
    && pip install --no-compile -r requirements.txt

# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY agent/ agent/
COPY shared/ shared/
COPY ingestion/ ingestion/
COPY system/ system/
# 216 kB of Python and JSON, against a 2 GB embedding model — the rule that
# keeps this image small is about what a student waits for on a cold start, and
# this is not that. It is here because the load balancer is internal: the evals
# can only reach the service from inside the VPC, and running them from an image
# CI already builds is what stops the thing being measured and the thing
# deployed from drifting apart.
COPY scripts/ scripts/
COPY config.yaml .

# Non-root runtime user.
RUN useradd --create-home --uid 1000 appuser
USER appuser

# Pre-warm matplotlib's font cache under the runtime user so the first
# request isn't the one paying for it.
RUN python -c "import matplotlib.pyplot" || true

RUN python -c "from shared.config import get_config; from shared.embeddings import get_embedder; get_embedder(get_config())"
ENV HF_HUB_OFFLINE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "system.asgi:app", "--host", "0.0.0.0", "--port", "8000"]
