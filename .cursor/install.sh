#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for the EduAI AI-Workflows repo.
# Recreates the Python virtualenv and refreshes dependencies against the
# checked-out source. Safe to re-run.
set -euo pipefail

cd "$(dirname "$0")/.."

# The virtualenv needs ensurepip, which the base image's python3 may lack.
# Install python3-venv only when it is missing so the script stays fast on reruns.
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-venv
fi

python3 -m venv .venv
.venv/bin/pip install --upgrade pip

# torch's default PyPI wheel bundles CUDA (multi-GB, unreliable to pull). This
# project only runs the embedder on CPU, so pull the smaller CPU wheel first
# from PyTorch's own index (mirrors the approach in the service Dockerfile),
# then satisfy the rest from PyPI.
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
.venv/bin/pip install -r requirements-dev.txt
