"""ASGI entrypoint: ``uvicorn system.asgi:app``."""

from .main import create_app

app = create_app()
