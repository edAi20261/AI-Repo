"""Shared library used by both deployable halves (ingestion, agent) and system.

Houses only cross-cutting concerns that MUST behave identically on both sides:
configuration loading, the embedding model, and logging setup.
"""
