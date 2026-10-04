"""The only two pathway-aware functions in the agent.

The parent's adapter nodes call these to translate a pathway into the concrete
namespaces and style a subgraph consumes; subgraphs themselves have no idea
pathways exist.

Scoping: a student's content is a UNION of layers —
general official syllabus (always) + the turn's subject content (a teacher's, when
the subject is school/external). A subject's origin is derivable from its own fields
(``created_by`` + ``school_id``). The per-origin namespace scheme
(``general:{curriculum}:{grade}:{code}`` + ``subject:{id}``) is stamped at ingestion,
so it takes effect on the re-ingest; until then general content is under the
legacy ``syllabus`` bucket and per-turn subject scoping is done by the ``subject_id``
payload filter in ``rag_search``.
"""

from __future__ import annotations

from typing import Optional

from ..models import PersonaVoice, Style

_SYLLABUS = "syllabus"


def build_namespaces(*, session_id: Optional[str] = None) -> list[str]:
    """Vector-query namespaces for a turn. ``syllabus`` (the general base) is always
    present; an ephemeral session namespace is appended when a session upload exists.

    Post-re-ingest this becomes the union ``general:{curriculum}:{grade}:{code}`` +
    ``subject:{subject_id}`` built from the subject's facts; today general content is
    the single ``syllabus`` bucket and the subject is scoped by the ``subject_id``
    payload filter."""
    # TODO (deferred): extend the scheme beyond general+subject — add school/center and
    # teacher scoping, plus the active-membership classroom namespace (re-checked per turn,
    # invariant 2).
    namespaces = [_SYLLABUS]
    if session_id:
        namespaces.append(f"session_{session_id}")
    return namespaces


def resolve_style(pathway: str, persona: Optional[PersonaVoice] = None) -> Style:
    """The tone a pathway is answered in, or the teacher's own when their class has
    an approved persona.

    There was an ``AdaptiveProfile`` parameter here, accepted "so callers don't
    change" when the per-user profile landed. No caller ever passed it, so the
    branch behind it could not run, and the model it took was unreachable from
    anywhere in the service."""
    if persona is not None:
        return Style(tone=persona.card.tone, persona=persona)
    return Style(tone="professional" if pathway == "teacher" else "friendly")
