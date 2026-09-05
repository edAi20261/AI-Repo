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

from ..models import AdaptiveProfile, Style

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


def resolve_style(
    pathway: str,
    *,
    profile: Optional[AdaptiveProfile] = None,
) -> Style:
    """Voice/style descriptor for a pathway. Neutral defaults now; the adaptive_profile source
    wires in later (the parameter is accepted now so callers don't change)."""
    # TODO (deferred): source `profile` from a per-user adaptive_profile (student/teacher/parent).
    # When answering a teacher's own subject, use THAT teacher's profile (their teaching style),
    # not only the current user's.
    # The pathway sets the default tone; a profile (any pathway) overrides it -> adaptive.
    if profile and profile.style:
        return Style(tone=profile.style.tone, voice="adaptive")
    default_tone = "professional" if pathway == "teacher" else "friendly"
    return Style(tone=default_tone, voice="neutral")
