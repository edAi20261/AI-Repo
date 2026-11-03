"""Assessment entrypoint (endpoint path).

The backend's ``POST /ai/assessments/generate`` calls ``run_assessment`` with a
structured request: lesson scope, count, types and difficulty. It runs the same
assessment_sg the chat's quiz turn runs, from the same defaults, without a turn to
plan it. Nothing is saved: the backend stores the quiz it is handed.
"""

from __future__ import annotations

from typing import Any, Optional

from shared.backend import get_backend_client
from shared.language import LanguagePolicy
from shared.llm import maybe_llm

from ..config import Config, get_config
from ..models import AssessmentSpec
from .graph import _subject_facts, assessment_inputs
from .nodes import _assessment_type_mix
from .pathways import build_namespaces
from .subgraphs import build_assessment_sg


def run_assessment(payload: dict[str, Any], *, config: Optional[Config] = None) -> dict:
    """Generate ``payload['count']`` questions grounded in the scoped lessons.

    Payload: ``user_id``, ``count``, ``subject_id?``, ``lecture_ids?``, ``topic_ids?``,
    ``focus?`` (what the questions should cover, in the student's words), ``types?``,
    ``difficulty?``, ``language?``. Returns assessment_sg's ``assessment``: ``items``,
    ``count``, ``requested_type_mix``, ``complete``, ``shortfall``, ``degraded``,
    ``language``, ``intro``.
    """
    config = config or get_config()
    try:
        client = get_backend_client(config)
    except ValueError:
        client = None
    facts = _subject_facts(client, payload.get("subject_id"))
    turn = LanguagePolicy.from_config(config).reply(
        subject_language=facts.language, api_language=payload.get("language"))
    spec = AssessmentSpec(count=payload["count"], types=payload.get("types"))
    overrides = {
        "query": payload.get("focus"),
        "topic_ids": payload.get("topic_ids"),
        "type_mix": _assessment_type_mix(spec, config.get("assessment.type_mix")),
        "difficulty": payload.get("difficulty"),
    }

    sg = build_assessment_sg(config=config, llm=maybe_llm(config, "assessment"))
    result = sg.invoke(assessment_inputs(
        config, facts, turn_language=turn.code, overrides=overrides,
        lecture_ids=payload.get("lecture_ids") or None,
        subject_id=payload.get("subject_id"),
        namespaces=build_namespaces(),
        grade=None,
        user_id=payload.get("user_id"),
    ))
    return result["assessment"]
