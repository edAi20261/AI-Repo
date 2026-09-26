"""What the tutor still knows about a student when the chat thread is new.

Every fact the tutor held was scoped to one ``source_conversation_id``. A
student who closed the tab and opened another chat met a stranger: same
student, same subject, same lesson they had been stuck on an hour earlier, and
nothing carried. For a platform where a student comes back tomorrow, that is
the difference between a tutor and a search box with manners.

Two kinds of fact live here and they are kept differently.

*Derived* facts — which languages this student actually works in, which
subjects they have been in lately — are computed from what is already stored
each time they are wanted. They are never written down, because a fact that is
recomputed cannot go stale, cannot disagree with its source, and cannot be
wrong in a row nobody thought to look at.

*Stated* preferences are the ones the student said out loud, and those have to
be stored, because nothing else records them. They go in ``ai.adaptive_profile``
— one row per student, the shape the platform already declared — and only ever
from :class:`~agent.models.StatedPreference`, which is a closed set of fields.
Nothing here writes a judgement about a student. "Prefers short answers"
because they said so is a fact; because a model decided it from three short
questions is a decision made about someone who cannot see it and cannot undo
it.

Read on the first turn of a conversation and nowhere else: it is two queries,
and a conversation that already has turns has better evidence in front of it.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from shared.postgres import connect

from .conversation.store import FOLDED, TURN
from .models import StatedPreference

logger = logging.getLogger(__name__)

# How far back "lately" reaches when deriving what a student has been doing.
# Their own last few conversations, not their term.
RECENT_TURNS = 40
MAX_SUBJECTS = 3
MAX_LANGUAGES = 3

NOTHING_CARRIED = ""

_DERIVED = (
    "SELECT fact FROM ai.user_memory "
    "WHERE user_id = %s::uuid AND kind = ANY(%s) "
    "  AND (expires_at IS NULL OR expires_at > now()) "
    "ORDER BY created_at DESC LIMIT %s"
)

_STATED = (
    "SELECT explanation_length, prefers_examples, prefers_visual, pace "
    "FROM ai.adaptive_profile WHERE user_id = %s::uuid"
)

_RECORD = (
    "INSERT INTO ai.adaptive_profile AS p "
    "  (user_id, explanation_length, prefers_examples, prefers_visual, pace, updated_at) "
    "VALUES (%s::uuid, %s, COALESCE(%s, false), COALESCE(%s, false), %s, now()) "
    "ON CONFLICT (user_id) DO UPDATE SET "
    "  explanation_length = COALESCE(EXCLUDED.explanation_length, p.explanation_length), "
    "  prefers_examples = COALESCE(%s, p.prefers_examples), "
    "  prefers_visual = COALESCE(%s, p.prefers_visual), "
    "  pace = COALESCE(EXCLUDED.pace, p.pace), "
    "  updated_at = now()"
)


@dataclass(frozen=True)
class Carryover:
    """What a new conversation starts already knowing about this student."""

    languages: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    stated: StatedPreference = field(default_factory=StatedPreference)

    def __bool__(self) -> bool:
        return bool(self.languages or self.subjects or self.stated)

    def render(self) -> str:
        """The carryover as the head of a conversation that has no turns yet."""
        if not self:
            return NOTHING_CARRIED
        lines = ["What you already know about this student, from before this conversation:"]
        if self.subjects:
            lines.append(f"- They have been working on: {', '.join(self.subjects)}.")
        if self.languages:
            lines.append(f"- They write to you in: {', '.join(self.languages)}.")
        lines.extend(f"- {line}" for line in _stated_lines(self.stated))
        lines.append("Do not greet them as a stranger and do not recap this back at them; "
                     "it is context, not something they asked about.")
        return "\n".join(lines)

    def log_fields(self) -> dict:
        return {"carried_subjects": len(self.subjects),
                "carried_languages": len(self.languages),
                "carried_preferences": bool(self.stated)}


def _stated_lines(stated: StatedPreference) -> list[str]:
    lines = []
    if stated.length:
        lines.append(f"They asked for {stated.length} answers.")
    if stated.examples is not None:
        lines.append("They asked for worked examples." if stated.examples
                     else "They asked not to be given worked examples.")
    if stated.visuals is not None:
        lines.append("They asked for diagrams and pictures." if stated.visuals
                     else "They asked not to be given diagrams.")
    if stated.pace:
        lines.append(f"They asked to be taken at a {stated.pace} pace.")
    return lines


def carried_over(user_id: Optional[str], subjects=None, *, config=None) -> Carryover:
    """What this student was doing before, or an empty carryover.

    A failure here is a conversation that starts cold, which is what every
    conversation did until now — so it degrades to the old behaviour rather
    than to an error, and says which it was.
    """
    if not user_id:
        return Carryover()
    try:
        with connect(config) as conn, conn.cursor() as cur:
            cur.execute(_DERIVED, (user_id, list(_TURN_KINDS), RECENT_TURNS))
            facts = [row[0] for row in cur.fetchall()]
            cur.execute(_STATED, (user_id,))
            row = cur.fetchone()
    except Exception:
        logger.warning("what this student did before could not be read; "
                       "starting the conversation cold", exc_info=True)
        return Carryover()

    languages, subject_ids = _from_turns(facts)
    return Carryover(
        languages=languages,
        subjects=_named(subject_ids, subjects),
        stated=_stated_from_row(row),
    )


def record(user_id: Optional[str], preference: StatedPreference, *, config=None) -> bool:
    """Keep a preference the student stated. Returns whether it was kept."""
    if not (user_id and preference):
        return False
    _warn_about_unstorable(preference)
    try:
        with connect(config) as conn:
            with conn.cursor() as cur:
                cur.execute(_RECORD, (user_id, preference.length, preference.examples,
                                      preference.visuals, preference.pace,
                                      preference.examples, preference.visuals))
            conn.commit()
    except Exception:
        logger.warning("a preference the student stated could not be kept",
                       exc_info=True, extra={"preference": preference.model_dump()})
        return False
    logger.info("student preference kept", extra=preference.model_dump(exclude_none=True))
    return True


_TURN_KINDS = (TURN, FOLDED)


def _from_turns(facts: list[str]) -> tuple[list[str], list[str]]:
    """The languages and subjects behind a student's recent turns, commonest first."""
    languages: Counter[str] = Counter()
    subjects: Counter[str] = Counter()
    for fact in facts:
        try:
            turn = json.loads(fact)
        except (TypeError, ValueError):
            continue
        if turn.get("language"):
            languages[turn["language"]] += 1
        if turn.get("subject_id"):
            subjects[turn["subject_id"]] += 1
    return ([code for code, _ in languages.most_common(MAX_LANGUAGES)],
            [sid for sid, _ in subjects.most_common(MAX_SUBJECTS)])


def _named(subject_ids: list[str], subjects) -> list[str]:
    """Subjects the tutor can name. An id it cannot name is left out."""
    if not (subject_ids and subjects):
        return []
    named = []
    for subject_id in subject_ids:
        title = (subjects.facts(subject_id).title or "").strip()
        if title:
            named.append(title)
    return named


def _warn_about_unstorable(preference: StatedPreference) -> None:
    """``prefers_examples`` and ``prefers_visual`` are NOT NULL DEFAULT false, so
    false in the column means "never said" and a student who asked *not* to be
    given examples cannot be told apart from one who never mentioned them.

    The write still happens — false is the right value, it simply cannot be read
    back as a statement. Logged rather than dropped, because the fix is one
    migration and nobody will make it if nothing ever says it is needed.
    """
    refused = [name for name in ("examples", "visuals")
               if getattr(preference, name) is False]
    if refused:
        logger.warning(
            "a student asked not to be given %s; the column cannot hold a "
            "stated 'no', so it will read back as never mentioned",
            " and ".join(refused))


def _stated_from_row(row) -> StatedPreference:
    if not row:
        return StatedPreference()
    length, examples, visuals, pace = row
    return StatedPreference(
        length=length or None,
        # false is "never said" here, not "said no": see _warn_about_unstorable.
        examples=True if examples else None,
        visuals=True if visuals else None,
        pace=pace or None,
    )
