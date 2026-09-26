"""Reading a conversation back out of the platform's own transcript.

What this service remembers of a conversation expires; what the student can see
of it does not. ``ttl_hours`` is 720, so after thirty days the turns are gone
from ``ai.user_memory`` while every word is still in
``engagement.ai_chat_messages``, where the app reads them to draw the thread.

Nobody could reach an old thread before, so nobody met the result. A history
panel is exactly a feature for reaching old threads, and without this the
student reopens a conversation, sees the whole of it on screen, asks "so why
does that happen?" — and is answered by a tutor for whom this is the first
message ever sent.

So when a conversation has nothing remembered and the platform still holds a
transcript, it is read back. What comes back is the words and which lesson the
answer came from; the passages it rested on are not in the transcript, so a
recovered turn can be followed up by what was said in it, not by the chunks
behind it. That is a smaller memory than the one that expired, and a much
larger one than none.

The transcript belongs to the platform. This reads it and never writes it.
"""

from __future__ import annotations

import logging
from typing import Optional

from shared.postgres import connect

from .records import SessionState, Turn

logger = logging.getLogger(__name__)

STUDENT = "student"
TUTOR = "ai"

_MESSAGES = (
    "SELECT sender, message_text, metadata, created_at "
    "FROM engagement.ai_chat_messages WHERE session_id = %s::uuid "
    "ORDER BY created_at DESC, id DESC LIMIT %s"
)

_SUBJECT = "SELECT subject_id::text FROM engagement.ai_chat_sessions WHERE id = %s::uuid"


def recover(session_id: Optional[str], *, turns: int,
            config=None) -> tuple[list[Turn], SessionState]:
    """The last ``turns`` exchanges of this thread, oldest first, and its subject.

    Empty when there is no transcript, when it cannot be read, or when the
    thread holds no completed exchange — all three of which mean the same thing
    to the caller, which is that there is nothing to carry.
    """
    if not session_id:
        return [], SessionState()
    try:
        with connect(config) as conn, conn.cursor() as cur:
            # Two messages to an exchange, and a couple spare for a question
            # the tutor never answered.
            cur.execute(_MESSAGES, (session_id, turns * 2 + 2))
            rows = list(reversed(cur.fetchall()))
            cur.execute(_SUBJECT, (session_id,))
            subject = cur.fetchone()
    except Exception:
        logger.warning("the platform transcript could not be read; the conversation "
                       "starts from nothing", exc_info=True,
                       extra={"session_id": session_id})
        return [], SessionState()

    recovered = _exchanges(rows)[-turns:]
    if recovered:
        logger.info("conversation recovered from the platform transcript",
                    extra={"session_id": session_id, "turns": len(recovered)})
    return recovered, SessionState(subject_id=(subject or [None])[0])


def _exchanges(rows) -> list[Turn]:
    """Messages paired into turns: a question and the answer that followed it.

    A question with no answer after it is dropped, the same way a turn that
    produced no reply is never written in the first place — so what comes back
    is what would have been remembered, not everything that was typed.
    """
    turns: list[Turn] = []
    asked: Optional[tuple] = None
    for sender, text, metadata, when in rows:
        if sender == STUDENT:
            asked = (text or "", when)
        elif sender == TUTOR and asked is not None:
            turns.append(Turn(
                message=asked[0],
                reply=text or "",
                kind="study" if (metadata or {}).get("tasks") else "conversation",
                tasks=list((metadata or {}).get("tasks") or []),
                # The transcript keeps which lesson was cited, not which
                # passages. A recovered turn can be followed up by what it
                # said; it cannot lend its evidence to the next answer.
                cited_chunk_ids=[],
            ))
            asked = None
    return turns
