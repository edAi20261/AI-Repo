"""The conversation as a model reads it: recent turns verbatim, the rest summarised.

A window of whole turns, newest first, within a character budget, so a long
answer cannot push the question it answered out of view. Turns that fall out of
the window are folded into one running summary written by a model, which keeps
goals, facts and open questions rather than a transcript.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from langchain_core.messages import HumanMessage

from ..prompts import CONVERSATION_SUMMARY_PROMPT
from .records import Conversation, Turn
from .store import ConversationStore

logger = logging.getLogger(__name__)

DEFAULT_RECENT_TURNS = 6
DEFAULT_FOLD_AFTER = 12
DEFAULT_HISTORY_CHARS = 6000
DEFAULT_REPLY_CHARS = 1500

NO_HISTORY = "(this is the first message of the conversation)"


@dataclass(frozen=True)
class HistoryWindow:
    """How much of a conversation a prompt carries."""

    recent_turns: int = DEFAULT_RECENT_TURNS
    fold_after: int = DEFAULT_FOLD_AFTER
    history_chars: int = DEFAULT_HISTORY_CHARS
    reply_chars: int = DEFAULT_REPLY_CHARS

    @classmethod
    def from_config(cls, config) -> HistoryWindow:
        settings = config.get("agent.conversation") or {}
        return cls(
            recent_turns=settings.get("recent_turns", DEFAULT_RECENT_TURNS),
            fold_after=settings.get("fold_after", DEFAULT_FOLD_AFTER),
            history_chars=settings.get("history_chars", DEFAULT_HISTORY_CHARS),
            reply_chars=settings.get("reply_chars", DEFAULT_REPLY_CHARS),
        )

    def render(self, conversation: Conversation) -> str:
        """The conversation so far, oldest first, as the prompts quote it."""
        chosen: list[str] = []
        spent = 0
        for turn in reversed(conversation.turns[-self.recent_turns:]):
            block = self.block(turn)
            if chosen and spent + len(block) > self.history_chars:
                break
            chosen.insert(0, block)
            spent += len(block)
        parts = []
        if conversation.summary:
            parts.append(f"Earlier in this conversation (summary):\n{conversation.summary}")
        parts.extend(chosen)
        return "\n\n".join(parts) or NO_HISTORY

    def overflow(self, conversation: Conversation) -> list[Turn]:
        """The turns to fold into the summary, oldest first, or none yet."""
        if len(conversation.turns) <= self.fold_after:
            return []
        return conversation.turns[:len(conversation.turns) - self.recent_turns]

    def block(self, turn: Turn) -> str:
        reply = turn.reply.strip()
        if len(reply) > self.reply_chars:
            reply = reply[:self.reply_chars].rstrip() + " …"
        return f"Student: {turn.message.strip()}\nTutor: {reply}"


def fold_if_due(
    store: ConversationStore,
    window: HistoryWindow,
    llm,
    *,
    user_id: Optional[str],
    user_type: Optional[str],
    session_id: Optional[str],
) -> bool:
    """Fold the oldest turns into the running summary when the window is full.

    Runs after a reply is composed. A failed summary leaves every turn in place,
    so the next turn sees more history rather than less.
    """
    if llm is None or not store.remembers(user_id, user_type, session_id):
        return False
    conversation = store.load(user_id, session_id)
    folded = window.overflow(conversation)
    if not folded:
        return False
    transcript = "\n\n".join(window.block(turn) for turn in folded)
    prompt = CONVERSATION_SUMMARY_PROMPT.render(
        summary=conversation.summary or "(none yet)", transcript=transcript)
    try:
        summary = (llm.invoke([HumanMessage(content=prompt)]).content or "").strip()
    except Exception:
        logger.warning("conversation summary failed; keeping the turns", exc_info=True)
        return False
    if not summary:
        logger.warning("conversation summary came back empty; keeping the turns")
        return False
    done = store.fold(user_id, user_type, session_id, summary=summary, folded=folded)
    logger.info("conversation folded", extra={"turns": len(folded), "stored": done})
    return done
