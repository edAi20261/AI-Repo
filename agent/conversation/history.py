"""The conversation as a model reads it: recent turns verbatim, the rest summarised.

A window of whole turns, newest first, within a token budget, so a long answer
cannot push the question it answered out of view. Turns that fall out of the
window are folded into one running summary written by a model, which keeps
goals, facts and open questions rather than a transcript.

Folding does not destroy the turns it covers. While the whole conversation
still fits the rewrite budget the summary is written from what was said, not
from the summary before it: a summary of a summary loses detail every time it
is taken, and nothing downstream can tell that it has. Only once there is more
past than a rewrite can read does the fold degrade to folding the old summary
into the new one, and then it says so in the log.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from langchain_core.messages import HumanMessage

from shared.tokens import clip_to_tokens, count_tokens

from ..prompts import CONVERSATION_SUMMARY_PROMPT
from .records import Conversation, Turn
from .store import ConversationStore

logger = logging.getLogger(__name__)

# Sized against the 93 conversation turns in production, which run to a median
# of 705 characters a turn and 1305 at the ninetieth percentile — 203 and 376
# tokens. The budgets this replaced were 6 turns within 6000 characters, and on
# any conversation above the median the character bound bit first: six turns
# at the ninetieth percentile is 7830 characters, so the window was quietly
# handing over four turns while claiming six.
#
# Twelve turns at the ninetieth percentile is 4512 tokens, inside the budget
# below, so the turn count is the policy and the token budget is the backstop —
# which is the way round it should be. It costs about 2400 tokens a turn more
# than six turns did, against a million-token context window.
DEFAULT_RECENT_TURNS = 12
DEFAULT_FOLD_AFTER = 24
DEFAULT_HISTORY_TOKENS = 6000
DEFAULT_REPLY_TOKENS = 600
# Roughly sixty turns of raw conversation. Past this the fold stops rewriting
# from the conversation and starts folding its own summary forward, so this is
# the point where memory begins to decay rather than merely shorten.
DEFAULT_REBUILD_TOKENS = 12000

NO_HISTORY = "(this is the first message of the conversation)"

NO_SUMMARY_YET = "(none yet)"


@dataclass(frozen=True)
class HistoryWindow:
    """How much of a conversation a prompt carries, and how far back a fold reads."""

    recent_turns: int = DEFAULT_RECENT_TURNS
    fold_after: int = DEFAULT_FOLD_AFTER
    history_tokens: int = DEFAULT_HISTORY_TOKENS
    reply_tokens: int = DEFAULT_REPLY_TOKENS
    rebuild_tokens: int = DEFAULT_REBUILD_TOKENS

    @classmethod
    def from_config(cls, config) -> HistoryWindow:
        settings = config.get("agent.conversation") or {}
        return cls(
            recent_turns=settings.get("recent_turns", DEFAULT_RECENT_TURNS),
            fold_after=settings.get("fold_after", DEFAULT_FOLD_AFTER),
            history_tokens=settings.get("history_tokens", DEFAULT_HISTORY_TOKENS),
            reply_tokens=settings.get("reply_tokens", DEFAULT_REPLY_TOKENS),
            rebuild_tokens=settings.get("rebuild_tokens", DEFAULT_REBUILD_TOKENS),
        )

    def render(self, conversation: Conversation, *, carried_over: str = "",
               standing: str = "") -> str:
        """The conversation so far, oldest first, as the prompts quote it.

        ``carried_over`` is what the tutor knows about this student from before
        this conversation, and stands in for the history only while there is no
        history: it is the weakest evidence here, and the moment there is a turn
        to read the turn is better. ``standing`` is what the student asked of
        every answer, and heads the history on every turn.
        """
        chosen: list[str] = []
        spent = 0
        for turn in reversed(conversation.turns[-self.recent_turns:]):
            block = self.block(turn)
            cost = count_tokens(block)
            if chosen and spent + cost > self.history_tokens:
                break
            chosen.insert(0, block)
            spent += cost
        parts = [standing] if standing else []
        if conversation.summary:
            parts.append(f"Earlier in this conversation (summary):\n{conversation.summary}")
        parts.extend(chosen)
        if chosen or conversation.summary:
            return "\n\n".join(parts)
        return "\n\n".join([*parts, NO_HISTORY, *([carried_over] if carried_over else [])])

    def overflow(self, conversation: Conversation) -> list[Turn]:
        """The turns to fold into the summary, oldest first, or none yet."""
        if len(conversation.turns) <= self.fold_after:
            return []
        return conversation.turns[:len(conversation.turns) - self.recent_turns]

    def block(self, turn: Turn) -> str:
        return (f"Student: {turn.message.strip()}\n"
                f"Tutor: {clip_to_tokens(turn.reply.strip(), self.reply_tokens)}")

    def transcript(self, turns: list[Turn]) -> str:
        return "\n\n".join(self.block(turn) for turn in turns)

    def within_rewrite_budget(self, turns: list[Turn]) -> list[Turn]:
        """The newest run of ``turns`` a rewrite can still afford to read.

        Dropping from the old end is what a fold does anyway; this only decides
        how much of the past is dropped for good rather than summarised again.
        """
        kept: list[Turn] = []
        spent = 0
        for turn in reversed(turns):
            cost = count_tokens(self.block(turn))
            if kept and spent + cost > self.rebuild_tokens:
                break
            kept.insert(0, turn)
            spent += cost
        return kept


def fold_if_due(
    store: ConversationStore,
    window: HistoryWindow,
    llm,
    *,
    user_id: Optional[str],
    user_type: Optional[str],
    session_id: Optional[str],
    language: str,
) -> bool:
    """Fold the oldest turns into the running summary when the window is full.

    Runs after a reply is composed. A failed summary leaves every turn in place
    and every folded turn where it was, so the next turn sees more history
    rather than less — and the next fold tries again with the same material.
    """
    if llm is None or not store.remembers(user_id, user_type, session_id):
        return False
    conversation = store.load(user_id, session_id)
    folding = window.overflow(conversation)
    if not folding:
        return False

    past = [*conversation.folded, *folding]
    keeping = window.within_rewrite_budget(past)
    discard = past[:len(past) - len(keeping)]
    # The rewrite reads the conversation itself only while the conversation is
    # all still here. The moment anything has to be dropped, what was said in
    # those turns survives only in the summary, so the summary has to come with.
    from_conversation = not discard

    if from_conversation:
        summary_before, material = NO_SUMMARY_YET, keeping
    else:
        summary_before, material = (conversation.summary or NO_SUMMARY_YET), folding

    prompt = CONVERSATION_SUMMARY_PROMPT.render(
        language=language,
        summary=summary_before,
        transcript=window.transcript(material))
    try:
        summary = (llm.invoke([HumanMessage(content=prompt)]).content or "").strip()
    except Exception:
        logger.warning("conversation summary failed; keeping the turns", exc_info=True)
        return False
    if not summary:
        logger.warning("conversation summary came back empty; keeping the turns")
        return False

    done = store.fold(user_id, user_type, session_id,
                      summary=summary, folded=folding, discard=discard)
    logger.info("conversation folded", extra={
        "turns": len(folding), "rewritten_from_conversation": from_conversation,
        "kept_raw": len(keeping), "dropped": len(discard),
        "language": language, "stored": done})
    return done
