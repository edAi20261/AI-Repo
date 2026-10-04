"""Talking to the tutor.

Push-to-talk: the student records, stops, and their words appear in the message
box to be read and sent like anything they typed. The tutor's reply is the same
reply, read aloud. Voice changes how a turn arrives and how it is heard, never
what the tutor answers: the transcript goes through the same chat turn as text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from langchain_core.messages import HumanMessage

from shared import speech
from shared.config import Config
from shared.errors import UnprocessableError
from shared.llm import maybe_llm

from ..prompts import SPEECH_SCRIPT_PROMPT
from .spoken import has_notation, readable

logger = logging.getLogger(__name__)

_SIGNATURES = (
    (b"\x1aE\xdf\xa3", "audio/webm"),
    (b"OggS", "audio/ogg"),
    (b"fLaC", "audio/flac"),
    (b"ID3", "audio/mpeg"),
)
GENERIC_TYPES = frozenset({"", "application/octet-stream", "binary/octet-stream"})

_LANGUAGES = {"ar": "ar", "ara": "ar", "arz": "ar", "en": "en", "eng": "en",
              "fr": "fr", "fra": "fr", "fre": "fr", "de": "de", "deu": "de", "ger": "de"}


@dataclass(frozen=True)
class Heard:
    text: str
    language: Optional[str]
    seconds: float

    @property
    def heard(self) -> bool:
        return bool(self.text)


@dataclass(frozen=True)
class Spoken:
    audio: bytes
    content_type: str
    voice: str
    characters: int


def hear(audio: bytes, content_type: str, *, config: Config,
         language: Optional[str] = None) -> Heard:
    """A recorded question, as the words the student said."""
    transcript = speech.transcribe(audio, _bare(content_type), config=config,
                                   language=language)
    heard = Heard(text=transcript.text,
                  language=_LANGUAGES.get(transcript.language or "") if transcript.text else None,
                  seconds=transcript.seconds)
    logger.info("speech transcribed", extra={
        "bytes": len(audio), "seconds": round(heard.seconds, 1),
        "language": heard.language, "provider_language": transcript.language,
        "heard": heard.heard})
    return heard


def speak(text: str, language: str, *, config: Config, llm: Any = None) -> Spoken:
    """A tutor reply read aloud, in the voice configured for its language."""
    clean = cap(readable(text), config.get("voice.speak.max_chars"))
    if not clean:
        raise UnprocessableError("There is nothing in this reply to read aloud.")
    spoken = script(clean, language, config=config, llm=llm)
    voice = config.get(f"voice.speak.voices.{language}") or config.get("voice.speak.voices.en")
    audio = speech.synthesize(spoken, voice, language, config=config)
    logger.info("reply spoken", extra={
        "language": language, "voice": voice, "characters": len(spoken),
        "bytes": len(audio)})
    return Spoken(audio=audio, content_type=config.get("voice.speak.content_type"),
                  voice=voice, characters=len(spoken))


def script(text: str, language: str, *, config: Config, llm: Any = None) -> str:
    """``text`` with its notation written as words, by the model, in ``language``.

    Without notation there is nothing to say differently, and no call is made.
    A rewrite that fails, or comes back far longer than what it was given, is
    not spoken: the text is read as it is, and the reason is logged.
    """
    if not has_notation(text):
        return text
    llm = llm if llm is not None else maybe_llm(config, "speech")
    if llm is None:
        logger.warning("no model for the speech script; notation will be read as written")
        return text
    prompt = SPEECH_SCRIPT_PROMPT.render(language=language, text=text)
    try:
        written = (llm.invoke([HumanMessage(content=prompt)]).content or "").strip()
    except Exception:
        logger.warning("speech script failed; notation will be read as written", exc_info=True)
        return text
    if not written or len(written) > config.get("voice.speak.script_max_growth") * len(text):
        logger.warning("speech script rejected; notation will be read as written", extra={
            "characters_in": len(text), "characters_out": len(written)})
        return text
    return written


def audio_type(content: bytes, declared: Optional[str]) -> Optional[str]:
    """What the recording is: as declared, or read off its first bytes when a proxy
    relabelled it as a generic upload on the way here. None when it is not audio
    we recognise."""
    bare = _bare(declared or "")
    if bare not in GENERIC_TYPES:
        return bare
    for signature, kind in _SIGNATURES:
        if content.startswith(signature):
            return kind
    if content[4:8] == b"ftyp":
        return "audio/mp4"
    if content[:4] == b"RIFF" and content[8:12] == b"WAVE":
        return "audio/wav"
    if len(content) > 1 and content[0] == 0xFF and content[1] & 0xE0 == 0xE0:
        return "audio/mpeg"
    return None


def cap(text: str, limit: int) -> str:
    """``text`` cut at the last sentence that fits in ``limit`` characters."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    end = max(head.rfind(mark) for mark in (".", "!", "?", "؟", "\n"))
    return (head[: end + 1] if end > limit // 2 else head.rsplit(" ", 1)[0]).strip()


def _bare(content_type: str) -> str:
    """``audio/webm;codecs=opus`` as ``audio/webm``: providers refuse the parameter."""
    return (content_type or "").split(";", 1)[0].strip().lower()
