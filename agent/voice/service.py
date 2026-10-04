"""Talking to the tutor.

Push-to-talk: the student records, stops, and their words appear in the message
box to be read and sent like anything they typed. The tutor's reply is the same
reply, read aloud. Voice changes how a turn arrives and how it is heard, never
what the tutor answers: the transcript goes through the same chat turn as text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from shared import speech
from shared.config import Config
from shared.errors import UnprocessableError

from .spoken import speakable

logger = logging.getLogger(__name__)

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
    heard = Heard(text=transcript.text, language=_LANGUAGES.get(transcript.language or ""),
                  seconds=transcript.seconds)
    logger.info("speech transcribed", extra={
        "bytes": len(audio), "seconds": round(heard.seconds, 1),
        "language": heard.language, "provider_language": transcript.language,
        "heard": heard.heard})
    return heard


def speak(text: str, language: str, *, config: Config) -> Spoken:
    """A tutor reply read aloud, in the voice configured for its language."""
    spoken = cap(speakable(text, language), config.get("voice.speak.max_chars"))
    if not spoken:
        raise UnprocessableError("There is nothing in this reply to read aloud.")
    voice = config.get(f"voice.speak.voices.{language}") or config.get("voice.speak.voices.en")
    audio = speech.synthesize(spoken, voice, language, config=config)
    logger.info("reply spoken", extra={
        "language": language, "voice": voice, "characters": len(spoken),
        "bytes": len(audio)})
    return Spoken(audio=audio, content_type=config.get("voice.speak.content_type"),
                  voice=voice, characters=len(spoken))


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
