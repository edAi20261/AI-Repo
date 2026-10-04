"""Speech in and speech out, both through ElevenLabs.

Speech to text is Scribe. On Egyptian Arabic mixed with English, the way our
students actually talk, it was the only production API with a usable error
rate (13.1% WER against 45.9% for the next best, arXiv 2605.19069). It also
detects the language itself, so a student switching between the two
mid-sentence needs no hint.

Text to speech is the same account, so one key covers both directions. Its
voices are multilingual; the reply's language is passed so a voice never
drifts into another one. It has no Egyptian Arabic voice, which is the cost of
one provider instead of two.

Both are one synchronous HTTP call. A failure raises with the provider's reason,
so an expired key and an unreadable recording are not the same "something went
wrong".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .config import Config
from .errors import ServiceUnavailableError, UnprocessableError
from .metering import record_usage

logger = logging.getLogger(__name__)


class SpeechUnavailable(ServiceUnavailableError):
    """The speech provider is not configured, or refused or failed the call."""


class SpeechUnreadable(UnprocessableError):
    """The provider could not read this recording as audio."""


@dataclass(frozen=True)
class Transcript:
    text: str
    language: Optional[str]
    seconds: float


def transcribe(audio: bytes, content_type: str, *, config: Config,
               language: Optional[str] = None) -> Transcript:
    """What was said in ``audio``, in the language it was said in."""
    import httpx

    key = _key(config, "voice.transcribe.api_key_env")
    data = {"model_id": config.get("voice.transcribe.model"),
            "tag_audio_events": "false", "diarize": "false"}
    if language:
        data["language_code"] = language
    try:
        response = httpx.post(
            config.get("voice.transcribe.url"),
            headers={"xi-api-key": key},
            params={"enable_logging": str(config.get("voice.transcribe.provider_logging")).lower()},
            data=data,
            files={"file": ("speech", audio, content_type)},
            timeout=config.get("voice.transcribe.timeout_seconds"),
        )
    except httpx.HTTPError as error:
        logger.warning("transcription call failed", extra={"error": type(error).__name__})
        raise SpeechUnavailable("The speech service could not be reached.") from error
    if response.status_code in (400, 415, 422):
        logger.warning("transcription refused the recording", extra={
            "status": response.status_code, "detail": response.text[:300]})
        raise SpeechUnreadable("The recording could not be read as speech.",
                               details={"provider_status": response.status_code})
    if response.status_code >= 300:
        logger.error("transcription rejected", extra={
            "status": response.status_code, "detail": response.text[:300]})
        raise SpeechUnavailable("The speech service refused the request.",
                                details={"provider_status": response.status_code})

    payload = response.json()
    seconds = _spoken_seconds(payload)
    record_usage({"cost": seconds / 60 * config.get("voice.transcribe.usd_per_minute")})
    return Transcript(text=" ".join(str(payload.get("text") or "").split()),
                      language=payload.get("language_code"), seconds=seconds)


def synthesize(text: str, voice: str, language: str, *, config: Config) -> bytes:
    """``text`` read aloud by ``voice``, held to ``language``."""
    import httpx

    key = _key(config, "voice.speak.api_key_env")
    try:
        response = httpx.post(
            f"{config.get('voice.speak.url')}/{voice}",
            headers={"xi-api-key": key, "content-type": "application/json"},
            params={"output_format": config.get("voice.speak.output_format"),
                    "enable_logging": str(config.get("voice.speak.provider_logging")).lower()},
            json={"text": text, "model_id": config.get("voice.speak.model"),
                  "language_code": language},
            timeout=config.get("voice.speak.timeout_seconds"),
        )
    except httpx.HTTPError as error:
        logger.warning("speech synthesis call failed", extra={"error": type(error).__name__})
        raise SpeechUnavailable("The speech service could not be reached.") from error
    if response.status_code >= 300 or not response.content:
        logger.error("speech synthesis rejected", extra={
            "status": response.status_code, "voice": voice, "detail": response.text[:300]})
        raise SpeechUnavailable("The speech service refused the request.",
                                details={"provider_status": response.status_code})
    record_usage({"cost": len(text) / 1_000_000 * config.get("voice.speak.usd_per_million_chars")})
    return response.content


def _key(config: Config, setting: str) -> str:
    name = config.get(setting)
    value = config.secret(name) if name else None
    if not value:
        raise SpeechUnavailable("Voice is not configured on this service.",
                                details={"missing": name})
    return value


def _spoken_seconds(payload: dict) -> float:
    ends = [w.get("end") for w in payload.get("words") or []
            if isinstance(w, dict) and isinstance(w.get("end"), (int, float))]
    return float(max(ends)) if ends else 0.0
