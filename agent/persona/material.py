"""A teacher's uploads, read into passages of their own words.

Reading is deterministic on purpose. What the tutor learns to imitate is copied
from what the teacher actually said, never paraphrased by a model on the way in,
so every example it is later shown can be traced to a line of a real upload.

The students in those uploads are not the teacher's to hand over. Their names,
and any contact details anyone typed, are replaced before a passage exists.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from shared.matching import fold_for_match

from .models import (
    Material,
    Passage,
    PersonaSource,
    SkippedSource,
    SourceKind,
    Utterance,
)

STUDENT = "[student]"

_TIMESTAMPED = re.compile(
    r"^\[?\d{1,4}[/.\-]\d{1,2}[/.\-]\d{1,4},?\s+\d{1,2}:\d{2}(?::\d{2})?"
    r"(?:\s?[AaPp]\.?[Mm]\.?)?\]?\s*(?:-\s*)?(?P<speaker>[^:\n]{1,60}?):\s*(?P<text>.*)$")
_LABELLED = re.compile(r"^(?P<speaker>[^:\n\d][^:\n]{0,39}?)\s*:\s+(?P<text>\S.*)$")
_CUE_TIMING = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3}\s+-->\s+")
_CUE_NUMBER = re.compile(r"^\d+$")
_VOICE_TAG = re.compile(r"^<v(?:\.[^\s>]+)?\s+(?P<speaker>[^>]+)>(?P<text>.*?)(?:</v>)?$")
_MARKUP = re.compile(r"</?[^>]+>")
_EMAIL = re.compile(r"[\w.+\-]+@[\w\-]+\.[\w.\-]+")
_LINK = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\w)\+?\d[\d\s\-().]{7,}\d(?!\w)")
_SENTENCE_END = re.compile(r"(?<=[.!?؟。])\s+")
_SPEAKER_KEYS = ("speaker", "author", "from", "name", "sender", "role")
_TEXT_KEYS = ("text", "message", "content", "body")


@dataclass(frozen=True)
class Limits:
    min_passage_chars: int
    max_passage_chars: int
    max_prompt_chars: int
    max_source_chars: int


class SourceRefused(ValueError):
    """A source that cannot be read as anyone's words in particular."""


def read_material(sources: Sequence[tuple[PersonaSource, str]], limits: Limits) -> Material:
    """Every usable source, as passages of the teacher's words, in upload order."""
    passages: list[Passage] = []
    corpus: list[str] = []
    used: list[str] = []
    skipped: list[SkippedSource] = []
    for source, raw in sources:
        try:
            utterances = redact(utterances_of(source, raw[: limits.max_source_chars]))
        except SourceRefused as refusal:
            skipped.append(SkippedSource(source.source_id, str(refusal)))
            continue
        found = _passages(utterances, source.source_id, limits, start=len(passages))
        if not found:
            skipped.append(SkippedSource(
                source.source_id, "has no stretch of the teacher explaining something"))
            continue
        passages.extend(found)
        corpus.extend(u.text for u in utterances if u.by_teacher)
        used.append(source.source_id)
    return Material(passages=passages, corpus="\n".join(corpus), used=used, skipped=skipped)


def utterances_of(source: PersonaSource, raw: str) -> list[Utterance]:
    text = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise SourceRefused("is empty")
    if source.kind is SourceKind.EXPLANATION:
        return [Utterance(None, part, True) for part in _paragraphs(text)]
    turns = _json_turns(text)
    if turns is None:
        turns = _cue_turns(text) if _is_subtitles(text) else _line_turns(text)
    return _attribute(turns, source.teacher_speaker)


def redact(utterances: list[Utterance]) -> list[Utterance]:
    """The utterances with every other speaker's name and all contact details replaced."""
    names = sorted({u.speaker for u in utterances if u.speaker and not u.by_teacher},
                   key=len, reverse=True)
    patterns = [re.compile(re.escape(n), re.IGNORECASE) for n in _name_forms(names)]
    return [Utterance(u.speaker, _scrub(u.text, patterns), u.by_teacher)
            for u in utterances]


def appears_in(phrase: str, corpus_folded: str) -> bool:
    return bool(phrase) and fold_for_match(phrase).casefold() in corpus_folded


def fold_corpus(texts: Iterable[str]) -> str:
    return fold_for_match("\n".join(texts)).casefold()


def _scrub(text: str, names: list[re.Pattern]) -> str:
    text = _EMAIL.sub("[email]", text)
    text = _LINK.sub("[link]", text)
    text = _PHONE.sub(_phone, text)
    for pattern in names:
        text = pattern.sub(STUDENT, text)
    return text


def _phone(match: re.Match) -> str:
    candidate = match.group(0)
    digits = sum(c.isdigit() for c in candidate)
    dense = digits >= 0.7 * len(candidate.replace(" ", ""))
    return "[phone]" if digits >= 9 and dense else candidate


def _name_forms(names: list[str]) -> list[str]:
    forms: list[str] = []
    for name in names:
        forms.append(name)
        forms.extend(part for part in name.split() if len(part) >= 3 and not part.isdigit())
    return sorted(set(forms), key=len, reverse=True)


def _paragraphs(text: str) -> list[str]:
    return [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]


def _json_turns(text: str) -> Optional[list[tuple[Optional[str], str]]]:
    if text[0] not in "[{":
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    items = data.get("messages") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise SourceRefused("is JSON without a list of messages")
    turns = []
    for item in items:
        if not isinstance(item, dict):
            continue
        speaker = next((str(item[k]) for k in _SPEAKER_KEYS if item.get(k)), None)
        body = _flatten(next((item[k] for k in _TEXT_KEYS if item.get(k)), ""))
        if body.strip():
            turns.append((speaker, body.strip()))
    return turns


def _flatten(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(v if isinstance(v, str) else str(v.get("text", ""))
                       for v in value if isinstance(v, (str, dict)))
    return ""


def _is_subtitles(text: str) -> bool:
    head = text.lstrip("﻿").splitlines()[:4]
    return bool(head) and (head[0].startswith("WEBVTT")
                           or any(_CUE_TIMING.match(line) for line in head))


def _cue_turns(text: str) -> list[tuple[Optional[str], str]]:
    turns: list[tuple[Optional[str], str]] = []
    for line in text.lstrip("﻿").splitlines():
        line = line.strip()
        if (not line or line.startswith(("WEBVTT", "NOTE", "STYLE"))
                or _CUE_NUMBER.match(line) or _CUE_TIMING.match(line) or "-->" in line):
            continue
        voiced = _VOICE_TAG.match(line)
        speaker, body = ((voiced["speaker"].strip(), voiced["text"]) if voiced
                         else (None, line))
        body = _MARKUP.sub("", body).strip()
        if body:
            _append(turns, speaker, body)
    return turns


def _line_turns(text: str) -> list[tuple[Optional[str], str]]:
    turns: list[tuple[Optional[str], str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _TIMESTAMPED.match(line) or _LABELLED.match(line)
        if match:
            turns.append((match["speaker"].strip(), match["text"].strip()))
        elif turns:
            speaker, body = turns[-1]
            turns[-1] = (speaker, f"{body} {line}")
        else:
            turns.append((None, line))
    return [(s, b) for s, b in turns if b and b != "<Media omitted>"]


def _append(turns: list[tuple[Optional[str], str]], speaker: Optional[str], body: str) -> None:
    if turns and turns[-1][0] == speaker:
        turns[-1] = (speaker, f"{turns[-1][1]} {body}")
    else:
        turns.append((speaker, body))


def _attribute(turns: list[tuple[Optional[str], str]],
               teacher_speaker: Optional[str]) -> list[Utterance]:
    speakers = {s for s, _ in turns if s}
    if not speakers:
        return [Utterance(None, body, True) for _, body in turns]
    if teacher_speaker is None:
        if len(speakers) > 1:
            raise SourceRefused(
                "has more than one speaker; teacher_speaker must say which one is the teacher")
        teacher_speaker = next(iter(speakers))
    wanted = _label(teacher_speaker)
    utterances = [Utterance(s, body, bool(s) and _label(s) == wanted) for s, body in turns]
    if not any(u.by_teacher for u in utterances):
        raise SourceRefused(f"has no speaker called {teacher_speaker!r}")
    return utterances


def _label(speaker: str) -> str:
    return fold_for_match(speaker).casefold()


def _passages(utterances: list[Utterance], source_id: str, limits: Limits, *,
              start: int) -> list[Passage]:
    passages: list[Passage] = []
    prompt, run = "", []

    def close() -> None:
        for text in _split(" ".join(run), limits.max_passage_chars):
            if len(text) >= limits.min_passage_chars:
                passages.append(Passage(start + len(passages), source_id, text,
                                        prompt[: limits.max_prompt_chars]))
        run.clear()

    for utterance in utterances:
        if utterance.by_teacher:
            if (run and utterance.speaker is None
                    and len(" ".join(run)) + len(utterance.text) > limits.max_passage_chars):
                close()
            run.append(utterance.text)
        else:
            if run:
                close()
            prompt = utterance.text
    if run:
        close()
    return passages


def _split(text: str, limit: int) -> list[str]:
    text = " ".join(text.split())
    if len(text) <= limit:
        return [text]
    pieces, current = [], ""
    for sentence in _SENTENCE_END.split(text):
        if current and len(current) + len(sentence) + 1 > limit:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return [p[:limit] for p in pieces]
