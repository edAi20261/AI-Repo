"""Building a persona from what a teacher uploaded.

Two things come out of a build. The card is a description of how the teacher
teaches, which the teacher reads and corrects. The examples are passages of the
teacher's own words, kept so the tutor can be shown how they actually sound —
a description alone gets the tone and misses the phrasing.

A build always ends as a draft. It reaches students only once the teacher has
approved it (``store.activate``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from shared.config import Config
from shared.embeddings import get_embedder
from shared.errors import ServiceUnavailableError, UnprocessableError
from shared.llm import maybe_llm
from shared.source import SourceError, materialise_remote
from shared.structured import call_structured, schema_hint

from ..models import PersonaCard
from ..prompts import PERSONA_CARD_PROMPT, PERSONA_EXEMPLARS_PROMPT
from .material import Limits, SourceRefused, appears_in, fold_corpus, read_material
from .models import (
    Exemplar,
    Material,
    Passage,
    Persona,
    PersonaSource,
    QuestionnaireAnswer,
    SkippedSource,
)
from .store import NewPersona, PersonaStore

logger = logging.getLogger(__name__)

TASK = "persona"


class ExemplarPick(BaseModel):
    number: int = Field(description="The passage's number, as given.")
    shows: str = Field(description="In a few words, what this passage shows about how "
                                   "the teacher explains.")


class ExemplarPicks(BaseModel):
    picks: list[ExemplarPick]


@dataclass(frozen=True)
class BuildRequest:
    teacher_id: str
    teacher_name: str
    sources: list[PersonaSource]
    questionnaire: list[QuestionnaireAnswer]
    language: str
    consented_at: datetime


@dataclass(frozen=True)
class BuildResult:
    persona: Persona
    exemplars: list[Exemplar]
    skipped: list[SkippedSource]


Reader = Callable[[PersonaSource, Config], str]


def build_persona(request: BuildRequest, *, store: PersonaStore, config: Config,
                  llm: Any = None, embedder: Any = None,
                  reader: Optional[Reader] = None) -> BuildResult:
    """Read the uploads, draft the card, choose the examples, store a draft."""
    llm = llm if llm is not None else maybe_llm(config, TASK)
    if llm is None:
        raise ServiceUnavailableError("No model is configured to build a persona.")
    limits = _limits(config)
    loaded, unreadable = _load(request.sources, config, reader or read_source)
    material = read_material(loaded, limits)
    skipped = unreadable + material.skipped
    if len(material.passages) < config.get("persona.min_passages") and not request.questionnaire:
        raise UnprocessableError(
            "There is not enough of the teacher's own words to learn from. Upload more "
            "conversations or explanations, or answer the questionnaire.",
            details={"passages": len(material.passages),
                     "skipped": [s.__dict__ for s in skipped]})

    card = draft_card(material, request, llm=llm, config=config)
    exemplars = choose_exemplars(material.passages, llm=llm, config=config,
                                 embedder=embedder or get_embedder(config))
    persona = store.create(NewPersona(
        teacher_id=request.teacher_id, teacher_name=request.teacher_name, card=card,
        consented_at=request.consented_at,
        sources=_provenance(request, material, skipped)), exemplars)
    logger.info("persona drafted", extra={
        "teacher_id": request.teacher_id, "version": persona.version,
        "passages": len(material.passages), "exemplars": len(exemplars),
        "skipped": len(skipped)})
    return BuildResult(persona=persona, exemplars=exemplars, skipped=skipped)


def read_source(source: PersonaSource, config: Config) -> str:
    """The text of one upload, sent inline or fetched from its URI."""
    if source.text is not None:
        return source.text
    with materialise_remote(source.uri, config) as path:
        return Path(path).read_text(encoding="utf-8", errors="replace")


def draft_card(material: Material, request: BuildRequest, *, llm: Any,
               config: Config) -> PersonaCard:
    corpus = fold_corpus([material.corpus, *(qa.answer for qa in request.questionnaire)])
    prompt = PERSONA_CARD_PROMPT.render(
        language=request.language,
        schema=schema_hint(PersonaCard),
        questionnaire=_questionnaire(request.questionnaire),
        passages=_numbered(_sample(material.passages,
                                   config.get("persona.card_sample_chars"))) or "(none)")
    return call_structured(PersonaCard, [HumanMessage(content=prompt)], task=TASK,
                           config=config, llm=llm, label="persona_card",
                           validate=_phrases_in(corpus))


def choose_exemplars(passages: Sequence[Passage], *, llm: Any, config: Config,
                     embedder: Any) -> list[Exemplar]:
    if not passages:
        return []
    limit = config.get("persona.max_exemplars")
    candidates = _sample(passages, config.get("persona.selection_sample_chars"))
    by_number = {n: p for n, p in enumerate(candidates, start=1)}
    prompt = PERSONA_EXEMPLARS_PROMPT.render(
        limit=limit, schema=schema_hint(ExemplarPicks), passages=_numbered(candidates))
    picks = call_structured(ExemplarPicks, [HumanMessage(content=prompt)], task=TASK,
                            config=config, llm=llm, label="persona_exemplars",
                            validate=_picks_among(set(by_number), limit)).picks
    chosen = [(by_number[p.number], " ".join(p.shows.split())[:200]) for p in picks]
    vectors = embedder.embed([_embedded_text(passage) for passage, _ in chosen])
    return [Exemplar(text=passage.text, prompt=passage.prompt, move=shows,
                     source_id=passage.source_id, embedding=vector)
            for (passage, shows), vector in zip(chosen, vectors)]


def _embedded_text(passage: Passage) -> str:
    return f"{passage.prompt}\n{passage.text}".strip()


def _load(sources: Sequence[PersonaSource], config: Config,
          reader: Reader) -> tuple[list[tuple[PersonaSource, str]], list[SkippedSource]]:
    loaded: list[tuple[PersonaSource, str]] = []
    skipped: list[SkippedSource] = []
    for source in sources:
        try:
            text = reader(source, config)
        except (SourceError, SourceRefused, OSError) as error:
            reason = getattr(error, "message", None) or str(error)
            logger.warning("persona source unusable", extra={
                "source_id": source.source_id, "kind": source.kind.value, "reason": reason})
            skipped.append(SkippedSource(source.source_id, reason))
            continue
        loaded.append((source, text))
    return loaded, skipped


def _phrases_in(corpus: str) -> Callable[[PersonaCard], None]:
    def check(card: PersonaCard) -> None:
        invented = [p for p in card.phrases if not appears_in(p, corpus)]
        if invented:
            raise ValueError(
                "these phrases do not appear in the teacher's material; copy phrases "
                f"exactly or leave them out: {invented}")
    return check


def _picks_among(numbers: set[int], limit: int) -> Callable[[ExemplarPicks], None]:
    def check(batch: ExemplarPicks) -> None:
        chosen = [p.number for p in batch.picks]
        unknown = sorted(set(chosen) - numbers)
        if unknown:
            raise ValueError(f"these numbers are not passages: {unknown}")
        if len(chosen) != len(set(chosen)):
            raise ValueError("a passage was chosen more than once")
        if len(chosen) > limit:
            raise ValueError(f"choose at most {limit} passages")
    return check


def _sample(passages: Sequence[Passage], budget: int) -> list[Passage]:
    """Passages spread evenly across the uploads until the budget is spent, so a
    long first upload cannot crowd out every other one."""
    if not passages:
        return []
    total = sum(len(p.text) + len(p.prompt) for p in passages)
    if total <= budget:
        return list(passages)
    step = total / budget
    picked, spent, position = [], 0, 0.0
    while position < len(passages):
        passage = passages[int(position)]
        cost = len(passage.text) + len(passage.prompt)
        if spent + cost > budget:
            break
        if not picked or picked[-1] is not passage:
            picked.append(passage)
            spent += cost
        position += step
    return picked


def _numbered(passages: Sequence[Passage]) -> str:
    blocks = []
    for number, passage in enumerate(passages, start=1):
        asked = f"(The student had said: {passage.prompt})\n" if passage.prompt else ""
        blocks.append(f"[{number}]\n{asked}{passage.text}")
    return "\n\n".join(blocks)


def _questionnaire(answers: Sequence[QuestionnaireAnswer]) -> str:
    if not answers:
        return "(the teacher did not answer the questionnaire)"
    return "\n".join(f"Q: {' '.join(qa.question.split())}\nA: {' '.join(qa.answer.split())}"
                     for qa in answers)


def _provenance(request: BuildRequest, material: Material,
                skipped: list[SkippedSource]) -> dict[str, Any]:
    return {
        "used": material.used,
        "skipped": [{"source_id": s.source_id, "reason": s.reason} for s in skipped],
        "passages": len(material.passages),
        "questionnaire_answers": len(request.questionnaire),
    }


def _limits(config: Config) -> Limits:
    return Limits(
        min_passage_chars=config.get("persona.min_passage_chars"),
        max_passage_chars=config.get("persona.max_passage_chars"),
        max_prompt_chars=config.get("persona.max_prompt_chars"),
        max_source_chars=config.get("persona.max_source_chars"),
    )
