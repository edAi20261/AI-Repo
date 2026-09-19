"""Turning a page's layout into the rows the pipeline already writes.

``PageChunk`` is what the chunker decided; ``Chunk`` is the record written to
``ai.chunks`` and indexed in Qdrant. Everything the layout stage learned that
the record has no column for — what kind of block this was, which figure its
callouts belong to, the answer key it must never embed — rides in the payload,
where retrieval can filter on it.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

from ..state import Chunk
from .blocks import ParsedPage
from .chunker import PageChunk, page_to_chunks


def chunks_for(
    pages: Sequence[ParsedPage],
    *,
    lecture_id: str,
    language: str,
    version: int = 1,
    namespace: str = "syllabus",
    subject_id: Optional[str] = None,
    grade: Optional[str] = None,
    subject_name: Optional[str] = None,
    lesson_title: Optional[str] = None,
    lesson_order: Optional[int] = None,
) -> list[Chunk]:
    """Every page's chunks, in page order, numbered for the whole lesson."""
    rows: list[Chunk] = []
    # A section runs past the page it started on. Without carrying its heading
    # forward, every chunk after the first on a spread has no heading at all —
    # which costs the context prefix its section name and leaves the chunk
    # findable only by its own words.
    heading = ""
    for page in pages:
        for chunk in page_to_chunks(page):
            if chunk.heading:
                heading = chunk.heading
            elif heading:
                chunk.heading = heading
            rows.append(_to_row(
                chunk, order_index=len(rows), lecture_id=lecture_id,
                language=language, version=version, namespace=namespace,
                subject_id=subject_id, grade=grade, subject_name=subject_name,
                lesson_title=lesson_title, lesson_order=lesson_order,
            ))
    return rows


def _to_row(chunk: PageChunk, *, order_index: int, **fields) -> Chunk:
    return Chunk(
        text=chunk.text,
        heading_path=chunk.heading or "",
        page_from=chunk.page,
        page_to=chunk.page,
        order_index=order_index,
        **fields,
    )


def payload_for(chunk: PageChunk) -> dict:
    """What a layout chunk adds to its Qdrant payload.

    ``answer_key`` is stored and never embedded: a self-check page prints its
    answers upside down at the bottom, and a search for the concept must not
    return the answer sheet.
    """
    payload = {
        "block_type": chunk.block_type,
        "has_figure": chunk.has_figure,
    }
    if chunk.figure_id:
        payload["figure_id"] = chunk.figure_id
    if chunk.page_image:
        payload["page_image"] = chunk.page_image
    if chunk.answer_key:
        payload["answer_key"] = chunk.answer_key
    return payload


def parsed_pages(raw: Iterable[dict | ParsedPage]) -> list[ParsedPage]:
    """Rehydrate pages that travelled through the pipeline state as plain data."""
    return [p if isinstance(p, ParsedPage) else ParsedPage.model_validate(p)
            for p in raw]
