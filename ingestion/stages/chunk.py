"""chunk stage — structure-aware chunks of ~500-800 tokens.

Each chunk carries lecture_id, its heading path, and page range. Breaks at
level-1 headings and at the token budget; short sections accrete rather than
making tiny fragments; a size split overlaps the previous chunk's tail. Works off
the elements the extract stage tagged — never re-reads the PDF.
"""

from __future__ import annotations

from collections import Counter

from ..context import PipelineContext
from ..state import Chunk, Element, IngestionState
from ..utils.tokens import count_tokens

# Sentence terminators for oversized-element splitting — Latin + Arabic.
_SENTENCE_ENDS = ".!?؟\n"

# Trailing punctuation stripped off heading text so a stored heading_path segment
# does not carry e.g. "Oxidation," (the source-side fix for U3; the agent renders a
# display-side strip as well). Latin + Arabic comma family.
_HEADING_TRAIL = " \t،,;:."


def _clean_heading(text: str) -> str:
    return text.rstrip(_HEADING_TRAIL)


def run(state: IngestionState, ctx: PipelineContext) -> IngestionState:
    cfg = ctx.config.get("ingestion.chunk", {})
    chunker = _Chunker(
        state=state,
        target=cfg.get("target_tokens", 650),
        min_tokens=cfg.get("min_tokens", 400),
        max_tokens=cfg.get("max_tokens", 800),
        overlap=cfg.get("overlap_tokens", 80) if cfg.get("structure_aware", True) else 0,
    )
    chunks = chunker.build(state.get("elements", []))
    state["chunks"] = chunks
    state.setdefault("stats", {})["chunk"] = {
        "chunks": len(chunks),
        "avg_tokens": round(
            sum(count_tokens(c.text) for c in chunks) / len(chunks), 1
        )
        if chunks
        else 0,
    }
    return state


class _Chunker:
    def __init__(self, state, target, min_tokens, max_tokens, overlap) -> None:
        self._state = state
        self._target = target
        self._min = min_tokens
        self._max = max_tokens
        self._overlap = overlap

        self._heading_stack: list[tuple[int, str]] = []
        self._parts: list[tuple[str, int]] = []  # (text, page)
        self._langs: list[str] = []
        self._tokens = 0
        self._order = 0
        self._out: list[Chunk] = []

    def build(self, elements: list[Element]) -> list[Chunk]:
        for el in elements:
            if el.is_heading:
                self._on_heading(el)
            else:
                self._on_body(el)
        self._flush(overlap=False)
        return self._out

    # -- element handlers ----------------------------------------------------

    def _on_heading(self, el: Element) -> None:
        # A heading ends the current chunk once it is big enough — flushed under
        # the *old* stack so the finished chunk keeps its own section's path.
        if self._tokens >= self._min:
            self._flush(overlap=False)

        while self._heading_stack and self._heading_stack[-1][0] >= el.level:
            self._heading_stack.pop()
        self._heading_stack.append((el.level, _clean_heading(el.text)))

        # Keep the heading text in the body so it embeds and short sections grow.
        self._add(el.text, el.page, el.language)

    def _on_body(self, el: Element) -> None:
        el_tokens = count_tokens(el.text)

        # A single element larger than the budget is split on sentence bounds.
        if el_tokens > self._max:
            self._flush(overlap=False)
            for piece in _split_text(el.text, self._target, self._max):
                self._add(piece, el.page, el.language)
                self._flush(overlap=False)
            return

        if self._tokens and self._tokens + el_tokens > self._max:
            self._flush(overlap=True)
        self._add(el.text, el.page, el.language)

    # -- buffer --------------------------------------------------------------

    def _add(self, text: str, page: int, language: str | None) -> None:
        if not text.strip():
            return
        self._parts.append((text, page))
        self._tokens += count_tokens(text)
        if language:
            self._langs.append(language)

    def _flush(self, *, overlap: bool) -> None:
        text = "\n\n".join(p for p, _ in self._parts).strip()
        if not text:
            self._reset()
            return
        pages = [pg for _, pg in self._parts]
        last_page = pages[-1]
        self._out.append(
            Chunk(
                lecture_id=self._state["lecture_id"],
                text=text,
                heading_path=self._path_str(),
                page_from=min(pages),
                page_to=max(pages),
                language=self._majority_language(),
                order_index=self._order,
                version=self._state.get("version", 1),
                namespace=self._state.get("namespace", "syllabus"),
                subject_id=self._state.get("subject_id"),
                grade=self._state.get("grade"),
                subject_name=self._state.get("subject_name"),
                lesson_title=self._state.get("lesson_title"),
                lesson_order=self._state.get("lesson_order"),
            )
        )
        self._order += 1
        self._reset()

        if overlap and self._overlap > 0:
            tail = _tail_by_tokens(text, self._overlap)
            if tail:
                self._parts.append((tail, last_page))
                self._tokens = count_tokens(tail)

    def _reset(self) -> None:
        self._parts = []
        self._langs = []
        self._tokens = 0

    # -- helpers -------------------------------------------------------------

    def _path_str(self) -> str:
        return " > ".join(t for _, t in self._heading_stack)

    def _majority_language(self) -> str:
        if not self._langs:
            return self._state.get("language_hint") or "en"
        return Counter(self._langs).most_common(1)[0][0]


def _split_text(text: str, target: int, max_tokens: int) -> list[str]:
    """Split an oversized block into <= target-token pieces on sentence bounds."""
    sentences = _sentences(text)
    pieces: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for sent in sentences:
        st = count_tokens(sent)
        if buf and buf_tokens + st > target:
            pieces.append(" ".join(buf).strip())
            buf, buf_tokens = [], 0
        buf.append(sent)
        buf_tokens += st
    if buf:
        pieces.append(" ".join(buf).strip())
    return [p for p in pieces if p]


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    for ch in text:
        cur.append(ch)
        if ch in _SENTENCE_ENDS:
            out.append("".join(cur).strip())
            cur = []
    if cur:
        out.append("".join(cur).strip())
    return [s for s in out if s]


def _tail_by_tokens(text: str, overlap_tokens: int) -> str:
    """Trailing slice of ``text`` about ``overlap_tokens`` long, on a word bound."""
    words = text.split()
    if not words:
        return ""
    tail: list[str] = []
    total = 0
    for word in reversed(words):
        tail.append(word)
        total += count_tokens(word)
        if total >= overlap_tokens:
            break
    return " ".join(reversed(tail))
