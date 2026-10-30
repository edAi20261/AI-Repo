"""Provider-agnostic model access: vision, embeddings, LLM.

All model clients are constructed here so a provider swap is a config change, not
a code change. Heavy imports (torch, langchain, httpx) are lazy, so importing
this module stays cheap and never requires an SDK that isn't installed.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from shared.vision import read_image

from .config import Config
from .prompts import (
    FIGURE_CAPTION_PROMPT,
    MATH_PAGE_TRANSCRIPTION_PROMPT,
    PAGE_TRANSCRIPTION_PROMPT,
)
from .schemas import FigureCaption, PageTranscription

logger = logging.getLogger(__name__)

# Marks stub output so it can never be mistaken for a real transcription. (used in testing only if vision API is not available)
VISION_STUB_SENTINEL = "[[VISION_STUB]]"


# ===========================================================================
# Vision
# ===========================================================================
# Output contracts (PageTranscription, FigureBox, FigureCaption) live in schemas.


class VisionClient(ABC):
    """Two calls: ``transcribe_page`` (text + figure boxes, one call per page) and
    ``caption_figure`` (one call per cropped figure)."""

    is_stub: bool = False

    @abstractmethod
    def transcribe_page(
        self,
        image_png: bytes,
        *,
        language_hint: str | None = None,
        math: bool = False,
    ) -> PageTranscription:
        """Page text + figure boxes. ``math`` selects the maths-fidelity prompt."""

    @abstractmethod
    def caption_figure(
        self,
        image_png: bytes,
        *,
        heading_path: str = "",
        language_hint: str | None = None,
    ) -> FigureCaption | None:
        """Caption + alt_text for a cropped figure."""


class StubVisionClient(VisionClient):
    """Runs the pipeline with no key. Output carries :data:`VISION_STUB_SENTINEL`
    and no figure boxes are fabricated."""

    is_stub = True

    def transcribe_page(
        self,
        image_png: bytes,
        *,
        language_hint: str | None = None,
        math: bool = False,
    ) -> PageTranscription:
        kind = "maths-page" if math else "scanned-page"
        return PageTranscription(
            text=f"{VISION_STUB_SENTINEL} {kind} transcription pending vision model",
            figures=[],
        )

    def caption_figure(
        self,
        image_png: bytes,
        *,
        heading_path: str = "",
        language_hint: str | None = None,
    ) -> FigureCaption | None:
        return FigureCaption(
            caption=f"{VISION_STUB_SENTINEL} caption pending vision model",
            alt_text=f"{VISION_STUB_SENTINEL} alt_text pending vision model",
        )


class OpenRouterVisionClient(VisionClient):
    """Vision via OpenRouter's OpenAI-compatible ``/chat/completions``.

    Model/fallback/base URL come from ``config.yaml`` (``vision.*``). ``fallback_model``
    is tried only when the primary response can't be parsed; if it still can't, the
    page is marked ``unreadable`` rather than indexed blank.
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    def transcribe_page(
        self,
        image_png: bytes,
        *,
        language_hint: str | None = None,
        math: bool = False,
    ) -> PageTranscription:
        template = MATH_PAGE_TRANSCRIPTION_PROMPT if math else PAGE_TRANSCRIPTION_PROMPT
        page = read_image(
            image_png, template.render(language_hint=language_hint or ""),
            PageTranscription, config=self._config, label="transcribe_page")
        # A page nothing could read is flagged, never indexed blank.
        return page or PageTranscription.unanswered()

    def caption_figure(
        self,
        image_png: bytes,
        *,
        heading_path: str = "",
        language_hint: str | None = None,
    ) -> FigureCaption | None:
        # A caption is read by a student, so it follows the page's language.
        prompt = FIGURE_CAPTION_PROMPT.render(
            language=language_hint or "en", heading_path=heading_path)
        return read_image(image_png, prompt, FigureCaption,
                          config=self._config, label="caption_figure")


# Vision gateways implemented. Only OpenRouter (OpenAI-compatible) is wired.
_VISION_GATEWAYS = {"openrouter", "google", "openai", "vertex"}


def get_vision_client(config: Config) -> VisionClient:
    """The real client, or the stub when this deployment has no model to call.

    The credential is asked of the provider rather than of one hardcoded
    variable. It used to read OPENROUTER_API_KEY whatever the provider was, so
    pointing vision at another gateway returned the stub — and the stub does
    not fail, it returns empty pages. A book ingested that way is indexed
    blank, and nothing in the run says so.

    Which is why a configured-but-unreachable gateway is now loud. Falling back
    to the stub silently is only correct when nobody asked for a real model.
    """
    engine = config.get("ingestion.ocr.engine", "vision")
    provider = (config.get("vision.provider") or "").lower()
    model = config.get("vision.model")
    if engine != "vision" or not model:
        return StubVisionClient()
    if provider not in _VISION_GATEWAYS:
        logger.warning("vision.provider %r is not a gateway this knows; "
                       "pages will not be read by a model", provider)
        return StubVisionClient()
    if not _credential_present(config, provider):
        logger.error(
            "vision is configured for %r but its credential is missing, so every "
            "page would be indexed blank; set it, or set ingestion.ocr.engine "
            "to something other than 'vision' deliberately", provider)
        return StubVisionClient()
    return OpenRouterVisionClient(config)


def _credential_present(config: Config, provider: str) -> bool:
    """Whether the gateway vision points at can actually be called.

    Vertex authenticates with a service account rather than a key, so what
    counts as present differs by provider — and asking the wrong question is
    how a misconfigured deployment silently indexes empty books.
    """
    if provider == "vertex":
        from shared.gcp import credential_source

        return bool(credential_source(config))
    named = config.get("vision.api_key_env")
    if named:
        return bool(config.secret(named))
    from shared.vision import PROVIDER_KEY_ENV

    return bool(config.secret(PROVIDER_KEY_ENV.get(provider, "OPENROUTER_API_KEY")))


# ===========================================================================
# Embeddings — moved to shared.embeddings (one embedder for both halves).
# Re-exported here for the ingestion package's existing imports.
# ===========================================================================

from shared.embeddings import (  # noqa: E402
    Embedder,
    HashEmbedder,
    LocalSentenceTransformerEmbedder,
    get_embedder,
)


# --- provider factory ------------------------------------------------------
# Moved to shared.llm (one chat seam for both halves). Re-exported for the
# ingestion package's existing imports.

from shared.llm import get_llm, llm_key_available  # noqa: E402,F401
