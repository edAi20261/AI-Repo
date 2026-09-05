"""Provider-agnostic model access: vision, embeddings, LLM.

All model clients are constructed here so a provider swap is a config change, not
a code change. Heavy imports (torch, langchain, httpx) are lazy, so importing
this module stays cheap and never requires an SDK that isn't installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from shared.metering import record_usage

from .config import Config
from .prompts import (
    FIGURE_CAPTION_PROMPT,
    MATH_PAGE_TRANSCRIPTION_PROMPT,
    PAGE_TRANSCRIPTION_PROMPT,
)
from .schemas import FigureBox, FigureCaption, PageTranscription

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
    ) -> FigureCaption:
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
    ) -> FigureCaption:
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
        self._model = config.get("vision.model")
        self._fallback_model = config.get("vision.fallback_model")
        self._base_url = config.get("vision.base_url", "https://openrouter.ai/api/v1")
        self._api_key = config.secret("OPENROUTER_API_KEY")
        self._timeout = config.get("vision.timeout_seconds", 180)
        self._max_tokens = config.get("vision.max_tokens", 4096)

    def transcribe_page(
        self,
        image_png: bytes,
        *,
        language_hint: str | None = None,
        math: bool = False,
    ) -> PageTranscription:
        template = MATH_PAGE_TRANSCRIPTION_PROMPT if math else PAGE_TRANSCRIPTION_PROMPT
        prompt = template.format(language_hint=language_hint or "")
        data = self._complete_json(prompt, image_png)
        if data is None:
            return PageTranscription(text="", figures=[], unreadable=True)
        return _page_from_json(data)

    def caption_figure(
        self,
        image_png: bytes,
        *,
        heading_path: str = "",
        language_hint: str | None = None,
    ) -> FigureCaption:
        prompt = FIGURE_CAPTION_PROMPT.format(
            heading_path=heading_path, language_hint=language_hint or ""
        )
        data = self._complete_json(prompt, image_png)
        if data is None:
            return FigureCaption(caption="", alt_text="")
        return _caption_from_json(data)

    # -- transport -----------------------------------------------------------

    def _complete_json(self, prompt: str, image_png: bytes) -> dict | None:
        content = self._post(prompt, image_png, self._model)
        data = _strip_json(content)
        if data is None and self._fallback_model:
            content = self._post(prompt, image_png, self._fallback_model)
            data = _strip_json(content)
        return data

    def _post(self, prompt: str, image_png: bytes, model: str) -> str | None:
        import base64

        import httpx  # lazy

        img_b64 = base64.b64encode(image_png).decode()
        try:
            resp = httpx.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": model,
                    "max_tokens": self._max_tokens,
                    "temperature": 0,
                    "usage": {"include": True},  # OpenRouter: return per-call cost
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_b64}"
                                    },
                                },
                            ],
                        }
                    ],
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            record_usage(payload.get("usage"))
            return payload["choices"][0]["message"]["content"]
        except Exception:
            return None


def _strip_json(text: str | None) -> dict | None:
    """Recover a JSON object from a response that may wrap it in a ```json fence
    or prose."""
    import json
    import re

    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    else:
        bare = re.search(r"\{.*\}", text, re.S)
        if bare:
            text = bare.group(0)
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _page_from_json(data: dict) -> PageTranscription:
    figures: list[FigureBox] = []
    for fig in data.get("figures") or []:
        bbox = fig.get("bbox") if isinstance(fig, dict) else None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                figures.append(FigureBox(bbox=tuple(float(v) for v in bbox)))
            except (TypeError, ValueError):
                continue
    return PageTranscription(
        text=str(data.get("text") or ""),
        figures=figures,
        unreadable=bool(data.get("unreadable", False)),
    )


def _caption_from_json(data: dict) -> FigureCaption:
    return FigureCaption(
        caption=str(data.get("caption") or ""),
        alt_text=str(data.get("alt_text") or ""),
    )


# Vision gateways implemented. Only OpenRouter (OpenAI-compatible) is wired.
_VISION_GATEWAYS = {"openrouter"}


def get_vision_client(config: Config) -> VisionClient:
    """Real client when engine is ``vision``, ``vision.model`` is set, the provider
    is a supported gateway, and the key is present; else the stub."""
    engine = config.get("ingestion.ocr.engine", "vision")
    provider = (config.get("vision.provider") or "").lower()
    model = config.get("vision.model")
    has_key = bool(config.secret("OPENROUTER_API_KEY"))
    if engine == "vision" and model and provider in _VISION_GATEWAYS and has_key:
        return OpenRouterVisionClient(config)
    return StubVisionClient()


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
