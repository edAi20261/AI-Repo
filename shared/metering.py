"""Per-run LLM usage/cost metering, shared by both halves.

OpenRouter returns the real cost of each call in the response ``usage`` block when
the request opts in (``usage: {include: true}``). Both the LangChain chat path
(topics/dependencies) and the raw-httpx vision path report into a Meter held in a
context variable, so a run can surface cost + tokens without any dashboard access.

No-op by default: if nothing called :func:`start_meter` for the current context,
``record_usage`` does nothing — so the agent runtime and tests pay no attention to it.
"""

from __future__ import annotations

import contextvars
import threading
from dataclasses import dataclass, field

_METER: contextvars.ContextVar["Meter | None"] = contextvars.ContextVar(
    "llm_meter", default=None)


@dataclass
class Meter:
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, cost_usd: float, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self.cost_usd += cost_usd or 0.0
            self.prompt_tokens += prompt_tokens or 0
            self.completion_tokens += completion_tokens or 0
            self.calls += 1

    def as_dict(self) -> dict:
        return {
            "cost_usd": round(self.cost_usd, 6),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "calls": self.calls,
        }


def start_meter() -> Meter:
    """Install a fresh Meter for the current context and return it."""
    meter = Meter()
    _METER.set(meter)
    return meter


def _first(usage: dict, *keys):
    for key in keys:
        value = usage.get(key)
        if value is not None:
            return value
    return None


def record_usage(usage) -> None:
    """Record an OpenAI/OpenRouter-style ``usage`` dict into the current meter.

    Reads ``cost`` (OpenRouter) and prompt/completion tokens defensively; silently
    does nothing when there is no active meter or no usable usage object.
    """
    meter = _METER.get()
    if meter is None or not isinstance(usage, dict):
        return
    cost = _first(usage, "cost", "total_cost") or 0.0
    prompt = _first(usage, "prompt_tokens", "input_tokens") or 0
    completion = _first(usage, "completion_tokens", "output_tokens") or 0
    try:
        meter.record(float(cost), int(prompt), int(completion))
    except (TypeError, ValueError):
        meter.record(0.0, 0, 0)


# LangChain callback: capture usage from every chat completion. Import guarded so
# importing this module never requires langchain.
try:
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # pragma: no cover
    BaseCallbackHandler = object  # type: ignore[assignment,misc]


class MeteringCallback(BaseCallbackHandler):  # type: ignore[misc]
    """Records token usage/cost from each LangChain chat call into the meter."""

    def on_llm_end(self, response, **kwargs) -> None:  # noqa: D401
        usage = None
        out = getattr(response, "llm_output", None)
        if isinstance(out, dict):
            usage = out.get("token_usage") or out.get("usage")
        if not usage:  # fall back to the first generation's response_metadata
            try:
                message = response.generations[0][0].message
                meta = getattr(message, "response_metadata", {}) or {}
                usage = meta.get("token_usage") or meta.get("usage")
            except Exception:
                usage = None
        record_usage(usage)
