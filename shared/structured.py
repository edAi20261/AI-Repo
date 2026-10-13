"""Asking a model for data and getting data back, or an error that says why.

A model asked for JSON can answer with prose, truncate mid-object, refuse
outright, or return a shape that validates as nothing. The old helper collapsed
all four into an empty dict, which callers could not tell apart from a real empty
result — a grading batch that failed to parse became "nothing was wrong with any
answer", and mastery was written from it.

Every structured call now goes through one contract: schema-enforced where the
provider supports it, retried once with feedback the model can act on, and
otherwise raised as a typed failure the caller must handle.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from collections.abc import Callable, Sequence
from typing import Any, Optional, TypeVar

from langchain_core.exceptions import ModelRateLimitError
from langchain_core.messages import HumanMessage
from openai import LengthFinishReasonError
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError, ValidationError

from .config import Config
from .errors import ServiceUnavailableError
from .llm import get_llm

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)

DEFAULT_ATTEMPTS = 2
MAX_REPORTED_ERRORS = 8
MAX_OBSERVED_CHARS = 80
MAX_ERROR_CHARS = 300

REFUSAL_FINISH_REASONS = frozenset({"content_filter"})
TRUNCATION_FINISH_REASONS = frozenset({"length", "max_tokens"})


class StructuredOutputError(ServiceUnavailableError):
    """A structured call did not produce a usable object."""

    def __init__(
        self,
        message: str,
        *,
        task: str,
        label: str,
        attempts: int = 1,
        errors: Optional[list[str]] = None,
        raw: Optional[list[str]] = None,
    ) -> None:
        super().__init__(message)
        self.task = task
        self.label = label
        self.attempts = attempts
        self.errors = errors or []
        self.raw = raw or []

    def log_fields(self) -> dict[str, Any]:
        return {"task": self.task, "label": self.label, "attempts": self.attempts,
                "errors": self.errors[:MAX_REPORTED_ERRORS]}


class StructuredValidationFailure(StructuredOutputError):
    """The model's output never satisfied the schema."""


class StructuredRefusal(StructuredOutputError):
    """The model declined the request. Retrying a policy decision wastes tokens."""


class StructuredTruncation(StructuredOutputError):
    """The reply was cut off by the token limit. A retry produces the same cut."""


class StructuredProviderError(StructuredOutputError):
    """The gateway answered with a body its own SDK could not read.

    The model never saw a schema it could get wrong, so feeding a correction back
    is answering the wrong question: the request is retried unchanged instead.
    """


def _as_conversation(messages: Any) -> list[Any]:
    """One prompt or a list of them, as a list of messages.

    A bare string is a prompt, not a sequence of characters — but it satisfies
    ``Sequence[Any]``, so ``list(messages)`` quietly turned a 4,000 character
    prompt into 4,000 single-character messages. Two ingestion call sites passed
    a string, and it looked like it worked: Gemini merges consecutive user turns
    back into one, so the model saw something close to the prompt. Moving the
    same call to a provider that does not merge produced an answer about
    nothing, with no error anywhere.
    """
    if isinstance(messages, str):
        return [HumanMessage(content=messages)]
    return list(messages)


def call_structured(
    schema: type[M],
    messages: Sequence[Any] | str,
    *,
    task: str,
    config: Config,
    llm: Any = None,
    label: Optional[str] = None,
    attempts: Optional[int] = None,
    validate: Optional[Callable[[M], None]] = None,
) -> M:
    """Return a validated ``schema`` instance, or raise :class:`StructuredOutputError`.

    The retry carries the validation errors back to the model in the form research
    on self-correction found works — where it went wrong, what it sent, and what
    was allowed — because a bare "invalid JSON" gives it nothing to act on.
    """
    name = label or schema.__name__
    settings = config.llm_for(task) if config else {}
    budget = attempts or settings.get("structured_attempts") or DEFAULT_ATTEMPTS
    client = llm if llm is not None else get_llm(config, task)
    runnable = _bind(client, schema, settings)

    conversation = _as_conversation(messages)
    failures: list[str] = []
    raw_replies: list[str] = []

    room = settings.get("max_tokens")
    for attempt in range(1, budget + 1):
        try:
            result = _invoke(runnable, conversation, config=config,
                             task=task, label=name)
        except _ProviderFailure as failure:
            raise StructuredProviderError(
                f"{name}: the model gateway returned a reply that could not be read.",
                task=task, label=name, attempts=attempt,
                errors=[str(failure.cause)[:MAX_ERROR_CHARS]], raw=raw_replies,
            ) from failure.cause
        except PARSE_ERRORS as exc:
            # A reply the provider could not parse arrives as an exception from
            # inside its own parser, not as a parsing_error on the result — so
            # the retry this function exists for never ran, and one truncated
            # reply killed a whole ingest at the topics stage.
            feedback = _reask(f"the reply could not be parsed: {exc}")
            failures.append(feedback)
            logger.warning("structured output could not be parsed",
                           extra={"task": task, "label": name, "attempt": attempt,
                                  "error": type(exc).__name__,
                                  "error_detail": str(exc)[:MAX_ERROR_CHARS]})
            if attempt >= budget:
                break
            if _ran_out_of_room(exc):
                # A model that ran out of room did not get anything wrong, so
                # asking it to try again in the same space asks for the same
                # reply. Give it more room instead.
                room = _more_room(room)
                runnable = _bind(_with_room(client, room), schema, settings)
            else:
                # A rule the schema enforces — an MCQ with no correct option —
                # is a mistake the model can fix once it is told which one.
                conversation = _with_feedback(messages, "", feedback)
            continue
        raw = result.get("raw") if isinstance(result, dict) else None
        parsed = result.get("parsed") if isinstance(result, dict) else result
        raw_text = _text_of(raw)
        if raw_text:
            raw_replies.append(raw_text)

        _guard_refusal(raw, task=task, label=name, attempt=attempt, raw_replies=raw_replies)

        candidate = parsed if isinstance(parsed, schema) else _recover(raw_text, schema)
        if candidate is not None:
            # A schema cannot express "one item per block, with the ids I gave you".
            # Caller-supplied checks raise ValueError and become retry feedback,
            # so an arity or identity mistake is corrected rather than mapped
            # blindly by position onto the wrong questions.
            try:
                if validate is not None:
                    validate(candidate)
            except ValueError as exc:
                feedback = _reask(str(exc))
                failures.append(feedback)
                if attempt < budget:
                    conversation = _with_feedback(messages, raw_text, feedback)
                    continue
                raise StructuredValidationFailure(
                    f"{name}: {exc}", task=task, label=name, attempts=attempt,
                    errors=failures, raw=raw_replies,
                ) from exc
            if attempt > 1:
                logger.info("structured output recovered on retry",
                            extra={"task": task, "label": name, "attempt": attempt})
            return candidate

        error = result.get("parsing_error") if isinstance(result, dict) else None
        feedback = _feedback(error, schema, raw_text)
        failures.append(feedback)
        logger.warning("structured output rejected",
                       extra={"task": task, "label": name, "attempt": attempt,
                              "error_detail": str(error)[:MAX_ERROR_CHARS] if error else "",
                              "reason": feedback.splitlines()[0] if feedback else ""})
        if attempt < budget:
            conversation = _with_feedback(messages, raw_text, feedback)

    raise StructuredValidationFailure(
        f"{name}: the model did not return valid output after {budget} attempts.",
        task=task, label=name, attempts=budget, errors=failures, raw=raw_replies,
    )


def _with_feedback(messages: Sequence[Any] | str, raw_text: str, feedback: str) -> list[Any]:
    """The original request, the model's failed reply, and what to fix."""
    return _as_conversation(messages) + [
        {"role": "assistant", "content": raw_text or ""},
        {"role": "user", "content": feedback},
    ]


def _reask(problem: str) -> str:
    return (
        "Your previous reply did not satisfy the request. Fix ONLY this and return "
        "the corrected JSON object — same shape, no prose, no code fence.\n\n"
        + f"- {problem}"
    )


def align_by_id(
    expected: Sequence[str], items: Sequence[Any], *, what: str = "item"
) -> list[Any]:
    """Order ``items`` to match ``expected`` by their ids, or say precisely what is wrong.

    Batch replies used to be mapped onto their inputs by position, so a model that
    returned them out of order, or skipped one, silently graded answers against the
    wrong questions — and wrote the result to mastery.
    """
    wanted = list(expected)
    allowed = set(wanted)
    found: dict[str, Any] = {}
    duplicates: list[str] = []
    unknown: list[str] = []

    for item in items:
        key = getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None)
        if key in found:
            duplicates.append(str(key))
        elif key not in allowed:
            unknown.append(str(key))
        else:
            found[key] = item

    missing = [key for key in wanted if key not in found]
    if missing or duplicates or unknown:
        problems = []
        if missing:
            problems.append(f"no {what} for id(s) {', '.join(missing)}")
        if duplicates:
            problems.append(f"more than one {what} for id(s) {', '.join(duplicates)}")
        if unknown:
            problems.append(f"unknown id(s) {', '.join(unknown)}")
        raise ValueError(
            f"return exactly one {what} per block, each carrying the block's id verbatim "
            f"({', '.join(wanted)}): " + "; ".join(problems)
        )
    return [found[key] for key in wanted]



# A reply the provider's own parser rejected. Raised from inside `invoke`, so it
# has to be caught rather than read off the result.
PARSE_ERRORS = (PydanticValidationError, json.JSONDecodeError, ValueError,
                LengthFinishReasonError)


class _ProviderFailure(Exception):
    """The gateway, not the model. Carries the SDK's own error."""

    def __init__(self, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.cause = cause


# The envelope a chat completion arrives in. A validation error about one of
# these is the gateway's reply being unreadable; an error about anything else is
# the model's own output, whatever the object was called.
ENVELOPE_FIELDS = frozenset({"choices", "object", "created", "model", "usage", "id"})


def _ran_out_of_room(exc: BaseException) -> bool:
    """Whether the reply was cut off mid-object rather than written wrongly."""
    if isinstance(exc, LengthFinishReasonError):
        return True
    return isinstance(exc, PydanticValidationError) and any(
        error.get("type") == "json_invalid" for error in exc.errors())


def _from_the_gateway(exc: BaseException) -> bool:
    """True when the response envelope was unreadable, not the model's own output.

    Both arrive as a ``ValidationError`` out of ``invoke`` and need opposite
    treatment: an envelope the SDK could not read needs the same request sent
    again, while anything the model got wrong needs the model told about it.

    Read narrowly on purpose. Calling a model's mistake a gateway fault re-sends
    a request that cannot succeed and ends the turn — a quiz died twice in
    production that way, because an MCQ with no correct option was reported as
    ``value_error`` and everything that was not ``json_invalid`` counted as the
    gateway. The opposite mistake costs one wasted correction.
    """
    if not isinstance(exc, PydanticValidationError):
        return False
    locations = [error.get("loc") or () for error in exc.errors()]
    return bool(locations) and all(
        location and str(location[0]) in ENVELOPE_FIELDS for location in locations)


def _invoke(runnable: Any, conversation: Sequence[Any], *,
            config: Config, task: str, label: str) -> Any:
    """One model call, retrying a gateway that answered with an unreadable body."""
    delay = config.get("llm.provider_retry_delay_seconds", 0.5) if config else 0.5
    budget = config.get("llm.provider_retries", 2) if config else 2
    patience = config.llm_for(task).get("rate_limit_wait_seconds", 0) if config else 0
    waited = 0.0
    pause = RATE_LIMIT_FIRST_PAUSE
    retry = 0
    while True:
        try:
            return runnable.invoke(conversation)
        except ModelRateLimitError as exc:
            if waited >= patience:
                raise
            sleep = min(pause * random.uniform(0.5, 1.5), patience - waited)
            logger.warning("rate limited; waiting before the same call again",
                           extra={"task": task, "label": label, "waited_seconds": round(waited),
                                  "sleeping_seconds": round(sleep, 1),
                                  "error_detail": str(exc)[:MAX_ERROR_CHARS]})
            time.sleep(sleep)
            waited += sleep
            pause = min(pause * 2, RATE_LIMIT_LONGEST_PAUSE)
        except PARSE_ERRORS as exc:
            if not _from_the_gateway(exc):
                raise
            if retry >= budget:
                raise _ProviderFailure(exc) from exc
            logger.warning("the model gateway returned an unreadable reply; retrying",
                           extra={"task": task, "label": label, "retry": retry + 1,
                                  "error_detail": str(exc)[:MAX_ERROR_CHARS]})
            time.sleep(delay * (2 ** retry))
            retry += 1


RATE_LIMIT_FIRST_PAUSE = 15.0
RATE_LIMIT_LONGEST_PAUSE = 120.0

# How much more room a retry gets. Doubling rather than adding a constant: the
# shortfall is proportional to how much the model chose to reason, not fixed.
ROOM_MULTIPLIER = 2
ROOM_CEILING = 16384


def _more_room(current: Optional[int]) -> int:
    """The budget a retry gets after a reply that would not fit in the last one."""
    if not current:
        return ROOM_CEILING // 2
    return min(current * ROOM_MULTIPLIER, ROOM_CEILING)


def _with_room(client: Any, room: int) -> Any:
    """The same client with a larger completion budget, where it can take one."""
    if not hasattr(client, "bind"):
        return client
    try:
        return client.bind(max_tokens=room)
    except TypeError:
        return client


def _bind(llm: Any, schema: type[M], settings: dict[str, Any]) -> Any:
    """Schema-enforced decoding where the provider offers it.

    ``strict`` is what turns the schema from a suggestion into a constraint on the
    provider side; ``include_raw`` keeps the untouched reply, without which a
    refusal is indistinguishable from a parse failure.
    """
    if not hasattr(llm, "with_structured_output"):
        # A client with no schema support still goes through this contract: its
        # reply is recovered and validated like any other, so it cannot skip the
        # checks by being simple.
        return _PlainCall(llm)

    kwargs: dict[str, Any] = {"include_raw": True}
    if settings.get("structured_output", True):
        kwargs["method"] = "json_schema"
        kwargs["strict"] = True
    try:
        return llm.with_structured_output(schema, **kwargs)
    except TypeError:
        return llm.with_structured_output(schema)


class _PlainCall:
    """Adapts a bare chat client to the structured-result shape."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def invoke(self, messages: Sequence[Any]) -> dict[str, Any]:
        return {"raw": self._llm.invoke(messages), "parsed": None, "parsing_error": None}


def _guard_refusal(raw: Any, *, task: str, label: str, attempt: int,
                   raw_replies: list[str]) -> None:
    """Separate "I won't" and "I ran out of room" from "I got the shape wrong"."""
    if raw is None:
        return
    extra = getattr(raw, "additional_kwargs", None) or {}
    refusal = extra.get("refusal")
    if refusal:
        raise StructuredRefusal(
            f"{label}: the model refused the request.",
            task=task, label=label, attempts=attempt,
            errors=[str(refusal)], raw=raw_replies,
        )

    metadata = getattr(raw, "response_metadata", None) or {}
    reason = metadata.get("finish_reason") or metadata.get("stop_reason")
    if reason in TRUNCATION_FINISH_REASONS:
        raise StructuredTruncation(
            f"{label}: the reply hit the token limit before the object closed. "
            "Raise max_tokens for this task or ask for fewer items.",
            task=task, label=label, attempts=attempt, raw=raw_replies,
        )
    if reason in REFUSAL_FINISH_REASONS:
        raise StructuredRefusal(
            f"{label}: the provider stopped the reply ({reason}).",
            task=task, label=label, attempts=attempt, raw=raw_replies,
        )


def _feedback(error: Any, schema: type[BaseModel], raw_text: str) -> str:
    """Tell the model where it went wrong, what it sent, and what is allowed.

    Structured feedback of that shape outperforms raw diagnostics: "invalid JSON"
    invites the same mistake again, while a named field with its permitted values
    is something a model can actually fix.
    """
    header = (
        "Your previous reply did not satisfy the required schema. Fix ONLY these "
        "fields and return the corrected JSON object — same shape, no prose, no "
        "code fence."
    )
    lines = [header, ""]

    problems = _problems(error, schema)
    if not problems:
        problems = [
            ("the reply was not a JSON object matching the schema; "
             "return the object described above and nothing else.")
        ]
    lines.extend(f"- {problem}" for problem in problems[:MAX_REPORTED_ERRORS])
    if len(problems) > MAX_REPORTED_ERRORS:
        lines.append(f"- (and {len(problems) - MAX_REPORTED_ERRORS} more)")
    return "\n".join(lines)


def _problems(error: Any, schema: type[BaseModel]) -> list[str]:
    if not isinstance(error, ValidationError):
        cause = getattr(error, "__cause__", None)
        error = cause if isinstance(cause, ValidationError) else None
    if error is None:
        return []

    allowed = _allowed_values(schema)
    problems = []
    for item in error.errors(include_url=False):
        where = _path_of(item.get("loc") or ())
        field = re.sub(r"\[\d+\]", "", where)
        problems.append(f"{where}: {_observed(item)}; {_admissible(item, allowed.get(field))}")
    return problems


def _path_of(loc: Sequence[Any]) -> str:
    parts: list[str] = []
    for piece in loc:
        if isinstance(piece, int):
            parts.append(f"[{piece}]")
        else:
            parts.append(f".{piece}" if parts else str(piece))
    return "".join(parts) or "(root)"


def _observed(item: dict[str, Any]) -> str:
    if item.get("type") == "missing":
        return "missing"
    value = repr(item.get("input"))
    if len(value) > MAX_OBSERVED_CHARS:
        value = value[:MAX_OBSERVED_CHARS] + "…"
    return f"you sent {value}"


def _admissible(item: dict[str, Any], enum: Optional[list[Any]]) -> str:
    context = item.get("ctx") or {}
    kind = item.get("type", "")

    if enum:
        return "must be exactly one of: " + ", ".join(json.dumps(v) for v in enum)
    if kind == "literal_error" and context.get("expected"):
        return f"must be {context['expected']}"
    if kind in ("greater_than_equal", "greater_than"):
        return f"must be a number >= {context.get('ge', context.get('gt'))}"
    if kind in ("less_than_equal", "less_than"):
        return f"must be a number <= {context.get('le', context.get('lt'))}"
    if kind == "string_too_short":
        return f"must be at least {context.get('min_length')} characters"
    if kind == "string_too_long":
        return f"must be at most {context.get('max_length')} characters"
    if kind == "missing":
        return "this field is required"
    return item.get("msg") or "is not valid for this field"


def _allowed_values(schema: type[BaseModel]) -> dict[str, list[Any]]:
    """Enum choices per field path, read from the schema rather than restated."""
    try:
        spec = schema.model_json_schema()
    except Exception:
        return {}

    definitions = spec.get("$defs", {})
    found: dict[str, list[Any]] = {}

    def walk(node: Any, path: str) -> None:
        if not isinstance(node, dict):
            return
        ref = node.get("$ref")
        if ref and ref.startswith("#/$defs/"):
            walk(definitions.get(ref.rsplit("/", 1)[-1], {}), path)
            return
        if node.get("enum"):
            found[path] = node["enum"]
        for key, value in (node.get("properties") or {}).items():
            walk(value, f"{path}.{key}" if path else key)
        if "items" in node:
            walk(node["items"], path)
        for branch in node.get("anyOf", []) or node.get("oneOf", []):
            walk(branch, path)

    walk(spec, "")
    return found


def recover(raw_text: str, schema: type[M]) -> Optional[M]:
    """Validate a reply into ``schema``, or return ``None``.

    For a caller that owns its own transport and so cannot use
    ``call_structured`` — the vision client posts an image alongside the prompt
    — but must not grow a private JSON scraper to do it. There were two of those
    before this existed.
    """
    return _recover(raw_text, schema)


def _recover(raw_text: str, schema: type[M]) -> Optional[M]:
    """Last resort: pull an object out of a reply that wrapped it in prose.

    Scanning is balanced rather than greedy — a first-brace-to-last-brace grab
    silently changes what gets parsed when the prose contains braces of its own —
    and every candidate still has to satisfy the schema, so recovery can never
    hand back something the model did not actually produce.
    """
    for payload in _candidates(raw_text):
        try:
            return schema.model_validate(payload)
        except ValidationError:
            coerced = _coerce_root_list(payload, schema)
            if coerced is not None:
                return coerced
    return None


def _candidates(text: str):
    """Every balanced JSON value in the reply, in the order they appear."""
    if not text:
        return
    body = text.strip()
    if body.startswith("```"):
        body = body.split("```")[1] if body.count("```") >= 2 else body
        body = body.partition("\n")[2] if body.lstrip().startswith("json") else body

    index = 0
    while index < len(body):
        char = body[index]
        if char not in "{[":
            index += 1
            continue
        end = _closing_index(body, index)
        if end is None:
            index += 1
            continue
        try:
            payload = json.loads(body[index:end + 1])
        except json.JSONDecodeError:
            payload = None
        # An empty object carries nothing, and every schema with defaults accepts
        # it — recovering "{}" out of prose would reintroduce the silent-empty
        # result this contract exists to prevent.
        if payload:
            yield payload
        index = end + 1


def _closing_index(body: str, start: int) -> Optional[int]:
    """Where the value opened at ``start`` closes, ignoring braces inside strings."""
    opener = body[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = escaped = False
    for index in range(start, len(body)):
        char = body[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return index
    return None


def _coerce_root_list(payload: Any, schema: type[M]) -> Optional[M]:
    """A bare array where the schema wraps one list — accept it, then validate."""
    if not isinstance(payload, list):
        return None
    fields = [name for name, field in schema.model_fields.items()
              if getattr(field.annotation, "__origin__", None) is list]
    if len(fields) != 1:
        return None
    try:
        return schema.model_validate({fields[0]: payload})
    except ValidationError:
        return None


def _text_of(raw: Any) -> str:
    if raw is None:
        return ""
    content = getattr(raw, "content", raw)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


def schema_hint(schema: type[BaseModel]) -> str:
    """The shape a prompt asks for, rendered from the schema itself.

    Prompts used to carry hand-written JSON examples that drifted from the models
    — two were not even valid JSON, and the ingestion prompts asked for an array
    while the schema wrapped it in an object.
    """
    spec = schema.model_json_schema()
    definitions = spec.get("$defs", {})
    lines: list[str] = []

    def describe(node: dict[str, Any], path: str, depth: int) -> None:
        for name, field in (node.get("properties") or {}).items():
            resolved = _resolve(field, definitions)
            where = f"{path}.{name}" if path else name
            # An Optional field carries its description on the outer node, while the
            # type lives on the non-null branch — take each from where it is.
            description = field.get("description") or resolved.get("description")
            lines.append(f"{'  ' * depth}- {where}: {_type_of(resolved, definitions)}"
                         f"{_constraints_of(resolved) or _constraints_of(field)}"
                         f"{' — ' + description if description else ''}")
            nested = _resolve(resolved.get("items", {}), definitions) if "items" in resolved else resolved
            if nested.get("properties") and depth < 2:
                describe(nested, where + ("[]" if "items" in resolved else ""), depth + 1)

    describe(spec, "", 0)
    return "\n".join(lines)


def _resolve(node: dict[str, Any], definitions: dict[str, Any]) -> dict[str, Any]:
    ref = node.get("$ref")
    if ref and ref.startswith("#/$defs/"):
        return definitions.get(ref.rsplit("/", 1)[-1], {})
    for branch in node.get("anyOf", []):
        if branch.get("type") != "null":
            return _resolve(branch, definitions)
    return node


def _type_of(node: dict[str, Any], definitions: dict[str, Any]) -> str:
    if node.get("enum"):
        return " | ".join(json.dumps(value) for value in node["enum"])
    kind = node.get("type", "object")
    if kind == "array":
        return f"array of {_type_of(_resolve(node.get('items', {}), definitions), definitions)}"
    return kind


def _constraints_of(node: dict[str, Any]) -> str:
    bits = []
    for key, text in (("minimum", ">="), ("maximum", "<="),
                      ("minLength", "min length"), ("maxLength", "max length")):
        if key in node:
            bits.append(f"{text} {node[key]}")
    return f" ({', '.join(bits)})" if bits else ""
