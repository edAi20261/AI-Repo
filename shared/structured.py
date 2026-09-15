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
import re
from collections.abc import Callable, Sequence
from typing import Any, Optional, TypeVar

from pydantic import BaseModel, ValidationError

from .config import Config
from .llm import get_llm

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)

DEFAULT_ATTEMPTS = 2
MAX_REPORTED_ERRORS = 8
MAX_OBSERVED_CHARS = 80

REFUSAL_FINISH_REASONS = frozenset({"content_filter"})
TRUNCATION_FINISH_REASONS = frozenset({"length", "max_tokens"})


class StructuredOutputError(RuntimeError):
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


def call_structured(
    schema: type[M],
    messages: Sequence[Any],
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

    conversation = list(messages)
    failures: list[str] = []
    raw_replies: list[str] = []

    for attempt in range(1, budget + 1):
        result = runnable.invoke(conversation)
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
                              "reason": feedback.splitlines()[0] if feedback else ""})
        if attempt < budget:
            conversation = _with_feedback(messages, raw_text, feedback)

    raise StructuredValidationFailure(
        f"{name}: the model did not return valid output after {budget} attempts.",
        task=task, label=name, attempts=budget, errors=failures, raw=raw_replies,
    )


def _with_feedback(messages: Sequence[Any], raw_text: str, feedback: str) -> list[Any]:
    """The original request, the model's failed reply, and what to fix."""
    return list(messages) + [
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
    """Validate a model's raw reply into ``schema``, or return ``None``.

    For callers that own their own transport (the vision client posts an image
    alongside the prompt) and so cannot use :func:`call_structured`, but must not
    grow a fourth private JSON scraper to do it.
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
