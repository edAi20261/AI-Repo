"""The structured-call contract: what happens when a model does not comply.

No network. Every case here was previously indistinguishable from success —
`parse_structured` returned an empty dict for a refusal, a truncation, prose and
a genuine empty result alike, and callers read that as "nothing to report".
"""

from __future__ import annotations

from typing import Literal, Optional

import pytest
from pydantic import BaseModel, Field

from shared.config import Config
from shared.structured import (
    StructuredOutputError,
    StructuredRefusal,
    StructuredTruncation,
    StructuredValidationFailure,
    align_by_id,
    call_structured,
    schema_hint,
)


class Item(BaseModel):
    id: str = Field(description="Copy the block id verbatim.")
    type: Literal["mcq", "true_false", "short_answer", "essay"]
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class Batch(BaseModel):
    items: list[Item] = Field(default_factory=list)


class Reply:
    """Stands in for an AIMessage."""

    def __init__(self, content: str = "", refusal: str | None = None,
                 finish_reason: str | None = None):
        self.content = content
        self.additional_kwargs = {"refusal": refusal} if refusal else {}
        self.response_metadata = {"finish_reason": finish_reason} if finish_reason else {}


class FakeLLM:
    """Replays a scripted sequence of provider results, one per attempt."""

    def __init__(self, *results):
        self.results = list(results)
        self.conversations: list[list] = []

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        self.kwargs = kwargs
        return self

    def invoke(self, messages):
        self.conversations.append(list(messages))
        return self.results[min(len(self.conversations) - 1, len(self.results) - 1)]


@pytest.fixture
def config() -> Config:
    return Config.load()


def parsed(**fields) -> dict:
    return {"raw": Reply("{}"), "parsed": Batch(**fields), "parsing_error": None}


def unparsed(text: str, error: Exception | None = None, **reply) -> dict:
    return {"raw": Reply(text, **reply), "parsed": None, "parsing_error": error}


def _validation_error() -> Exception:
    try:
        Batch(items=[{"id": "b1", "type": "multiple_choice", "confidence": 1.7}])
    except Exception as exc:  # pydantic.ValidationError
        return exc
    raise AssertionError("expected a validation error")


# --- the happy path ---------------------------------------------------------

def test_valid_output_is_returned_as_the_model(config):
    llm = FakeLLM(parsed(items=[{"id": "b1", "type": "mcq"}]))
    result = call_structured(Batch, [{"role": "user", "content": "go"}],
                             task="assessment", config=config, llm=llm)
    assert isinstance(result, Batch)
    assert result.items[0].id == "b1"


def test_strict_schema_is_requested(config):
    llm = FakeLLM(parsed())
    call_structured(Batch, [], task="assessment", config=config, llm=llm)
    assert llm.kwargs["strict"] is True
    assert llm.kwargs["method"] == "json_schema"
    assert llm.kwargs["include_raw"] is True


# --- retry with feedback ----------------------------------------------------

def test_a_schema_violation_is_retried_with_actionable_feedback(config):
    llm = FakeLLM(
        unparsed('{"items": [{"id": "b1", "type": "multiple_choice", "confidence": 1.7}]}',
                 _validation_error()),
        parsed(items=[{"id": "b1", "type": "mcq", "confidence": 0.9}]),
    )
    result = call_structured(Batch, [{"role": "user", "content": "grade"}],
                             task="grade", config=config, llm=llm)

    assert result.items[0].type == "mcq"
    retry = llm.conversations[1][-1]["content"]
    assert "items[0].type" in retry            # where
    assert "multiple_choice" in retry          # what it sent
    assert '"mcq"' in retry                    # what is allowed
    assert "items[0].confidence" in retry
    assert "between 0 and 1" in retry or "<= 1" in retry


def test_exhausted_retries_raise_with_the_evidence(config):
    llm = FakeLLM(unparsed("not json at all", _validation_error()))
    with pytest.raises(StructuredValidationFailure) as excinfo:
        call_structured(Batch, [], task="grade", config=config, llm=llm, attempts=2)

    error = excinfo.value
    assert error.attempts == 2
    assert error.errors and error.raw
    assert len(llm.conversations) == 2


def test_failure_is_never_an_empty_result(config):
    """The defect this contract exists for: `{}` read as 'nothing was wrong'."""
    llm = FakeLLM(unparsed("Sorry, I can't produce that.", _validation_error()))
    with pytest.raises(StructuredOutputError):
        call_structured(Batch, [], task="grade", config=config, llm=llm, attempts=1)


# --- refusal and truncation are not parse failures --------------------------

def test_a_refusal_is_not_retried(config):
    llm = FakeLLM(unparsed("", refusal="I can't help with that."))
    with pytest.raises(StructuredRefusal):
        call_structured(Batch, [], task="grade", config=config, llm=llm, attempts=3)
    assert len(llm.conversations) == 1


def test_truncation_says_what_to_change(config):
    llm = FakeLLM(unparsed('{"items": [{"id": "b1"', finish_reason="length"))
    with pytest.raises(StructuredTruncation) as excinfo:
        call_structured(Batch, [], task="assessment", config=config, llm=llm, attempts=3)
    assert "max_tokens" in str(excinfo.value)
    assert len(llm.conversations) == 1


# --- recovery ---------------------------------------------------------------

def test_an_object_wrapped_in_prose_is_recovered_and_validated(config):
    llm = FakeLLM(unparsed(
        'Sure! Here you go:\n```json\n{"items": [{"id": "b2", "type": "essay"}]}\n```\nHope that helps.',
        _validation_error()))
    result = call_structured(Batch, [], task="assessment", config=config, llm=llm)
    assert result.items[0].id == "b2"


def test_recovery_does_not_bypass_the_schema(config):
    llm = FakeLLM(unparsed('{"items": [{"id": "b1", "type": "hologram"}]}', _validation_error()))
    with pytest.raises(StructuredOutputError):
        call_structured(Batch, [], task="assessment", config=config, llm=llm, attempts=1)


def test_prose_containing_braces_does_not_confuse_recovery(config):
    """A greedy first-to-last brace scan would swallow the trailing sentence."""
    llm = FakeLLM(unparsed(
        'Note: use {} for empty. {"items": [{"id": "b3", "type": "mcq"}]} Done {see above}',
        _validation_error()))
    result = call_structured(Batch, [], task="assessment", config=config, llm=llm)
    assert result.items[0].id == "b3"


def test_a_bare_array_is_accepted_for_a_single_list_schema(config):
    """The ingestion prompts asked for an array while the schema wrapped it."""
    llm = FakeLLM(unparsed('[{"id": "b4", "type": "mcq"}]', _validation_error()))
    result = call_structured(Batch, [], task="extract_topics", config=config, llm=llm)
    assert result.items[0].id == "b4"


# --- the prompt hint --------------------------------------------------------

def test_schema_hint_describes_fields_enums_and_bounds():
    hint = schema_hint(Batch)
    assert "items" in hint
    assert '"mcq"' in hint and '"essay"' in hint
    assert ">= 0" in hint and "<= 1" in hint
    assert "Copy the block id verbatim." in hint


# --- arity and identity -----------------------------------------------------

def test_a_missing_item_is_retried_then_fails_loudly(config):
    """Position mapping used to graft a grade onto whichever answer sat at that index."""
    short = unparsed('{"items": [{"id": "b1", "type": "mcq"}]}')
    short["parsed"] = Batch(items=[Item(id="b1", type="mcq")])

    llm = FakeLLM(short)
    with pytest.raises(StructuredValidationFailure) as excinfo:
        call_structured(
            Batch, [], task="grade", config=config, llm=llm, attempts=2,
            validate=lambda batch: align_by_id(["b1", "b2"], batch.items, what="grade"),
        )
    assert "b2" in str(excinfo.value)
    assert len(llm.conversations) == 2
    retry = llm.conversations[1][-1]["content"]
    assert "b2" in retry and "verbatim" in retry


def test_a_shuffled_reply_is_put_back_in_order(config):
    shuffled = unparsed("")
    shuffled["parsed"] = Batch(items=[Item(id="b2", type="essay"), Item(id="b1", type="mcq")])
    llm = FakeLLM(shuffled)

    result = call_structured(
        Batch, [], task="grade", config=config, llm=llm,
        validate=lambda batch: align_by_id(["b1", "b2"], batch.items, what="grade"),
    )
    assert align_by_id(["b1", "b2"], result.items, what="grade")[0].id == "b1"


def test_duplicate_and_unknown_ids_are_named(config):
    with pytest.raises(ValueError) as excinfo:
        align_by_id(["b1"], [Item(id="b1", type="mcq"), Item(id="b1", type="essay")], what="grade")
    assert "more than one" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        align_by_id(["b1"], [Item(id="zz", type="mcq")], what="grade")
    assert "unknown id" in str(excinfo.value)


def test_a_prompt_string_is_one_message_not_one_per_character():
    """A bare string satisfies Sequence[Any], so list() turned a 4,000-character
    prompt into 4,000 single-character messages. It looked like it worked
    because Gemini merges consecutive user turns back into one; moving the same
    call to a provider that does not merge produced an answer about nothing,
    with no error raised anywhere.

    Both ingestion extractors passed a string.
    """
    from langchain_core.messages import HumanMessage

    from shared.structured import _as_conversation

    prompt = "Identify the skills in this lesson."
    assert _as_conversation(prompt) == [HumanMessage(content=prompt)]
    assert len(_as_conversation(prompt)) == 1

    already = [HumanMessage(content="a"), HumanMessage(content="b")]
    assert _as_conversation(already) == already


def test_a_vision_failure_says_which_model_and_why(caplog):
    """The reason is the point. The vision transport used to be a bare
    `except Exception: return None`, so an expired key, a rate limit and a
    genuinely unreadable page were indistinguishable — operators saw "needs
    review" on a whole textbook with nothing to act on."""
    import logging

    import httpx
    import pytest

    from shared.config import Config
    from shared.vision import read_image
    from ingestion.schemas import PageTranscription

    def refuse(*args, **kwargs):
        raise httpx.ConnectError("no route to host")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(httpx, "post", refuse)
    try:
        with caplog.at_level(logging.WARNING):
            assert read_image(b"png", "read this", PageTranscription,
                              config=Config.load(), label="transcribe_page") is None
    finally:
        monkeypatch.undo()

    # The structured fields, not the message: the JSON formatter emits these as
    # first-class keys, which is what an operator actually filters on.
    failures = [r for r in caplog.records if r.message == "vision call failed"]
    assert failures, "the failed call was not logged"
    assert {r.label for r in failures} == {"transcribe_page"}
    assert {r.error for r in failures} == {"ConnectError"}
    assert any(r.message == "vision failed on every model" for r in caplog.records)


# --- a reply the model ran out of room to finish ----------------------------

class _Truncated:
    """A client whose first reply is cut off mid-JSON, like a thinking model
    that spent its budget reasoning."""

    def __init__(self, then=None) -> None:
        self.calls = 0
        self.budgets: list = []
        self._then = then

    def with_structured_output(self, schema, **_kwargs):
        return self

    def bind(self, **kwargs):
        self.budgets.append(kwargs.get("max_tokens"))
        return self

    def invoke(self, _messages):
        from pydantic import ValidationError
        self.calls += 1
        if self.calls == 1:
            raise ValidationError.from_exception_data(
                "TopicList",
                [{"type": "json_invalid", "loc": (), "input": '{"topics": [{"lab',
                  "ctx": {"error": "EOF while parsing a string"}}])
        return self._then


def test_a_truncated_reply_is_retried_rather_than_raised(monkeypatch):
    """A 371-page textbook failed to ingest because the topic extractor's reply
    was cut off mid-string. The parse error was raised out of invoke() instead
    of becoming a failed attempt, so the retry that exists for exactly this
    never ran and the whole ingest died at the topics stage."""
    from shared import structured

    llm = _Truncated(then=Batch(items=[]))
    out = structured.call_structured(
        Batch, "list the topics", task="extract_topics",
        config=Config.load(), llm=llm, attempts=2)

    assert out.items == []
    assert llm.calls == 2


def test_a_truncated_reply_is_retried_with_more_room_not_more_words(monkeypatch):
    """Feeding "invalid JSON" back to a model that ran out of tokens asks it to
    fix something it did not get wrong. What it needs is a larger budget."""
    from shared import structured

    llm = _Truncated(then=Batch(items=[]))
    structured.call_structured(
        Batch, "list the topics", task="extract_topics",
        config=Config.load(), llm=llm, attempts=2)

    assert llm.budgets, "the retry must rebind the model with a bigger budget"
    assert llm.budgets[-1] > Config.load().llm_for("extract_topics")["max_tokens"]


# --- the provider, not the model --------------------------------------------

class _Choiceless(BaseModel):
    """Stands in for the SDK's own response model."""
    choices: list[str]


def _provider_error() -> Exception:
    """What the OpenAI SDK raises when the gateway returns a body it cannot read."""
    try:
        _Choiceless()
    except Exception as exc:
        return exc
    raise AssertionError("expected a validation error")


class ExplodingLLM(FakeLLM):
    """Raises on its first ``explosions`` calls, then replays its results."""

    def __init__(self, *results, explosions: int, error: Exception):
        super().__init__(*results)
        self.explosions = explosions
        self.error = error
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls <= self.explosions:
            raise self.error
        return super().invoke(messages)


def test_a_gateway_hiccup_is_retried_rather_than_read_as_a_schema_mistake(config):
    """OpenRouter can answer with a body the SDK cannot read. That raised out of
    `invoke`, was counted as the model getting the shape wrong, and killed the
    turn on the second one — a student saw "the tutor could not answer"."""
    llm = ExplodingLLM(parsed(items=[{"id": "b1", "type": "mcq"}]),
                       explosions=1, error=_provider_error())
    result = call_structured(Batch, [{"role": "user", "content": "go"}],
                             task="assessment", config=config, llm=llm)

    assert result.items[0].id == "b1"
    assert llm.calls == 2
    assert len(llm.conversations) == 1          # the prompt was never rewritten
    assert llm.conversations[0][-1]["content"] == "go"


def test_a_gateway_that_keeps_failing_says_it_was_the_gateway(config):
    from shared.structured import StructuredProviderError

    llm = ExplodingLLM(parsed(), explosions=99, error=_provider_error())
    with pytest.raises(StructuredProviderError) as excinfo:
        call_structured(Batch, [], task="assessment", config=config, llm=llm)

    assert "choices" in " ".join(excinfo.value.errors)


def test_a_schema_mistake_still_reaches_the_model_as_feedback(config):
    """The classifier must not swallow a real schema violation."""
    llm = FakeLLM(
        unparsed('{"items": [{"id": "b1", "type": "multiple_choice"}]}', _validation_error()),
        parsed(items=[{"id": "b1", "type": "mcq"}]),
    )
    call_structured(Batch, [{"role": "user", "content": "grade"}],
                    task="grade", config=config, llm=llm)
    assert "multiple_choice" in llm.conversations[1][-1]["content"]


def test_an_unreadable_reply_is_logged_with_its_reason(config, caplog):
    """`error: ValidationError` with no field and no raw text is not diagnosable:
    a week of production failures said nothing about what was wrong."""
    import logging

    llm = FakeLLM(unparsed("nope", _validation_error()))
    with caplog.at_level(logging.WARNING, logger="shared.structured"):
        with pytest.raises(StructuredOutputError):
            call_structured(Batch, [], task="grade", config=config, llm=llm, attempts=1)

    record = next(r for r in caplog.records if r.name == "shared.structured")
    assert getattr(record, "error_detail", "")


def _rejected_by_our_own_validator() -> Exception:
    """What a schema's own validator raises — 'value_error', not 'json_invalid'."""
    from pydantic import field_validator

    class Item(BaseModel):
        kind: str

        @field_validator("kind")
        @classmethod
        def _known(cls, v):
            raise ValueError("an mcq needs correct_indices naming at least one option")

    try:
        Item(kind="mcq")
    except Exception as exc:
        return exc
    raise AssertionError("expected a validation error")


def test_a_rule_the_schema_enforces_is_the_models_mistake_not_the_gateways(config):
    """A quiz turn died twice in production with 'the tutor could not answer'.
    The model had returned an MCQ with no correct option — exactly what the
    retry exists to correct — but the classifier read anything that was not
    'json_invalid' as an unreadable envelope, re-sent the same request, and
    gave up."""
    llm = ExplodingLLM(parsed(items=[{"id": "b1", "type": "mcq"}]),
                       explosions=1, error=_rejected_by_our_own_validator())

    result = call_structured(Batch, [{"role": "user", "content": "quiz me"}],
                             task="assessment", config=config, llm=llm)

    assert result.items[0].id == "b1"
    assert "correct_indices" in llm.conversations[-1][-1]["content"]


def test_an_unreadable_envelope_is_still_the_gateway(config):
    from shared.structured import StructuredProviderError

    llm = ExplodingLLM(parsed(), explosions=99, error=_provider_error())
    with pytest.raises(StructuredProviderError):
        call_structured(Batch, [], task="assessment", config=config, llm=llm)
