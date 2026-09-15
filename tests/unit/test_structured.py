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
    align_by_id,
    StructuredRefusal,
    StructuredTruncation,
    StructuredValidationFailure,
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
