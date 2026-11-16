"""CloudFallback: the fixed send order, ADR-003 locks and validation of the answer."""

from __future__ import annotations

from typing import Any

import pytest

from slmstrict import (
    FallbackConfig,
    FallbackStatus,
    Schema,
    Step,
    StrictConfig,
    TokenUsage,
    ValidationFailed,
    agenerate,
)
from slmstrict.errors import SlmStrictError
from slmstrict.fallback import (
    CloudFallback,
    FallbackDisabledError,
    FallbackError,
    FallbackValidationFailed,
    RegexMasker,
)
from slmstrict.hooks import FallbackEvent, FallbackHandler, FallbackResult
from slmstrict.testing import FakeBackend
from tests._ladder_support import GARBAGE, ITEM_JSON_SCHEMA, Item, RecordingObserver
from tests.fallback import SECRET_KEY, VALID_ANSWER, FakeProvider

USAGE = TokenUsage(input_tokens=120, output_tokens=15, cached_input_tokens=100)


def _event(observer: RecordingObserver) -> FallbackEvent:
    return next(e for e in observer.events if isinstance(e, FallbackEvent))


def test_cloud_fallback_is_a_fallback_handler() -> None:
    assert isinstance(CloudFallback(FakeProvider()), FallbackHandler)


# -- local_only: both layers ---------------------------------------------------------------


async def test_router_local_only_never_calls_the_provider() -> None:
    provider = FakeProvider()
    handler = CloudFallback(provider, local_only=True)
    with pytest.raises(FallbackDisabledError, match="local_only"):
        await handler.handle(Schema.of(Item), "p", ())
    assert provider.calls == 0


async def test_router_local_only_does_not_even_run_the_masker() -> None:
    class Spy:
        calls = 0

        def mask(self, text: str) -> str:
            self.calls += 1
            return text

    masker = Spy()
    handler = CloudFallback(FakeProvider(), masker=masker, local_only=True)
    with pytest.raises(FallbackDisabledError):
        await handler.handle(Schema.of(Item), "p", ())
    assert masker.calls == 0


async def test_ladder_local_only_never_reaches_the_provider() -> None:
    provider = FakeProvider()
    observer = RecordingObserver()
    config = StrictConfig(
        local_only=True,
        fallback=FallbackConfig(handler=CloudFallback(provider)),
        observer=observer,
    )
    with pytest.raises(ValidationFailed, match="local_only"):
        await agenerate(FakeBackend([GARBAGE] * 3), Item, "p", config=config)
    assert provider.calls == 0
    assert _event(observer).status is FallbackStatus.DISABLED_LOCAL_ONLY


async def test_no_provider_call_when_the_local_rungs_succeed() -> None:
    provider = FakeProvider()
    config = StrictConfig(fallback=FallbackConfig(handler=CloudFallback(provider)))
    result = await agenerate(
        FakeBackend(['{"name": "apple", "count": 3}']), Item, "p", config=config
    )
    assert provider.calls == 0
    assert result.value == Item(name="apple", count=3)


def test_local_only_has_the_sensible_default() -> None:
    assert "local_only=False" in repr(CloudFallback(FakeProvider()))


# -- masking -------------------------------------------------------------------------------


async def test_masked_prompt_reaches_the_provider_and_the_original_does_not() -> None:
    provider = FakeProvider()
    handler = CloudFallback(provider, masker=RegexMasker())
    prompt = "Extract from: mail ayse@example.com, TC 10000000146"
    await handler.handle(Schema.of(Item), prompt, ())
    assert provider.prompts == ["Extract from: mail [EMAIL], TC [TCKN]"]
    assert "ayse@example.com" not in provider.prompts[0]
    assert "10000000146" not in provider.prompts[0]


async def test_without_a_masker_the_prompt_goes_out_unchanged() -> None:
    provider = FakeProvider()
    await CloudFallback(provider).handle(Schema.of(Item), "mail ayse@example.com", ())
    assert provider.prompts == ["mail ayse@example.com"]


async def test_masking_happens_through_the_ladder_too() -> None:
    provider = FakeProvider()
    config = StrictConfig(
        fallback=FallbackConfig(handler=CloudFallback(provider, masker=RegexMasker()))
    )
    await agenerate(FakeBackend([GARBAGE] * 3), Item, "mail ayse@example.com", config=config)
    assert provider.prompts == ["mail [EMAIL]"]


async def test_failed_local_attempts_are_not_sent() -> None:
    provider = FakeProvider()
    config = StrictConfig(fallback=FallbackConfig(handler=CloudFallback(provider)))
    leaky = "garbage with ayse@example.com inside"
    await agenerate(FakeBackend([leaky] * 3), Item, "p", config=config)
    assert provider.prompts == ["p"]


async def test_a_masker_that_raises_blocks_the_send() -> None:
    class Broken:
        def mask(self, text: str) -> str:
            raise RuntimeError(f"cannot mask {text}")

    provider = FakeProvider()
    handler = CloudFallback(provider, masker=Broken())
    with pytest.raises(FallbackError, match="nothing was sent") as caught:
        await handler.handle(Schema.of(Item), "secret prompt", ())
    assert provider.calls == 0
    assert "secret prompt" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None  # the masker's exception may quote the prompt


async def test_a_masker_that_returns_a_non_string_blocks_the_send() -> None:
    class Wrong:
        def mask(self, text: str) -> Any:
            return None

    provider = FakeProvider()
    with pytest.raises(FallbackError, match="nothing was sent"):
        await CloudFallback(provider, masker=Wrong()).handle(Schema.of(Item), "p", ())
    assert provider.calls == 0


# -- result and validation -----------------------------------------------------------------


async def test_valid_answer_becomes_a_fallback_result() -> None:
    provider = FakeProvider(usage=USAGE)
    result = await CloudFallback(provider).handle(Schema.of(Item), "p", ())
    assert isinstance(result, FallbackResult)
    assert result.value == Item(name="cloud", count=9)
    assert result.path == ()
    assert [a.step for a in result.attempts] == [Step.FALLBACK]
    assert result.attempts[0].succeeded
    assert result.attempts[0].raw_output == VALID_ANSWER
    assert (result.usage, result.provider, result.model) == (USAGE, "fake", "fake-model-1")


async def test_json_schema_input_yields_a_dict() -> None:
    result = await CloudFallback(FakeProvider()).handle(Schema.of(ITEM_JSON_SCHEMA), "p", ())
    assert result.value == {"name": "cloud", "count": 9}


async def test_ladder_reports_the_fallback_with_usage_provider_and_model() -> None:
    observer = RecordingObserver()
    config = StrictConfig(
        fallback=FallbackConfig(handler=CloudFallback(FakeProvider(usage=USAGE))),
        observer=observer,
    )
    result = await agenerate(FakeBackend([GARBAGE] * 3), Item, "p", config=config)
    assert result.used_fallback
    assert result.path[-1] is Step.FALLBACK
    event = _event(observer)
    assert event.status is FallbackStatus.USED
    assert (event.usage, event.provider, event.model) == (USAGE, "fake", "fake-model-1")


@pytest.mark.parametrize(
    "answer",
    [
        '{"name": "cloud"}',  # missing field
        '{"name": "cloud", "count": "nine"}',  # wrong type
        "I cannot help with that.",  # not JSON
        "",  # nothing at all
    ],
)
async def test_invalid_answer_is_reported_as_validation_failed(answer: str) -> None:
    provider = FakeProvider(answer, usage=USAGE)
    with pytest.raises(FallbackValidationFailed) as caught:
        await CloudFallback(provider).handle(Schema.of(Item), "p", ())
    error = caught.value
    assert isinstance(error, ValidationFailed)
    assert error.issues
    assert error.last_output == answer
    assert len(error.attempts) == 1
    assert error.attempts[0].step is Step.FALLBACK
    assert not error.attempts[0].succeeded
    assert (error.usage, error.provider, error.model) == (USAGE, "fake", "fake-model-1")
    assert answer not in str(error) or answer == ""


async def test_invalid_answer_through_the_ladder_is_validation_failed() -> None:
    observer = RecordingObserver()
    config = StrictConfig(
        fallback=FallbackConfig(handler=CloudFallback(FakeProvider('{"name": 1}'))),
        observer=observer,
    )
    with pytest.raises(ValidationFailed):
        await agenerate(FakeBackend([GARBAGE] * 3), Item, "p", config=config)
    assert _event(observer).status is FallbackStatus.FAILED
    assert _event(observer).error_type == "FallbackValidationFailed"


# -- provider failures ---------------------------------------------------------------------


class _SdkError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


async def test_provider_error_is_wrapped_without_the_sdk_message() -> None:
    provider = FakeProvider(
        error=_SdkError(f"Incorrect API key provided: {SECRET_KEY} for prompt 'hello'", 401)
    )
    with pytest.raises(FallbackError) as caught:
        await CloudFallback(provider).handle(Schema.of(Item), "hello", ())
    message = str(caught.value)
    assert SECRET_KEY not in message
    assert "hello" not in message
    assert "_SdkError" in message
    assert "401" in message
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None  # the SDK exception is not reachable at all


async def test_provider_error_through_the_ladder_is_a_slmstrict_error() -> None:
    observer = RecordingObserver()
    config = StrictConfig(
        fallback=FallbackConfig(handler=CloudFallback(FakeProvider(error=_SdkError("boom", 500)))),
        observer=observer,
    )
    with pytest.raises(SlmStrictError):
        await agenerate(FakeBackend([GARBAGE] * 3), Item, "p", config=config)
    assert _event(observer).status is FallbackStatus.FAILED


async def test_fallback_errors_from_a_provider_pass_through_unchanged() -> None:
    original = FallbackError("the openai request failed (RateLimitError, status 429)")
    with pytest.raises(FallbackError) as caught:
        await CloudFallback(FakeProvider(error=original)).handle(Schema.of(Item), "p", ())
    assert caught.value is original


async def test_non_exception_cancellation_is_not_swallowed() -> None:
    import asyncio

    provider = FakeProvider(error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await CloudFallback(provider).handle(Schema.of(Item), "p", ())


# -- construction --------------------------------------------------------------------------


def test_constructor_validates_its_arguments() -> None:
    with pytest.raises(TypeError, match="CloudProvider"):
        CloudFallback(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Masker"):
        CloudFallback(FakeProvider(), masker=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bool"):
        CloudFallback(FakeProvider(), local_only="yes")  # type: ignore[arg-type]


def test_repr_never_holds_prompts_or_keys() -> None:
    text = repr(CloudFallback(FakeProvider(), masker=RegexMasker()))
    assert text == (
        "CloudFallback(provider='fake', model='fake-model-1', masker=True, "
        "local_only=False, log_content=False, allow_unmasked_content_log=False)"
    )
