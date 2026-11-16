"""ADR-007: meta data only by default; content only on request and only masked."""

from __future__ import annotations

import logging

import pytest

from slmstrict import FallbackConfig, Schema, StrictConfig, agenerate
from slmstrict.errors import SlmStrictError
from slmstrict.fallback import (
    CloudFallback,
    FallbackConfigError,
    FallbackError,
    FallbackValidationFailed,
    RegexMasker,
)
from slmstrict.testing import FakeBackend
from tests._ladder_support import GARBAGE, Item
from tests.fallback import SECRET_KEY, FakeProvider

LOGGER = "slmstrict.fallback"
PROMPT = "Extract the order for ayse@example.com, card 4111 1111 1111 1111"
ANSWER = '{"name": "order for bob@corp.io", "count": 9}'


def _messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == LOGGER]


async def test_default_logs_meta_data_and_no_content(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    schema = Schema.of(Item)
    provider = FakeProvider(ANSWER)
    await CloudFallback(provider, masker=RegexMasker()).handle(schema, PROMPT, ())

    messages = _messages(caplog)
    assert len(messages) == 2
    request, done = messages
    assert "provider=fake" in request
    assert "model=fake-model-1" in request
    assert f"schema={schema.schema_hash}" in request
    assert f"chars={len(provider.prompts[0])}" in request
    assert "masked=True" in request
    assert "duration_s=" in done
    assert "result=ok" in done

    everything = "\n".join(record.getMessage() + str(record.args) for record in caplog.records)
    for forbidden in ("ayse@example.com", "4111", "[EMAIL]", "bob@corp.io", "order for", "Extract"):
        assert forbidden not in everything


async def test_default_logs_no_content_through_the_ladder(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    config = StrictConfig(fallback=FallbackConfig(handler=CloudFallback(FakeProvider(ANSWER))))
    await agenerate(FakeBackend([GARBAGE] * 3), Item, PROMPT, config=config)
    everything = "\n".join(record.getMessage() for record in caplog.records)
    assert "ayse@example.com" not in everything
    assert "bob@corp.io" not in everything
    assert "Extract the order" not in everything


async def test_invalid_answer_content_is_not_logged_by_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    provider = FakeProvider("leaked ayse@example.com nonsense")
    with pytest.raises(Exception, match="does not match"):
        await CloudFallback(provider).handle(Schema.of(Item), PROMPT, ())
    everything = "\n".join(record.getMessage() for record in caplog.records)
    assert "nonsense" not in everything
    assert "ayse@example.com" not in everything
    assert "result=invalid" in everything


async def test_log_content_writes_the_masked_prompt_and_answer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    handler = CloudFallback(FakeProvider(ANSWER), masker=RegexMasker(), log_content=True)
    await handler.handle(Schema.of(Item), PROMPT, ())

    messages = "\n".join(_messages(caplog))
    assert "Extract the order for [EMAIL], card [CARD]" in messages
    assert 'order for [EMAIL]", "count": 9' in messages
    assert "ayse@example.com" not in messages
    assert "4111" not in messages
    assert "bob@corp.io" not in messages


def test_log_content_without_a_masker_is_a_configuration_error() -> None:
    with pytest.raises(FallbackConfigError, match="needs a masker") as caught:
        CloudFallback(FakeProvider(ANSWER), log_content=True)
    assert isinstance(caught.value, FallbackError)
    assert isinstance(caught.value, SlmStrictError)
    assert "allow_unmasked_content_log=True" in str(caught.value)


def test_log_content_error_is_raised_at_construction_before_any_data_moves() -> None:
    provider = FakeProvider(ANSWER)
    with pytest.raises(FallbackConfigError):
        CloudFallback(provider, log_content=True, local_only=False)
    assert provider.calls == 0


async def test_the_unmasked_flag_alone_logs_no_content(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    handler = CloudFallback(FakeProvider(ANSWER), allow_unmasked_content_log=True)
    await handler.handle(Schema.of(Item), PROMPT, ())
    everything = "\n".join(_messages(caplog))
    assert "ayse@example.com" not in everything
    assert "UNMASKED" not in everything


async def test_both_flags_log_unmasked_content_and_say_so_first(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    handler = CloudFallback(FakeProvider(ANSWER), log_content=True, allow_unmasked_content_log=True)
    await handler.handle(Schema.of(Item), PROMPT, ())

    messages = _messages(caplog)
    content = [m for m in messages if PROMPT in m or "bob@corp.io" in m or "UNMASKED" in m]
    assert content[0].startswith("fallback content log is UNMASKED")
    assert any(m.startswith("fallback prompt (UNMASKED)") and PROMPT in m for m in content)
    assert any(m.startswith("fallback answer (UNMASKED)") and "bob@corp.io" in m for m in content)
    warning = next(r for r in caplog.records if r.getMessage() == content[0])
    assert warning.levelno == logging.WARNING


async def test_a_masker_keeps_the_log_masked_and_unlabelled_even_with_both_flags(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    handler = CloudFallback(
        FakeProvider(ANSWER),
        masker=RegexMasker(),
        log_content=True,
        allow_unmasked_content_log=True,
    )
    await handler.handle(Schema.of(Item), PROMPT, ())
    messages = "\n".join(_messages(caplog))
    assert "UNMASKED" not in messages
    assert "fallback prompt (masked)" in messages
    assert "ayse@example.com" not in messages


async def test_log_content_withholds_the_answer_when_the_masker_breaks_late(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Flaky:
        def __init__(self) -> None:
            self.calls = 0

        def mask(self, text: str) -> str:
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("late failure")
            return text

    caplog.set_level(logging.DEBUG)
    handler = CloudFallback(FakeProvider(ANSWER), masker=Flaky(), log_content=True)
    await handler.handle(Schema.of(Item), "p", ())
    messages = "\n".join(_messages(caplog))
    assert "answer withheld" in messages
    assert "bob@corp.io" not in messages


async def test_api_key_is_in_no_log_record_and_no_exception_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class SdkError(Exception):
        status_code = 401

    caplog.set_level(logging.DEBUG)
    provider = FakeProvider(error=SdkError(f"bad key {SECRET_KEY}"))
    with pytest.raises(FallbackError) as caught:
        await CloudFallback(provider, log_content=True, allow_unmasked_content_log=True).handle(
            Schema.of(Item), "p", ()
        )
    assert SECRET_KEY not in str(caught.value)
    assert SECRET_KEY not in repr(caught.value)
    assert SECRET_KEY not in "\n".join(
        f"{record.getMessage()} {record.args} {record.exc_text}" for record in caplog.records
    )
    assert any("result=error" in m for m in _messages(caplog))


async def test_blocked_attempts_are_logged_as_meta_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    provider = FakeProvider()
    with pytest.raises(FallbackError):
        await CloudFallback(provider, local_only=True).handle(Schema.of(Item), PROMPT, ())
    messages = "\n".join(_messages(caplog))
    assert "result=disabled_local_only" in messages
    assert "ayse@example.com" not in messages


async def test_a_provider_refusal_is_logged_as_refused_not_invalid(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    provider = FakeProvider("", finish_reason="refusal")
    with pytest.raises(FallbackValidationFailed):
        await CloudFallback(provider).handle(Schema.of(Item), "p", ())
    assert any("result=refused" in m for m in _messages(caplog))
