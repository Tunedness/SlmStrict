"""ADR-003: the cloud fallback never runs unless it is configured and local_only is off."""

import pytest

from slmstrict import (
    FallbackConfig,
    FallbackStatus,
    Step,
    StrictConfig,
    ValidationFailed,
    agenerate,
)
from slmstrict.hooks import FallbackEvent
from slmstrict.testing import FakeBackend
from tests._ladder_support import GARBAGE, VALID, Item, RecordingObserver, SpyHandler


def _fallback_event(observer: RecordingObserver) -> FallbackEvent:
    return next(e for e in observer.events if isinstance(e, FallbackEvent))


async def test_local_only_never_calls_a_configured_handler() -> None:
    handler = SpyHandler()
    observer = RecordingObserver()
    config = StrictConfig(
        local_only=True, fallback=FallbackConfig(handler=handler), observer=observer
    )
    with pytest.raises(ValidationFailed, match="local_only"):
        await agenerate(FakeBackend([GARBAGE] * 3), Item, "p", config=config)
    assert handler.calls == 0
    assert _fallback_event(observer).status is FallbackStatus.DISABLED_LOCAL_ONLY


async def test_failure_under_local_only_carries_all_attempts() -> None:
    config = StrictConfig(local_only=True, fallback=FallbackConfig(handler=SpyHandler()))
    with pytest.raises(ValidationFailed) as caught:
        await agenerate(FakeBackend([GARBAGE] * 3), Item, "p", config=config)
    assert len(caught.value.attempts) == 3
    assert all(a.step is not Step.FALLBACK for a in caught.value.attempts)


async def test_unconfigured_fallback_is_never_called() -> None:
    handler = SpyHandler()  # exists, but is not wired into the config
    observer = RecordingObserver()
    for config in (
        None,
        StrictConfig(observer=observer),
        StrictConfig(fallback=FallbackConfig(), observer=observer),  # section without a handler
    ):
        with pytest.raises(ValidationFailed, match="no cloud fallback is configured"):
            await agenerate(FakeBackend([GARBAGE] * 3), Item, "p", config=config)
    assert handler.calls == 0
    assert _fallback_event(observer).status is FallbackStatus.NOT_CONFIGURED


async def test_local_only_wins_over_every_other_setting() -> None:
    handler = SpyHandler()
    config = StrictConfig(local_only=True, max_reask=0, fallback=FallbackConfig(handler=handler))
    backend = FakeBackend([GARBAGE])
    with pytest.raises(ValidationFailed):
        await agenerate(backend, Item, "p", config=config)
    assert handler.calls == 0
    assert len(backend.calls) == 1


async def test_local_success_never_touches_the_handler() -> None:
    handler = SpyHandler()
    config = StrictConfig(local_only=True, fallback=FallbackConfig(handler=handler))
    result = await agenerate(FakeBackend([VALID]), Item, "p", config=config)
    assert not result.used_fallback
    assert handler.calls == 0


async def test_handler_is_called_once_when_allowed() -> None:
    # control for the tests above: the same setup without the lock does reach the handler
    handler = SpyHandler()
    config = StrictConfig(fallback=FallbackConfig(handler=handler))
    result = await agenerate(FakeBackend([GARBAGE] * 3), Item, "p", config=config)
    assert result.used_fallback
    assert handler.calls == 1
