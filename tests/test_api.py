import asyncio
import warnings

import pytest

import slmstrict
from slmstrict import (
    BackendError,
    FallbackConfig,
    Schema,
    SchemaError,
    Step,
    StrictConfig,
    ValidationFailed,
    agenerate,
    generate,
)
from slmstrict.testing import FakeBackend
from tests._ladder_support import (
    FENCED,
    GARBAGE,
    ITEM_JSON_SCHEMA,
    VALID,
    GrammarBackend,
    Item,
    SpyHandler,
)

SCRIPTS = {
    "grammar_ok": ([VALID], True),
    "repaired": ([FENCED], True),
    "reask": ([GARBAGE, GARBAGE, VALID], False),
}


@pytest.mark.parametrize("name", sorted(SCRIPTS))
async def test_generate_and_agenerate_return_equal_results(name: str) -> None:
    outputs, grammar = SCRIPTS[name]

    def backend() -> FakeBackend:
        return GrammarBackend(outputs) if grammar else FakeBackend(outputs)

    expected = await agenerate(backend(), Item, "prompt")
    # generate() starts its own loop, so it runs off the test's loop
    actual = await asyncio.to_thread(generate, backend(), Item, "prompt")
    assert actual == expected
    assert actual.path == expected.path
    assert actual.attempts == expected.attempts


def test_generate_works_in_plain_sync_code() -> None:
    result = generate(FakeBackend([VALID]), Item, "p")
    assert result.value == Item(name="apple", count=3)
    assert result.path == ()


def test_generate_and_agenerate_raise_the_same_failure() -> None:
    with pytest.raises(ValidationFailed) as sync_error:
        generate(FakeBackend([GARBAGE] * 3), Item, "p")
    with pytest.raises(ValidationFailed) as async_error:
        asyncio.run(agenerate(FakeBackend([GARBAGE] * 3), Item, "p"))
    assert sync_error.value.attempts == async_error.value.attempts
    assert str(sync_error.value) == str(async_error.value)


async def test_generate_inside_a_running_loop_fails_clearly() -> None:
    backend = FakeBackend([VALID])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an un-awaited coroutine would surface as a warning
        with pytest.raises(RuntimeError, match=r"await agenerate"):
            generate(backend, Item, "p")
    assert backend.calls == []


async def test_schema_input_forms() -> None:
    from_model = await agenerate(FakeBackend([VALID]), Item, "p")
    from_schema = await agenerate(FakeBackend([VALID]), Schema.of(Item), "p")
    from_json = await agenerate(FakeBackend([VALID]), ITEM_JSON_SCHEMA, "p")
    assert from_model.value == from_schema.value == Item(name="apple", count=3)
    assert from_json.value == {"name": "apple", "count": 3}


async def test_config_is_honoured() -> None:
    backend = FakeBackend([GARBAGE] * 10)
    config = StrictConfig(max_reask=1)
    with pytest.raises(ValidationFailed):
        await agenerate(backend, Item, "p", config=config)
    assert len(backend.calls) == 2


async def test_fallback_through_the_public_api() -> None:
    handler = SpyHandler()
    config = StrictConfig(fallback=FallbackConfig(handler=handler))
    result = await agenerate(FakeBackend([GARBAGE] * 3), Item, "p", config=config)
    assert result.path[-1] is Step.FALLBACK
    assert handler.calls == 1


async def test_errors_surface_unchanged() -> None:
    with pytest.raises(BackendError, match="boom"):
        await agenerate(FakeBackend([BackendError("boom")]), Item, "p")
    with pytest.raises(SchemaError):
        await agenerate(FakeBackend([VALID]), {"type": "array"}, "p")
    with pytest.raises(SchemaError):
        await agenerate(FakeBackend([VALID]), 42, "p")  # type: ignore[call-overload]


async def test_argument_type_checks() -> None:
    with pytest.raises(TypeError, match="Backend"):
        await agenerate(object(), Item, "p")  # type: ignore[call-overload]
    with pytest.raises(TypeError, match="prompt"):
        await agenerate(FakeBackend([VALID]), Item, 5)  # type: ignore[call-overload]
    with pytest.raises(TypeError, match="StrictConfig"):
        await agenerate(FakeBackend([VALID]), Item, "p", config={})  # type: ignore[call-overload]


def test_public_exports() -> None:
    for name in (
        "generate",
        "agenerate",
        "FallbackHandler",
        "LadderObserver",
        "BaseLadderObserver",
        "FallbackResult",
        "FallbackStatus",
        "TokenUsage",
        "AttemptStarted",
        "StepFinished",
        "FallbackEvent",
        "LadderFinished",
    ):
        assert name in slmstrict.__all__
        assert hasattr(slmstrict, name)
